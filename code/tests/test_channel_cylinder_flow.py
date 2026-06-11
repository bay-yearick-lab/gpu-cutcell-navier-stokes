"""Tests for the sparse channel-cylinder engine (channel_cylinder_flow)."""

from __future__ import annotations

import os
import unittest

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from math import pi

import numpy as np

from channel_cylinder_flow import (
    ChannelCylinderSolver,
    ChannelGeometry,
    arc_quadrature,
    assemble_channel_operators,
    basis_derivative_banded,
    basis_values_banded,
    disk_polar_quadrature,
    make_scaled_basis,
    marsden_monomial_coefficients,
    positive_channel_volume_quadrature,
    sparse_marsden_extension,
    sparse_tensor_eval,
)


class TestSparseBasis(unittest.TestCase):
    def test_derivative_matches_finite_differences(self):
        b = make_scaled_basis(7, 3, 2.2)
        x = np.linspace(0.01, 2.19, 31)
        dv, dc = basis_derivative_banded(b, x)
        d = 1.0e-7
        vp, _ = basis_values_banded(b, x + d)
        vm, _ = basis_values_banded(b, x - d)
        self.assertLess(float(np.abs((vp - vm) / (2 * d) - dv).max()), 1.0e-5)

    def test_partition_of_unity_and_gradient_null(self):
        bx = make_scaled_basis(11, 3, 2.2)
        by = make_scaled_basis(5, 3, 0.41)
        phi, dphix, dphiy = sparse_tensor_eval(bx, by, np.array([0.3, 2.0]), np.array([0.1, 0.39]))
        self.assertLess(float(np.abs(np.asarray(phi.sum(axis=1)) - 1.0).max()), 1.0e-13)
        self.assertLess(float(np.abs(np.asarray(dphix.sum(axis=1))).max()), 1.0e-12)
        self.assertLess(float(np.abs(np.asarray(dphiy.sum(axis=1))).max()), 1.0e-12)

    def test_marsden_reproduces_monomials(self):
        b = make_scaled_basis(9, 4, 2.2)
        lam = marsden_monomial_coefficients(b, 4)
        xs = np.linspace(0.0, 2.2, 41)
        v, c = basis_values_banded(b, xs)
        for r in range(5):
            rec = np.einsum("ik,ik->i", v, lam[r][c])
            self.assertLess(float(np.abs(rec - xs**r).max()), 1.0e-12)


class TestChannelGeometry(unittest.TestCase):
    def test_quadrature_measures(self):
        geo = ChannelGeometry()
        bx = make_scaled_basis(44, 3, geo.length)
        by = make_scaled_basis(8, 3, geo.height)
        _, _, w = disk_polar_quadrature(geo, bx.knots, by.knots, 10)
        self.assertLess(abs(float(w.sum()) - pi * geo.radius**2), 1.0e-12)
        _, _, aw, anx, any_ = arc_quadrature(geo, bx.knots, by.knots, 10)
        self.assertLess(abs(float(aw.sum()) - 2 * pi * geo.radius), 1.0e-12)
        self.assertLess(float(np.abs(anx**2 + any_**2 - 1.0).max()), 1.0e-13)
        # The positive rule is capture-accurate (convection, loads, norms); its
        # vertical-panel tangency floor is documented, the operators use the
        # rectangle-minus-hole rule instead.
        _, _, vw, _ = positive_channel_volume_quadrature(bx, by, geo, 5)
        fluid = geo.length * geo.height - pi * geo.radius**2
        self.assertLess(abs(float(vw.sum()) - fluid), 1.0e-5)


class TestChannelAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = assemble_channel_operators(p=2, nx_elements=66, ny_elements=12)

    def test_gauss_identity_constant_pressure(self):
        ops = self.ops
        ones = np.ones(ops.E_p.shape[1])
        gx1 = ops.Gx @ ones
        gy1 = ops.Gy @ ones
        mask = ops.fb_x > ops.geo.length - 1.0e-9
        out_x = np.asarray(ops.E_u.T @ (ops.phi_fb.T @ (ops.fb_w * mask * ops.fb_nx))).ravel()
        out_y = np.asarray(ops.E_u.T @ (ops.phi_fb.T @ (ops.fb_w * mask * ops.fb_ny))).ravel()
        self.assertLess(float(np.abs(gx1 + out_x).max()), 1.0e-10)
        self.assertLess(float(np.abs(gy1 + out_y).max()), 1.0e-10)

    def test_blocks_symmetric(self):
        self.assertEqual(float(np.abs(self.ops.Diff - self.ops.Diff.T).max()), 0.0)
        self.assertEqual(float(np.abs(self.ops.H_u - self.ops.H_u.T).max()), 0.0)

    def test_extension_polynomial_exactness(self):
        # E reproduces every tensor monomial of total degree <= p on the rows
        # it covers: E @ (full coefficients at interior rows) == full there.
        ops = self.ops
        p = ops.p
        lamx = marsden_monomial_coefficients(ops.basis_ux, p)
        lamy = marsden_monomial_coefficients(ops.basis_uy, p)
        gamma_pos = np.asarray(np.abs(ops.E_u).sum(axis=1)).ravel() > 0
        E = ops.E_u.tocsc()
        n_red = E.shape[1]
        interior_rows = np.zeros(n_red, dtype=int)
        for j in range(n_red):
            col = E.getcol(j)
            rows = col.nonzero()[0]
            vals = col.data
            unit = rows[np.isclose(vals, 1.0)]
            interior_rows[j] = int(unit[0]) if unit.size else int(rows[0])
        for r in range(p + 1):
            for s in range(p + 1 - r):
                full = np.kron(lamy[s], lamx[r])
                rec = ops.E_u @ full[interior_rows]
                err = float(np.abs((rec - full)[gamma_pos]).max())
                self.assertLess(err, 1.0e-9, msg=f"monomial x^{r} y^{s}")


class TestShortChannelRun(unittest.TestCase):
    def test_twenty_steps_stable_and_constrained(self):
        ops = assemble_channel_operators(p=2, nx_elements=66, ny_elements=12)
        solver = ChannelCylinderSolver(ops, dt=2.0e-3, t_ramp=0.5)
        for _ in range(20):
            solver.step()
        ke = solver.kinetic_energy()
        self.assertTrue(np.isfinite(ke))
        self.assertGreater(ke, 0.0)
        self.assertLess(solver.divergence_residual(), 1.0e-11)
        fx, _fy = solver.forces()
        self.assertGreater(fx, 0.0)  # drag pushes downstream




class TestRemedyExtensions(unittest.TestCase):
    def test_merge_preserves_constants_delete_breaks_partition(self):
        import numpy as np
        from channel_cylinder_flow import (
            sparse_delete_extension,
            sparse_merge_extension,
        )

        bx = make_scaled_basis(44, 2, 2.2)
        by = make_scaled_basis(8, 2, 0.41)
        rng = np.random.default_rng(1)
        gamma = rng.uniform(0.0, 1.0, bx.n_basis * by.n_basis)
        gamma[rng.choice(gamma.size, 30, replace=False)] = 1.0e-3  # small ring
        E_m, interior, small, _ = sparse_merge_extension(bx, by, gamma, 0.005, 2)
        ones = np.ones(E_m.shape[1])
        rec = E_m @ ones
        self.assertLess(float(np.abs(rec[interior] - 1.0).max()), 1.0e-14)
        self.assertLess(float(np.abs(rec[small] - 1.0).max()), 1.0e-14)
        E_d, interior_d, small_d, _ = sparse_delete_extension(bx, by, gamma, 0.005, 2)
        rec_d = E_d @ np.ones(E_d.shape[1])
        self.assertEqual(float(np.abs(rec_d[small_d]).max()), 0.0)

    def test_delete_breaks_gauss_identity_merge_keeps_it(self):
        import numpy as np
        from channel_cylinder_flow import assemble_channel_operators

        defects = {}
        for kind in ("merge", "delete"):
            ops = assemble_channel_operators(
                p=2, nx_elements=66, ny_elements=12, extension_kind=kind
            )
            ones = np.ones(ops.E_p.shape[1])
            mask = ops.fb_x > ops.geo.length - 1.0e-9
            out_x = np.asarray(
                ops.E_u.T @ (ops.phi_fb.T @ (ops.fb_w * mask * ops.fb_nx))
            ).ravel()
            defects[kind] = float(np.abs(ops.Gx @ ones + out_x).max())
        self.assertLess(defects["merge"], 1.0e-10)
        self.assertGreater(defects["delete"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
