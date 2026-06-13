from __future__ import annotations

import math
import unittest

import numpy as np

from cut_assembly import basis_value_table, element_support
from cut_spline_sbp import make_open_uniform_basis
from oblique_cut_sbp import (
    HalfPlaneCut,
    assemble_oblique_cut_matrices,
    assemble_oblique_cut_matrices_reference,
)


class BasisTableTests(unittest.TestCase):
    def test_tables_match_closures(self) -> None:
        rng = np.random.default_rng(7)
        for degree in (1, 2, 3, 5):
            basis = make_open_uniform_basis(n_elements=7, degree=degree)
            x = np.concatenate([np.linspace(0.0, 1.0, 101), rng.uniform(0.0, 1.0, 200)])
            ref_v = np.vstack([np.nan_to_num(f(x), nan=0.0) for f in basis.basis])
            ref_d = np.vstack([np.nan_to_num(f(x), nan=0.0) for f in basis.dbasis])
            self.assertLess(float(np.max(np.abs(basis_value_table(basis, x, 0) - ref_v))), 1.0e-13)
            scale = degree * basis.n_elements
            self.assertLess(float(np.max(np.abs(basis_value_table(basis, x, 1) - ref_d))), 1.0e-10 * scale)

    def test_element_support_is_sharp(self) -> None:
        basis = make_open_uniform_basis(n_elements=9, degree=3)
        for e in (0, 4, 8):
            lo, hi = element_support(basis, e)
            xs = np.linspace(e / 9 + 1e-9, (e + 1) / 9 - 1e-9, 5)
            vals = basis_value_table(basis, xs, 0)
            nz = np.where(np.max(np.abs(vals), axis=1) > 1.0e-14)[0]
            self.assertGreaterEqual(int(nz.min()), lo)
            self.assertLess(int(nz.max()), hi)
            self.assertEqual(hi - lo, basis.degree + 1)


class FastObliqueAssemblyTests(unittest.TestCase):
    def test_fast_assembly_matches_reference_to_roundoff(self) -> None:
        bx = make_open_uniform_basis(n_elements=6, degree=3)
        by = make_open_uniform_basis(n_elements=6, degree=3)
        cut = HalfPlaneCut(normal=np.array([math.cos(0.37), math.sin(0.37)]), offset=0.58)
        ref = assemble_oblique_cut_matrices_reference(bx, by, cut, capture_volume_quadrature=True)
        fast = assemble_oblique_cut_matrices(bx, by, cut, capture_volume_quadrature=True)
        for name in ("H", "Qx", "Qy", "K", "Sx", "Sy"):
            err = float(np.max(np.abs(getattr(ref, name) - getattr(fast, name))))
            self.assertLess(err, 1.0e-13, name)
        self.assertEqual(ref.volume_quadrature.x.size, fast.volume_quadrature.x.size)
        self.assertLess(
            float(np.max(np.abs(ref.volume_quadrature.basis_values - fast.volume_quadrature.basis_values))),
            1.0e-13,
        )
        self.assertLess(
            float(np.max(np.abs(ref.volume_quadrature.basis_grad_x - fast.volume_quadrature.basis_grad_x))),
            1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
