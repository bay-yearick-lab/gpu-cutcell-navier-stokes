"""Gates for the 3D compressible Navier-Stokes demonstration on a sharp cut.

The four gates are checked in order on a genuinely cut (corner-sliver) box with
the aggregated cut SBP operators and the split
skew-symmetric convective form (as in the periodic reference
``cpp/include/solvers/compressible.h``):

  Gate 1  the cut convective derivative is discretely skew in the mass inner
          product to round-off, and the SBP / discrete Green identity
          ``Q_i + Q_i^T = S_i`` holds to round-off;
  Gate 2  mass, momentum and total energy are conserved to round-off over >=100
          steps in the boundary-accounted sense (the discrete conservation law
          closes), kinetic energy is preserved inviscidly, and the scheme is
          free-stream preserving;
  Gate 3  a smooth manufactured / exact solution is reproduced and the cut-local
          L2 error converges under refinement (second order, the order of the
          SOTA conservative cut-cell Euler competitor);
  Gate 4  the explicit-step spectral radius stays at the background scale under
          aggregation as the smallest cut fraction shrinks across decades, while
          the unaggregated active control collapses as 1/eta -- the relief ratio
          is the compressible analogue of the cut-channel time-step relief.

Run:  cd code && MPLBACKEND=Agg uv run python -m unittest test_comp3d -v
"""

from __future__ import annotations

import unittest

import numpy as np

from comp3d_solver import (
    CompressibleCut3D,
    acoustic_pulse_state,
    build_grid,
    gate1_operator_structure,
    gate2_conservation_ke,
    gate3_convergence,
    gate4_cut_independence,
    isentropic_vortex_state,
    set_state,
)


class Gate1OperatorStructure(unittest.TestCase):
    """Gate 1: discrete skewness in the mass inner product + SBP/Green on the cut."""

    def test_green_and_skew_to_roundoff(self):
        r = gate1_operator_structure(cut_fractions=(0.6, 0.6, 0.6))
        # genuinely cut corner sliver
        self.assertLess(r["volume_fraction"], 1.0e-1)
        # SBP/discrete Green identity Q + Q^T = S to round-off
        self.assertLess(r["green_residual_max"], 1.0e-12)
        # the boundary-free convective derivative is skew in the P mass norm
        self.assertLess(r["skew_residual_P_norm_max"], 1.0e-11)
        # the full derivative annihilates constants (consistency)
        self.assertLess(r["constant_annihilation_max"], 1.0e-11)

    def test_skew_holds_for_deep_corner_sliver(self):
        # at fraction 0.9 on three axes the corner cell volume ratio is 1e-3
        r = gate1_operator_structure(cut_fractions=(0.9, 0.9, 0.9))
        self.assertLess(r["green_residual_max"], 1.0e-11)
        self.assertLess(r["skew_residual_P_norm_max"], 1.0e-10)


class Gate2ConservationAndKineticEnergy(unittest.TestCase):
    """Gate 2: conservation to round-off over >=100 steps + inviscid KE preservation."""

    @classmethod
    def setUpClass(cls):
        cls.r = gate2_conservation_ke(
            cut_fractions=(0.6, 0.6, 0.6), n_steps=120, record=False
        )

    def test_runs_at_least_100_steps_stable(self):
        self.assertGreaterEqual(self.r["n_steps"], 100)
        self.assertGreater(self.r["min_density"], 0.0)

    def test_conservation_budget_closes_to_roundoff(self):
        bc = self.r["budget_closure"]
        self.assertLess(bc["mass"], 1.0e-11)
        self.assertLess(bc["mom_x"], 1.0e-11)
        self.assertLess(bc["mom_y"], 1.0e-11)
        self.assertLess(bc["mom_z"], 1.0e-11)
        self.assertLess(bc["energy"], 1.0e-11)

    def test_inviscid_kinetic_energy_preserved(self):
        # convective KE rate (pressure off) vanishes to round-off: no spurious KE
        self.assertLess(self.r["ke_rate_relative"], 1.0e-12)

    def test_free_stream_preserved(self):
        self.assertLess(self.r["free_stream_residual"], 1.0e-11)


class Gate3Convergence(unittest.TestCase):
    """Gate 3: cut-local L2 of a smooth/exact solution converges under refinement."""

    @classmethod
    def setUpClass(cls):
        cls.r = gate3_convergence(
            degree=3, elements=(4, 6, 8, 10), cut_fractions=(0.4, 0.4, 0.4)
        )

    def test_weak_derivative_converges_second_order(self):
        # weak Galerkin convective derivative on the cut: at least second order
        self.assertGreater(self.r["weak_derivative_slope"], 1.8)
        # error decreases monotonically under refinement
        e = self.r["weak_derivative_l2"]
        self.assertTrue(all(e[i + 1] < e[i] for i in range(len(e) - 1)))

    def test_exact_vortex_drift_converges(self):
        # stationary isentropic vortex is an exact steady solution; its short-time
        # drift converges as the cut box is refined
        self.assertGreater(self.r["vortex_drift_slope"], 1.0)
        e = self.r["vortex_drift_l2"]
        self.assertLess(e[-1], e[0])
        self.assertLess(e[-1], 1.0e-3)


class Gate4CutIndependence(unittest.TestCase):
    """Gate 4: cut-independent explicit step vs 1/eta collapse of the control."""

    @classmethod
    def setUpClass(cls):
        cls.r = gate4_cut_independence(
            degree=3, n_elements=4, fractions=(0.5, 0.9, 0.99, 0.999)
        )

    def test_aggregated_radius_stays_background_scale(self):
        # background-scale spectral radius across decades of sliver volume
        self.assertLess(self.r["aggregated_radius_max"], 10.0)
        self.assertGreater(self.r["aggregated_radius_min"], 1.0e-3)

    def test_control_collapses_and_relief_is_decisive(self):
        rows = self.r["rows"]
        # the unaggregated control collapses roughly as 1/eta: monotone growth
        naive = [row["naive_radius"] for row in rows]
        self.assertTrue(all(naive[i + 1] > naive[i] for i in range(len(naive) - 1)))
        # decisive relief at the smallest sliver (compressible analogue of 206x)
        self.assertGreater(self.r["max_relief_ratio"], 100.0)


class IntegrationSmoke(unittest.TestCase):
    """A short end-to-end advance to confirm the production solver runs and is sane."""

    def test_acoustic_pulse_short_run(self):
        g = build_grid(n_elements=4, degree=3, cut_fractions=(0.5, 0.5, 0.5))
        S = CompressibleCut3D(g, mu=0.0)
        rho, u, v, w, p = acoustic_pulse_state(g.coords, 0.0, eps=1.0e-3)
        set_state(S, rho, u, v, w, p)
        dt = 0.2 * g.h / S.max_wavespeed()
        for _ in range(20):
            S.step(dt)
        self.assertTrue(np.isfinite(S.rho).all())
        self.assertGreater(S.min_density(), 0.0)

    def test_viscous_dissipation_is_nonpositive_ke(self):
        # with mu>0 and pressure off, the viscous term should not create KE
        g = build_grid(n_elements=5, degree=3, cut_fractions=(0.4, 0.4, 0.4))
        S = CompressibleCut3D(g, mu=1.0e-2)
        x, y, z = g.coords[:, 0], g.coords[:, 1], g.coords[:, 2]
        k = 2.0 * np.pi
        set_state(
            S,
            1.0 + 0.0 * x,
            0.2 * np.sin(k * x) * np.cos(k * y),
            0.2 * np.cos(k * x) * np.sin(k * y),
            0.1 * np.sin(k * z),
            1.0 + 0.0 * x,
        )
        S.with_pressure = False
        S.slip_walls = True
        Rr, Rmx, Rmy, Rmz, Re = S.rhs(S.rho, S.mx, S.my, S.mz, S.E)
        U, V, W, _ = S.primitives()
        ke_rate = float(
            g.P @ (U * Rmx + V * Rmy + W * Rmz - 0.5 * (U * U + V * V + W * W) * Rr)
        )
        # viscous KE rate is dissipative (<= small positive tolerance)
        self.assertLess(ke_rate, 1.0e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
