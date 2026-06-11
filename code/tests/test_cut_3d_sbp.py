from __future__ import annotations

import unittest

from cut_3d_sbp import (
    assemble_cut_3d,
    degree_p_cut_frontier,
    green_residuals_3d,
    polynomial_reproduction_residual_3d,
    run_poisson_3d_convergence,
    run_small_cell_3d_sweep,
    scaled_advection_radius,
    scaled_mass_min_eig,
    solve_poisson_3d,
)


class Cut3DOperatorTest(unittest.TestCase):
    def test_green_identity_exact_in_3d(self) -> None:
        ops = assemble_cut_3d(n_elements=6, degree=3, cut_fractions=(0.99, 0.99, 0.99))
        rx, ry, rz = green_residuals_3d(ops)
        self.assertLess(max(rx, ry, rz), 1.0e-12)

    def test_polynomial_reproduction_exact_under_extreme_corner_cut(self) -> None:
        res = polynomial_reproduction_residual_3d(8, 3, (0.999, 0.999, 0.999))
        self.assertLess(res, 1.0e-10)

    def test_pulled_mass_floor_is_cut_independent(self) -> None:
        """The pulled-back mass floor stabilises as the sliver shrinks."""
        uncut = scaled_mass_min_eig(assemble_cut_3d(n_elements=8, degree=3, cut_fractions=(0.0, 0.0, 0.0)))
        deep = scaled_mass_min_eig(assemble_cut_3d(n_elements=8, degree=3, cut_fractions=(0.999, 0.0, 0.0)))
        deeper = scaled_mass_min_eig(assemble_cut_3d(n_elements=8, degree=3, cut_fractions=(0.99999, 0.0, 0.0)))
        # Bounded below by a cut-independent constant, and stable as frac -> 1.
        self.assertGreater(deep, 0.1 * uncut)
        self.assertAlmostEqual(deep, deeper, places=8)

    def test_naive_floor_collapses_but_relief_is_enormous(self) -> None:
        ops = assemble_cut_3d(n_elements=8, degree=3, cut_fractions=(0.999, 0.999, 0.999))
        pulled = scaled_mass_min_eig(ops)
        self.assertLess(ops.naive_scaled_mass_min_eig, 1.0e-30)
        self.assertGreater(pulled / max(ops.naive_scaled_mass_min_eig, 1.0e-300), 1.0e20)

    def test_advection_radius_bounded(self) -> None:
        rad = scaled_advection_radius(
            assemble_cut_3d(n_elements=8, degree=3, cut_fractions=(0.999, 0.999, 0.999))
        )
        self.assertLess(rad, 5.0)

    def test_high_degree_frontier_is_monotone(self) -> None:
        """Lower degrees tolerate more extreme cuts than higher degrees."""
        f2 = degree_p_cut_frontier(2, n_elements=8)["max_reproduced_fraction"]
        f4 = degree_p_cut_frontier(4, n_elements=8)["max_reproduced_fraction"]
        self.assertGreaterEqual(f2, f4)
        self.assertGreater(f4, 0.0)

    def test_sweep_runs_clean(self) -> None:
        out = run_small_cell_3d_sweep(n_elements=8, degrees=(2, 3), write=False)
        self.assertLess(out["summary"]["max_green_residual"], 1.0e-12)
        self.assertLess(out["summary"]["max_poly_reproduction_residual"], 1.0e-10)
        self.assertGreater(out["summary"]["max_mass_floor_relief"], 1.0e20)


class Poisson3DSolveTest(unittest.TestCase):
    def test_single_solve_is_spd_and_accurate(self) -> None:
        r = solve_poisson_3d(n_elements=6, degree=3, cut_fractions=(0.6, 0.6, 0.6))
        self.assertGreater(r["spd_min_eig"], 0.0)
        self.assertLess(r["relative_l2_error"], 1.0e-3)

    def test_degree_p_convergence_in_3d(self) -> None:
        out = run_poisson_3d_convergence(degree=3, elements=(4, 6, 8), write=False)
        # H1 slope tracks the degree, L2 slope tracks degree+1.
        self.assertGreater(out["summary"]["h1_slope"], 2.6)
        self.assertGreater(out["summary"]["l2_slope"], 3.5)


if __name__ == "__main__":
    unittest.main()
