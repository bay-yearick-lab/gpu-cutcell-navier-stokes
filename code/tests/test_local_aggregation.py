from __future__ import annotations

from math import cos, sin
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circular_corner_sbp import (  # noqa: E402
    CircularCornerCut,
    assemble_circular_corner_matrices,
    directional_sbp_residuals_circular,
    pullback_circular_corner_2d,
)
from cut_spline_sbp import make_open_uniform_basis  # noqa: E402
from local_aggregation import local_circular_extension, local_oblique_extension  # noqa: E402
from oblique_cut_sbp import (  # noqa: E402
    HalfPlaneCut,
    assemble_oblique_cut_matrices,
    directional_sbp_residuals_oblique,
    pullback_oblique_2d,
    tensor_polynomial_coefficients,
)


class LocalAggregationTests(unittest.TestCase):
    def test_local_oblique_extension_preserves_polynomials_and_sbp(self) -> None:
        basis = make_open_uniform_basis(n_elements=5, degree=3)
        h = basis.h
        angle = 0.71
        normal = np.array([cos(angle), sin(angle)])
        normal = normal / np.linalg.norm(normal)
        cut = HalfPlaneCut(normal=normal, offset=float((normal[0] + normal[1]) * h * 0.95))
        matrices = assemble_oblique_cut_matrices(basis, basis, cut, quadrature_order=10)
        local = local_oblique_extension(basis, basis, matrices, support_threshold=0.05)
        self.assertGreater(len(local.small_support), 0)
        self.assertEqual(local.polynomial_degree, 3)
        self.assertLess(local.polynomial_residual, 1.0e-10)
        self.assertLessEqual(local.max_stencil_size, 10)
        self.assertLessEqual(local.fallback_count, 0)
        red = pullback_oblique_2d(matrices, local.E)
        self.assertLess(max(directional_sbp_residuals_oblique(red)), 1.0e-11)

    def test_local_circular_extension_preserves_each_monomial(self) -> None:
        basis = make_open_uniform_basis(n_elements=5, degree=3)
        radius = np.sqrt(2.0) * basis.h * 0.95
        matrices = assemble_circular_corner_matrices(
            basis,
            basis,
            CircularCornerCut(radius=radius),
            quadrature_order=14,
        )
        local = local_circular_extension(basis, basis, matrices, support_threshold=0.05)
        poly, powers = tensor_polynomial_coefficients(basis, basis, local.polynomial_degree)
        active_mask = local.gamma > 1.0e-12
        for row, power in zip(poly, powers):
            err = np.linalg.norm(local.E @ row[local.interior] - row * active_mask, ord=np.inf)
            self.assertLess(err, 1.0e-10, msg=f"failed for monomial {power}")
        red = pullback_circular_corner_2d(matrices, local.E)
        self.assertLess(max(directional_sbp_residuals_circular(red)), 1.0e-10)



if __name__ == "__main__":
    unittest.main()
