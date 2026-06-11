import unittest

import numpy as np
import scipy.sparse as sp

from fast_solver_backends import csr_memory_bytes, solve_cpu_cg, solve_spd


class FastSolverBackendTests(unittest.TestCase):
    def test_cpu_cg_solves_small_spd_system(self):
        A = sp.diags([2.0, 3.0, 4.0, 5.0], format="csr")
        b = np.array([1.0, -2.0, 0.5, 4.0])
        x, report = solve_cpu_cg(A, b, tol=1.0e-12, maxiter=20)
        self.assertTrue(report.success, report)
        self.assertLess(report.relative_residual, 1.0e-10)
        np.testing.assert_allclose(A @ x, b, rtol=1.0e-10, atol=1.0e-12)

    def test_auto_backend_falls_back_to_cpu_when_needed(self):
        A = sp.diags([4.0, 5.0, 6.0], format="csr")
        b = np.ones(3)
        x, report = solve_spd(A, b, backend="auto", tol=1.0e-10, maxiter=20)
        self.assertIsNotNone(x)
        self.assertTrue(report.success, report)
        self.assertLess(report.relative_residual, 1.0e-8)

    def test_csr_memory_is_positive(self):
        A = sp.eye(5, format="csr")
        self.assertGreater(csr_memory_bytes(A), 0)


if __name__ == "__main__":
    unittest.main()
