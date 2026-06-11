from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigvalsh, svdvals


@dataclass(frozen=True)
class SplineBasis1D:
    degree: int
    n_elements: int
    knots: np.ndarray
    basis: list[BSpline]
    dbasis: list[BSpline]
    greville: np.ndarray
    supports: np.ndarray

    @property
    def n_basis(self) -> int:
        return len(self.basis)

    @property
    def h(self) -> float:
        return 1.0 / self.n_elements


@dataclass(frozen=True)
class CutMatrices1D:
    H: np.ndarray
    Q: np.ndarray
    K: np.ndarray
    H_background: np.ndarray
    boundary_form: np.ndarray
    left_trace: np.ndarray
    right_trace: np.ndarray


@dataclass(frozen=True)
class AggregationResult:
    E: np.ndarray
    interior: np.ndarray
    small_support: np.ndarray
    active: np.ndarray
    gamma: np.ndarray
    polynomial_degree: int
    polynomial_residual: float


def make_open_uniform_basis(n_elements: int = 32, degree: int = 3) -> SplineBasis1D:
    if n_elements < 2:
        raise ValueError("n_elements must be at least 2.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    h = 1.0 / n_elements
    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            np.arange(1, n_elements, dtype=float) * h,
            np.ones(degree + 1),
        ]
    )
    n_basis = len(knots) - degree - 1
    coeffs = np.eye(n_basis)
    basis = [BSpline(knots, coeffs[i], degree, extrapolate=False) for i in range(n_basis)]
    dbasis = [phi.derivative() for phi in basis]
    if degree == 0:
        greville = 0.5 * (knots[:-1] + knots[1:])
    else:
        greville = np.array(
            [np.sum(knots[i + 1 : i + degree + 1]) / degree for i in range(n_basis)]
        )
    supports = np.array([(knots[i], knots[i + degree + 1]) for i in range(n_basis)])
    return SplineBasis1D(degree, n_elements, knots, basis, dbasis, greville, supports)


def _eval_all(functions: list[BSpline], x: np.ndarray) -> np.ndarray:
    values = np.vstack([np.nan_to_num(f(x), nan=0.0) for f in functions])
    return values


def _quadrature_panels(knots: np.ndarray, left: float, right: float) -> np.ndarray:
    breaks = np.unique(knots)
    interior = breaks[(breaks > left) & (breaks < right)]
    return np.unique(np.concatenate(([left], interior, [right])))


def assemble_cut_matrices(
    basis: SplineBasis1D,
    left: float,
    right: float = 1.0,
    quadrature_order: int | None = None,
) -> CutMatrices1D:
    if not (0.0 <= left < right <= 1.0):
        raise ValueError("The cut interval must satisfy 0 <= left < right <= 1.")
    p = basis.degree
    q = quadrature_order or max(8, p + 4)
    xg, wg = np.polynomial.legendre.leggauss(q)
    n = basis.n_basis
    H = np.zeros((n, n))
    Q = np.zeros((n, n))
    K = np.zeros((n, n))
    H_background = np.zeros((n, n))

    for lo, hi in zip(*[_quadrature_panels(basis.knots, left, right)[i:] for i in (0, 1)]):
        if hi <= lo:
            continue
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        V = _eval_all(basis.basis, x)
        dV = _eval_all(basis.dbasis, x)
        H += (V * w) @ V.T
        Q += (V * w) @ dV.T
        K += (dV * w) @ dV.T

    for lo, hi in zip(*[_quadrature_panels(basis.knots, 0.0, 1.0)[i:] for i in (0, 1)]):
        if hi <= lo:
            continue
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        V = _eval_all(basis.basis, x)
        H_background += (V * w) @ V.T

    left_trace = _eval_all(basis.basis, np.array([left])).ravel()
    right_trace = _eval_all(basis.basis, np.array([right])).ravel()
    boundary_form = np.outer(right_trace, right_trace) - np.outer(left_trace, left_trace)
    return CutMatrices1D(H, Q, K, H_background, boundary_form, left_trace, right_trace)


def active_sets(
    matrices: CutMatrices1D,
    support_threshold: float = 0.05,
    active_tol: float = 1.0e-25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    background = np.diag(matrices.H_background)
    physical = np.diag(matrices.H)
    gamma = np.zeros_like(physical)
    positive = background > 0.0
    gamma[positive] = physical[positive] / background[positive]
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small_support = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    return active, interior, small_support, gamma


def polynomial_coefficients(basis: SplineBasis1D, max_degree: int) -> np.ndarray:
    A = _eval_all(basis.basis, basis.greville).T
    coeffs = []
    for r in range(max_degree + 1):
        coeffs.append(np.linalg.solve(A, basis.greville**r))
    return np.asarray(coeffs)


def polynomial_preserving_extension(
    basis: SplineBasis1D,
    matrices: CutMatrices1D,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
) -> AggregationResult:
    active, interior, small_support, gamma = active_sets(matrices, support_threshold=support_threshold)
    if len(interior) == 0:
        raise ValueError("No interior active basis functions were found.")
    degree = min(basis.degree if preserve_degree is None else preserve_degree, len(interior) - 1)
    poly = polynomial_coefficients(basis, degree)
    n_red = len(interior)
    n_full = basis.n_basis
    col = {int(g): j for j, g in enumerate(interior)}
    E = np.zeros((n_full, n_red))
    for g in interior:
        E[int(g), col[int(g)]] = 1.0

    for i in small_support:
        ordered_good = sorted([int(g) for g in interior], key=lambda g: abs(basis.greville[g] - basis.greville[int(i)]))
        chosen = ordered_good[: degree + 1]
        C = poly[:, chosen]
        # Expand the local stencil if the polynomial moments are rank deficient.
        for k in range(degree + 1, min(len(ordered_good), degree + 6) + 1):
            candidate = ordered_good[:k]
            C_candidate = poly[:, candidate]
            if np.linalg.matrix_rank(C_candidate, tol=1.0e-12) == degree + 1:
                chosen = candidate
                C = C_candidate
                break
        target = poly[:, int(i)]
        weights = np.linalg.lstsq(C, target, rcond=None)[0]
        for g, weight in zip(chosen, weights):
            E[int(i), col[int(g)]] = weight

    # The outside basis functions have zero rows. The extension only represents
    # active physical modes.
    residuals = []
    for r in range(degree + 1):
        c = poly[r]
        residuals.append(np.linalg.norm(E @ c[interior] - c * (gamma > 0.0), ord=np.inf))
    residual = float(max(residuals) if residuals else 0.0)
    return AggregationResult(E, interior.astype(int), small_support.astype(int), active.astype(int), gamma, degree, residual)


def pullback(matrices: CutMatrices1D, E: np.ndarray) -> CutMatrices1D:
    H = E.T @ matrices.H @ E
    Q = E.T @ matrices.Q @ E
    K = E.T @ matrices.K @ E
    H_background = E.T @ matrices.H_background @ E
    boundary_form = E.T @ matrices.boundary_form @ E
    left_trace = matrices.left_trace @ E
    right_trace = matrices.right_trace @ E
    return CutMatrices1D(H, Q, K, H_background, boundary_form, left_trace, right_trace)


def restrict_to_active(matrices: CutMatrices1D, active: np.ndarray) -> CutMatrices1D:
    idx = np.ix_(active, active)
    return CutMatrices1D(
        matrices.H[idx],
        matrices.Q[idx],
        matrices.K[idx],
        matrices.H_background[idx],
        matrices.boundary_form[idx],
        matrices.left_trace[active],
        matrices.right_trace[active],
    )


def sbp_residual(matrices: CutMatrices1D) -> float:
    residual = matrices.Q + matrices.Q.T - matrices.boundary_form
    return float(np.linalg.norm(residual, ord=np.inf))


def mass_condition_number(H: np.ndarray) -> float:
    s = svdvals(0.5 * (H + H.T))
    return float(s[0] / s[-1])


def heat_lambda_max(K: np.ndarray, H: np.ndarray) -> float:
    vals = eigvalsh(0.5 * (K + K.T), 0.5 * (H + H.T))
    return float(vals[-1])


def heat_rhs(c: np.ndarray, H: np.ndarray, K: np.ndarray) -> np.ndarray:
    return -np.linalg.solve(H, K @ c)


def rk4_step(c: np.ndarray, dt: float, H: np.ndarray, K: np.ndarray) -> np.ndarray:
    k1 = heat_rhs(c, H, K)
    k2 = heat_rhs(c + 0.5 * dt * k1, H, K)
    k3 = heat_rhs(c + 0.5 * dt * k2, H, K)
    k4 = heat_rhs(c + dt * k3, H, K)
    return c + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def heat_energy(c: np.ndarray, H: np.ndarray) -> float:
    return float(0.5 * c.T @ H @ c)
