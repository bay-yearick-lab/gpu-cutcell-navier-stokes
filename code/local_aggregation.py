from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable

import numpy as np
import scipy.linalg as sla

from cut_spline_sbp import SplineBasis1D
from oblique_cut_sbp import (
    ObliqueAggregation2D,
    ObliqueCutMatrices2D,
    VolumeQuadrature2D,
    tensor_polynomial_coefficients,
)
from circular_corner_sbp import CircularCornerMatrices2D


EXTENSION_RULES: tuple[str, ...] = (
    "distance_greedy",
    "scaled_greedy",
    "qr_pivoted",
    "coefficient_aware",
)


@dataclass(frozen=True)
class LocalAggregationDiagnostics2D:
    """Diagnostics for a graph-local polynomial extension.

    The extension matrix has one column for each interior active basis function.
    Healthy rows are the identity, small-support active rows are reconstructed
    from a local aggregate neighborhood of interior rows, and inactive rows are zero.
    """

    E: np.ndarray
    interior: np.ndarray
    small_support: np.ndarray
    active: np.ndarray
    gamma: np.ndarray
    polynomial_degree: int
    polynomial_residual: float
    powers: tuple[tuple[int, int], ...]
    monomial_residuals: np.ndarray
    stencil_sizes: np.ndarray
    aggregate_radii: np.ndarray
    stencil_min_singular_values: np.ndarray
    stencil_row_normalized_min_singular_values: np.ndarray
    stencil_condition_numbers: np.ndarray
    coefficient_max_abs_values: np.ndarray
    stencil_indices: tuple[tuple[int, ...], ...]
    stencil_weights: tuple[tuple[float, ...], ...]
    fallback_count: int
    max_aggregate_radius_allowed: int
    extension_rule: str = "distance_greedy"

    @property
    def max_stencil_size(self) -> int:
        if self.stencil_sizes.size == 0:
            return 1
        return int(np.max(self.stencil_sizes))

    @property
    def mean_stencil_size(self) -> float:
        if self.stencil_sizes.size == 0:
            return 1.0
        return float(np.mean(self.stencil_sizes))

    @property
    def max_aggregate_radius(self) -> int:
        if self.aggregate_radii.size == 0:
            return 0
        return int(np.max(self.aggregate_radii))

    @property
    def mean_aggregate_radius(self) -> float:
        if self.aggregate_radii.size == 0:
            return 0.0
        return float(np.mean(self.aggregate_radii))

    @property
    def min_stencil_singular_value(self) -> float:
        if self.stencil_min_singular_values.size == 0:
            return float("inf")
        return float(np.min(self.stencil_min_singular_values))

    @property
    def min_row_normalized_stencil_singular_value(self) -> float:
        if self.stencil_row_normalized_min_singular_values.size == 0:
            return float("inf")
        return float(np.min(self.stencil_row_normalized_min_singular_values))

    @property
    def max_stencil_condition_number(self) -> float:
        if self.stencil_condition_numbers.size == 0:
            return 1.0
        return float(np.max(self.stencil_condition_numbers))

    @property
    def max_coefficient_abs(self) -> float:
        if self.coefficient_max_abs_values.size == 0:
            return 1.0
        return float(np.max(self.coefficient_max_abs_values))

    @property
    def mean_coefficient_abs(self) -> float:
        if self.coefficient_max_abs_values.size == 0:
            return 1.0
        return float(np.mean(self.coefficient_max_abs_values))

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(np.abs(self.E) > 0.0))

    @property
    def density(self) -> float:
        return float(self.nnz / max(1, self.E.size))


@dataclass(frozen=True)
class AggregateFrameDiagnostics2D:
    rows: tuple[dict[str, float | int | str], ...]
    c_F: float
    C_F: float
    M_Omega: int
    M_G: int
    min_quadrature_node_cover: int
    min_reduced_index_cover: int
    c0: float
    c1: float
    n_aggregates: int
    n_empty_aggregates: int
    mass_reconstruction_error_inf: float
    lower_bound_ok: bool
    upper_bound_ok: bool

    def as_dict(self, prefix: str = "aggregate_frame") -> dict[str, float | int | bool]:
        return {
            f"{prefix}_c_F": float(self.c_F),
            f"{prefix}_C_F": float(self.C_F),
            f"{prefix}_M_Omega": int(self.M_Omega),
            f"{prefix}_M_G": int(self.M_G),
            f"{prefix}_min_quadrature_node_cover": int(self.min_quadrature_node_cover),
            f"{prefix}_min_reduced_index_cover": int(self.min_reduced_index_cover),
            f"{prefix}_c0": float(self.c0),
            f"{prefix}_c1": float(self.c1),
            f"{prefix}_n_aggregates": int(self.n_aggregates),
            f"{prefix}_n_empty_aggregates": int(self.n_empty_aggregates),
            f"{prefix}_mass_reconstruction_error_inf": float(self.mass_reconstruction_error_inf),
            f"{prefix}_lower_bound_ok": bool(self.lower_bound_ok),
            f"{prefix}_upper_bound_ok": bool(self.upper_bound_ok),
        }


@dataclass(frozen=True)
class StiffnessAggregateFrameDiagnostics2D:
    """Stiffness analog of the mass aggregate-frame diagnostic.

    The mass certificate measures ``h^{-d} P_ell E^T V^T W V E P_ell^T``.
    The stiffness certificate measures

        h^{d-2} times the inverse-scaled lower envelope plus
        h^{2-d} times the upper envelope

    of the aggregate-neighborhood Gram matrices built from gradient quadrature.  The
    constants ``c_K_F`` and ``C_K_F`` are the worst lower and upper
    aggregate-neighborhood eigenvalues across the cut domain; combined with the aggregate neighborhood
    overlap multiplicities, they give the cut-position-independent
    stiffness bound ``E^T A_Omega E lesssim h^{d-2} I``.
    """

    rows: tuple[dict[str, float | int | str], ...]
    c_K_F: float
    C_K_F: float
    M_Omega: int
    M_G: int
    c_K_lower: float
    c_K_upper: float
    n_aggregates: int
    n_empty_aggregates: int
    stiffness_reconstruction_error_inf: float

    def as_dict(self, prefix: str = "stiffness_aggregate_frame") -> dict[str, float | int]:
        return {
            f"{prefix}_c_K_F": float(self.c_K_F),
            f"{prefix}_C_K_F": float(self.C_K_F),
            f"{prefix}_M_Omega": int(self.M_Omega),
            f"{prefix}_M_G": int(self.M_G),
            f"{prefix}_c_K_lower": float(self.c_K_lower),
            f"{prefix}_c_K_upper": float(self.c_K_upper),
            f"{prefix}_n_aggregates": int(self.n_aggregates),
            f"{prefix}_n_empty_aggregates": int(self.n_empty_aggregates),
            f"{prefix}_stiffness_reconstruction_error_inf": float(
                self.stiffness_reconstruction_error_inf
            ),
        }


def tensor_grid_coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    ix = np.tile(np.arange(nx, dtype=int), ny)
    iy = np.repeat(np.arange(ny, dtype=int), nx)
    return ix, iy


def support_ratios_from_diagonal(
    H: np.ndarray,
    H_background: np.ndarray,
) -> np.ndarray:
    background = np.diag(H_background)
    physical = np.diag(H)
    gamma = np.zeros_like(physical)
    positive = background > 0.0
    gamma[positive] = physical[positive] / background[positive]
    return gamma


def active_sets_from_diagonal(
    H: np.ndarray,
    H_background: np.ndarray,
    *,
    support_threshold: float,
    active_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma = support_ratios_from_diagonal(H, H_background)
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small_support = np.array([int(i) for i in active if gamma[int(i)] < support_threshold], dtype=int)
    return active.astype(int), interior.astype(int), small_support.astype(int), gamma


def _max_degree_by_count(n_interior: int) -> int:
    degree = 0
    while (degree + 1) * (degree + 2) // 2 <= n_interior:
        degree += 1
    return degree - 1


def _rank_greedy_subset(
    poly: np.ndarray,
    candidates: Iterable[int],
    n_terms: int,
    *,
    tol: float,
) -> list[int] | None:
    """Distance-ordered rank growth for the local polynomial stencil.

    Each candidate is appended if it increases the rank of the partial coefficient
    block. This is the baseline distance-greedy rule. Operating on a locally
    rescaled polynomial coefficient block keeps the tolerance interpretation
    consistent across cut positions.
    """
    chosen: list[int] = []
    rank = 0
    for g in candidates:
        trial = chosen + [int(g)]
        new_rank = int(np.linalg.matrix_rank(poly[:, trial], tol=tol))
        if new_rank > rank:
            chosen = trial
            rank = new_rank
            if rank == n_terms:
                return chosen
    return None


def _pivoted_qr_subset(
    poly_scaled: np.ndarray,
    candidates: list[int],
    n_terms: int,
    *,
    tol: float,
) -> list[int] | None:
    """Select n_terms columns by column-pivoted QR on the row-normalized block.

    Column-pivoted QR maximizes the diagonal of R at each step, which is the
    norm of the residual projected away from the previously chosen columns.
    The resulting stencil is as well-conditioned as the candidate set permits.
    """
    if len(candidates) < n_terms:
        return None
    C = poly_scaled[:, candidates]
    row_norms = np.linalg.norm(C, axis=1)
    row_scaled = C / np.maximum(row_norms[:, None], 1.0e-300)
    _, R, piv = sla.qr(row_scaled, mode="economic", pivoting=True)
    if abs(R[n_terms - 1, n_terms - 1]) <= tol:
        return None
    return [candidates[int(p)] for p in piv[:n_terms]]


def _local_scaled_polynomial_coefficients(
    poly_global: np.ndarray,
    powers: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    xb: float,
    yb: float,
    hx: float,
    hy: float,
) -> np.ndarray:
    """Return T @ poly_global where T maps to monomials centered at (xb, yb)
    and scaled by (hx, hy).

    The local monomial m_{p,q}(x, y) = ((x - xb)/hx)^p ((y - yb)/hy)^q has
    B-spline Greville coefficients sum_{i<=p, j<=q} t_{(p,q),(i,j)} * c_{(i,j)},
    where t_{(p,q),(i,j)} is the binomial expansion coefficient divided by
    hx^p hy^q. Selection and weight-solve become a well-scaled n_terms x
    n_terms problem; the resulting weights still reproduce the original
    global monomials because T is invertible.
    """
    n_terms = poly_global.shape[0]
    T = np.zeros((n_terms, n_terms), dtype=float)
    index = {pq: k for k, pq in enumerate(powers)}
    for k_row, (p, q) in enumerate(powers):
        scale = 1.0 / (hx**p * hy**q)
        for i in range(p + 1):
            for j in range(q + 1):
                k_col = index.get((i, j))
                if k_col is None:
                    continue
                T[k_row, k_col] = (
                    scale
                    * comb(p, i)
                    * comb(q, j)
                    * (-xb) ** (p - i)
                    * (-yb) ** (q - j)
                )
    return T @ poly_global


def _solve_weights(C: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares weights with a safe fallback when C is exactly square.

    For an n_terms x n_terms full-rank block the two routes agree to roundoff;
    the explicit branch keeps the conditioning visible.
    """
    if C.shape[0] == C.shape[1]:
        try:
            return np.linalg.solve(C, target)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.lstsq(C, target, rcond=None)[0]


def _stencil_diagnostics_on_global(
    poly_global: np.ndarray,
    selected: list[int],
) -> tuple[float, float, float]:
    C = poly_global[:, selected]
    svals = np.linalg.svd(C, compute_uv=False)
    row_norms = np.linalg.norm(C, axis=1)
    row_scaled = C / np.maximum(row_norms[:, None], 1.0e-300)
    scaled_svals = np.linalg.svd(row_scaled, compute_uv=False)
    sigma_min = float(svals[-1])
    sigma_row_min = float(scaled_svals[-1])
    cond = float(svals[0] / max(svals[-1], 1.0e-300))
    return sigma_min, sigma_row_min, cond


def _select_for_small_support_mode(
    *,
    b: int,
    interior: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    poly_global: np.ndarray,
    powers: tuple[tuple[int, int], ...],
    greville_x: np.ndarray,
    greville_y: np.ndarray,
    hx: float,
    hy: float,
    n_terms: int,
    max_radius_allowed: int,
    rank_tol: float,
    extension_rule: str,
    coefficient_threshold: float,
) -> tuple[list[int] | None, np.ndarray | None, int]:
    """Pick a interior extension stencil for small-support basis function `b` under `extension_rule`.

    Returns (selected_indices, weights, selected_radius). When no in-aggregate
    stencil works, returns (None, None, max_radius_allowed) so the caller can
    invoke the global fallback.
    """
    xb = float(greville_x[b])
    yb = float(greville_y[b])

    if extension_rule == "distance_greedy":
        poly_local = poly_global
    else:
        poly_local = _local_scaled_polynomial_coefficients(
            poly_global, powers, xb, yb, hx, hy
        )

    def _ordered_candidates(radius: int) -> list[int]:
        in_aggregate_neighborhood = np.maximum(np.abs(ix[interior] - ix[b]), np.abs(iy[interior] - iy[b])) <= radius
        candidates = interior[in_aggregate_neighborhood]
        return sorted(
            [int(g) for g in candidates],
            key=lambda g: (
                max(abs(int(ix[g]) - int(ix[b])), abs(int(iy[g]) - int(iy[b]))),
                (int(ix[g]) - int(ix[b])) ** 2 + (int(iy[g]) - int(iy[b])) ** 2,
                g,
            ),
        )

    def _try_rule(rule: str, candidates: list[int]) -> list[int] | None:
        if len(candidates) < n_terms:
            return None
        if rule in ("distance_greedy", "scaled_greedy"):
            return _rank_greedy_subset(poly_local, candidates, n_terms, tol=rank_tol)
        if rule == "qr_pivoted":
            return _pivoted_qr_subset(poly_local, candidates, n_terms, tol=rank_tol)
        raise ValueError(f"unknown extension sub-rule: {rule}")

    if extension_rule in ("distance_greedy", "scaled_greedy", "qr_pivoted"):
        for radius in range(0, max_radius_allowed + 1):
            candidates = _ordered_candidates(radius)
            selected = _try_rule(extension_rule, candidates)
            if selected is None:
                continue
            C_local = poly_local[:, selected]
            target_local = poly_local[:, b]
            weights = _solve_weights(C_local, target_local)
            return selected, weights, radius
        return None, None, max_radius_allowed

    if extension_rule == "coefficient_aware":
        # Try distance_greedy and qr_pivoted at each radius; keep the smaller
        # max|weight|. If still above threshold, grow the aggregate neighborhood. The final
        # stencil is the one with the smallest observed max|weight| across
        # all radii tried, so radius growth is only paid for when it actually
        # helps the coefficient bound.
        best_selected: list[int] | None = None
        best_weights: np.ndarray | None = None
        best_radius = max_radius_allowed
        best_coef = float("inf")
        for radius in range(0, max_radius_allowed + 1):
            candidates = _ordered_candidates(radius)
            for sub_rule in ("distance_greedy", "qr_pivoted"):
                if sub_rule == "distance_greedy":
                    selected = _rank_greedy_subset(poly_local, candidates, n_terms, tol=rank_tol)
                else:
                    selected = _pivoted_qr_subset(poly_local, candidates, n_terms, tol=rank_tol)
                if selected is None:
                    continue
                C_local = poly_local[:, selected]
                target_local = poly_local[:, b]
                weights = _solve_weights(C_local, target_local)
                coef = float(np.max(np.abs(weights))) if weights.size else 0.0
                if coef < best_coef:
                    best_coef = coef
                    best_selected = selected
                    best_weights = weights
                    best_radius = radius
            if best_selected is not None and best_coef <= coefficient_threshold:
                return best_selected, best_weights, best_radius
        return best_selected, best_weights, best_radius

    raise ValueError(f"unknown extension_rule: {extension_rule!r}; expected one of {EXTENSION_RULES}")


def local_polynomial_aggregation_2d(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    H: np.ndarray,
    H_background: np.ndarray,
    *,
    nx: int,
    ny: int,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    active_tol: float = 1.0e-12,
    max_aggregate_radius: int | None = None,
    rank_tol: float = 1.0e-12,
    extension_rule: str = "distance_greedy",
    coefficient_threshold: float = 20.0,
) -> LocalAggregationDiagnostics2D:
    """Build a local, sparse, polynomial-preserving active-space extension.

    For each small-support active tensor-product B-spline, candidate extension modes
    are interior modes in a growing Chebyshev aggregate neighborhood on the tensor index grid.  A
    geometry-agnostic extension-selection rule (see ``EXTENSION_RULES``) picks a local
    stencil that spans every total-degree monomial coefficient row, and the
    extension weights are the least-squares solution against the small-support
    monomial coefficient column. The diagnostics record stencil singular values
    on the global (unscaled) coefficient block so the rule comparison stays
    numerically consistent with prior work.
    """

    if nx * ny != H.shape[0] or H.shape != H_background.shape:
        raise ValueError("Matrix dimensions are inconsistent with nx, ny.")
    if extension_rule not in EXTENSION_RULES:
        raise ValueError(f"unknown extension_rule {extension_rule!r}; expected one of {EXTENSION_RULES}")
    active, interior, small_support, gamma = active_sets_from_diagonal(
        H,
        H_background,
        support_threshold=support_threshold,
        active_tol=active_tol,
    )
    if len(interior) == 0:
        raise ValueError("No interior active tensor-product basis functions were found.")

    requested_degree = basis_x.degree if preserve_degree is None else preserve_degree
    degree = min(requested_degree, _max_degree_by_count(len(interior)))
    if degree < 0:
        raise ValueError("Not enough interior modes to preserve constants.")
    poly, raw_powers = tensor_polynomial_coefficients(basis_x, basis_y, degree)
    powers = tuple((int(a), int(b)) for a, b in raw_powers)
    n_terms = int(poly.shape[0])
    n_full = int(H.shape[0])
    n_red = int(len(interior))
    col = {int(g): j for j, g in enumerate(interior)}

    E = np.zeros((n_full, n_red))
    for g in interior:
        E[int(g), col[int(g)]] = 1.0

    ix, iy = tensor_grid_coordinates(nx, ny)
    greville_x = np.tile(basis_x.greville, ny)
    greville_y = np.repeat(basis_y.greville, nx)
    hx = float(basis_x.h)
    hy = float(basis_y.h)
    max_radius_allowed = int(max_aggregate_radius if max_aggregate_radius is not None else max(nx, ny))
    global_radius = int(max(nx, ny))
    stencil_sizes: list[int] = []
    aggregate_radii: list[int] = []
    stencil_min_singular_values: list[float] = []
    stencil_row_normalized_min_singular_values: list[float] = []
    stencil_condition_numbers: list[float] = []
    coefficient_max_abs_values: list[float] = []
    selected_stencils: list[tuple[int, ...]] = []
    selected_weights: list[tuple[float, ...]] = []
    fallback_count = 0

    for b in small_support:
        b = int(b)
        selected, weights, selected_radius = _select_for_small_support_mode(
            b=b,
            interior=interior,
            ix=ix,
            iy=iy,
            poly_global=poly,
            powers=powers,
            greville_x=greville_x,
            greville_y=greville_y,
            hx=hx,
            hy=hy,
            n_terms=n_terms,
            max_radius_allowed=max_radius_allowed,
            rank_tol=rank_tol,
            extension_rule=extension_rule,
            coefficient_threshold=coefficient_threshold,
        )

        if selected is None or weights is None:
            # Global-graph fallback: distance-greedy over every interior column.
            # Recorded explicitly so any loss of locality stays visible.
            fallback_count += 1
            order = sorted(
                [int(g) for g in interior],
                key=lambda g: (
                    (int(ix[g]) - int(ix[b])) ** 2 + (int(iy[g]) - int(iy[b])) ** 2,
                    g,
                ),
            )
            selected = _rank_greedy_subset(poly, order, n_terms, tol=rank_tol)
            if selected is None:
                raise np.linalg.LinAlgError(
                    "Could not find a full-rank interior stencil for local polynomial aggregation."
                )
            weights = _solve_weights(poly[:, selected], poly[:, b])
            selected_radius = global_radius

        for g, weight in zip(selected, weights):
            E[b, col[int(g)]] = float(weight)

        sigma_min, sigma_row_min, cond = _stencil_diagnostics_on_global(poly, selected)
        stencil_sizes.append(len(selected))
        aggregate_radii.append(int(selected_radius))
        stencil_min_singular_values.append(sigma_min)
        stencil_row_normalized_min_singular_values.append(sigma_row_min)
        stencil_condition_numbers.append(cond)
        coefficient_max_abs_values.append(float(np.max(np.abs(weights))) if weights.size else 0.0)
        selected_stencils.append(tuple(int(g) for g in selected))
        selected_weights.append(tuple(float(w) for w in weights))

    active_mask = gamma > active_tol
    monomial_residuals = np.array(
        [np.linalg.norm(E @ row[interior] - row * active_mask, ord=np.inf) for row in poly],
        dtype=float,
    )
    polynomial_residual = float(np.max(monomial_residuals) if monomial_residuals.size else 0.0)
    return LocalAggregationDiagnostics2D(
        E=E,
        interior=interior,
        small_support=small_support,
        active=active,
        gamma=gamma,
        polynomial_degree=int(degree),
        polynomial_residual=polynomial_residual,
        powers=powers,
        monomial_residuals=monomial_residuals,
        stencil_sizes=np.asarray(stencil_sizes, dtype=int),
        aggregate_radii=np.asarray(aggregate_radii, dtype=int),
        stencil_min_singular_values=np.asarray(stencil_min_singular_values, dtype=float),
        stencil_row_normalized_min_singular_values=np.asarray(stencil_row_normalized_min_singular_values, dtype=float),
        stencil_condition_numbers=np.asarray(stencil_condition_numbers, dtype=float),
        coefficient_max_abs_values=np.asarray(coefficient_max_abs_values, dtype=float),
        stencil_indices=tuple(selected_stencils),
        stencil_weights=tuple(selected_weights),
        fallback_count=int(fallback_count),
        max_aggregate_radius_allowed=max_radius_allowed,
        extension_rule=extension_rule,
    )


def aggregate_frame_diagnostics_2d(
    local: LocalAggregationDiagnostics2D,
    volume: VolumeQuadrature2D,
    H: np.ndarray,
    *,
    h: float,
    support_tol: float = 1.0e-14,
    measured_lambda_min_over_hd: float | None = None,
    measured_lambda_max_over_hd: float | None = None,
) -> AggregateFrameDiagnostics2D:
    """Measure the local aggregate-frame constants in the metric theorem."""
    if volume.basis_values.shape[0] != local.E.shape[0]:
        raise ValueError("volume basis_values and extension matrix have inconsistent row counts")
    if volume.weights.ndim != 1 or volume.basis_values.shape[1] != volume.weights.size:
        raise ValueError("volume weights and basis_values have inconsistent quadrature dimensions")
    if np.any(volume.weights <= 0.0):
        raise ValueError("aggregate-frame diagnostics require positive volume weights")

    V = np.asarray(volume.basis_values, dtype=float)
    weights = np.asarray(volume.weights, dtype=float)
    Vtilde = V.T @ local.E
    H_from_volume = volume.mass_matrix()
    mass_error = float(np.linalg.norm(H_from_volume - H, ord=np.inf))
    n_full, n_points = V.shape
    n_red = local.E.shape[1]
    active_set = set(int(i) for i in local.active)
    small_to_pos = {int(b): j for j, b in enumerate(local.small_support)}
    reduced_col = {int(g): j for j, g in enumerate(local.interior)}
    reduced_cover = np.zeros(n_red, dtype=int)
    node_cover = np.zeros(n_points, dtype=int)
    rows: list[dict[str, float | int | str]] = []
    lambda_mins: list[float] = []
    lambda_maxs: list[float] = []
    empty_aggregates = 0

    for aggregate_index, active_row in enumerate(int(i) for i in local.active):
        if active_row not in active_set:
            continue
        if active_row in small_to_pos:
            pos = small_to_pos[active_row]
            aggregate_rows = [active_row, *local.stencil_indices[pos]]
            aggregate_kind = "small_support"
            aggregate_radius = int(local.aggregate_radii[pos])
            stencil_size = int(local.stencil_sizes[pos])
        else:
            aggregate_rows = [active_row]
            aggregate_kind = "interior"
            aggregate_radius = 0
            stencil_size = 1
        aggregate_rows = [row for row in aggregate_rows if 0 <= row < n_full]
        q_mask = np.any(np.abs(V[aggregate_rows, :]) > support_tol, axis=0)
        q_idx = np.where(q_mask)[0]
        if q_idx.size == 0:
            empty_aggregates += 1
            continue
        col_idx = np.array([reduced_col[row] for row in aggregate_rows if row in reduced_col], dtype=int)
        if col_idx.size == 0:
            empty_aggregates += 1
            continue
        local_values = Vtilde[np.ix_(q_idx, col_idx)]
        gram = (local_values.T * weights[q_idx]) @ local_values
        gram = 0.5 * (gram + gram.T) / (h * h)
        eigs = np.linalg.eigvalsh(gram)
        lam_min = float(eigs[0])
        lam_max = float(eigs[-1])
        lambda_mins.append(lam_min)
        lambda_maxs.append(lam_max)
        node_cover[q_idx] += 1
        reduced_cover[col_idx] += 1
        rows.append(
            {
                "aggregate_index": int(aggregate_index),
                "active_row": int(active_row),
                "aggregate_kind": aggregate_kind,
                "quadrature_node_count": int(q_idx.size),
                "reduced_index_count": int(col_idx.size),
                "lambda_min": lam_min,
                "lambda_max": lam_max,
                "aggregate_radius": int(aggregate_radius),
                "stencil_size": int(stencil_size),
            }
        )

    if not lambda_mins:
        raise ValueError("no nonempty aggregate-frame aggregate neighborhoods were found")
    positive_nodes = weights > 0.0
    min_node_cover = int(np.min(node_cover[positive_nodes])) if np.any(positive_nodes) else 0
    max_node_cover = int(np.max(node_cover[positive_nodes])) if np.any(positive_nodes) else 0
    min_reduced_cover = int(np.min(reduced_cover)) if reduced_cover.size else 0
    max_reduced_cover = int(np.max(reduced_cover)) if reduced_cover.size else 0
    c_F = float(np.min(lambda_mins))
    C_F = float(np.max(lambda_maxs))
    c0 = c_F / max(max_node_cover, 1)
    c1 = C_F * max(max_reduced_cover, 1)
    lower_ok = True
    upper_ok = True
    if measured_lambda_min_over_hd is not None:
        lower_ok = bool(c0 <= float(measured_lambda_min_over_hd) * (1.0 + 1.0e-10))
    if measured_lambda_max_over_hd is not None:
        upper_ok = bool(c1 + 1.0e-12 >= float(measured_lambda_max_over_hd))
    return AggregateFrameDiagnostics2D(
        rows=tuple(rows),
        c_F=c_F,
        C_F=C_F,
        M_Omega=max_node_cover,
        M_G=max_reduced_cover,
        min_quadrature_node_cover=min_node_cover,
        min_reduced_index_cover=min_reduced_cover,
        c0=float(c0),
        c1=float(c1),
        n_aggregates=len(rows),
        n_empty_aggregates=int(empty_aggregates),
        mass_reconstruction_error_inf=mass_error,
        lower_bound_ok=lower_ok,
        upper_bound_ok=upper_ok,
    )


def aggregate_frame_stiffness_diagnostics_2d(
    local: LocalAggregationDiagnostics2D,
    volume: VolumeQuadrature2D,
    K: np.ndarray,
    *,
    h: float,
    support_tol: float = 1.0e-14,
) -> StiffnessAggregateFrameDiagnostics2D:
    """Measure the aggregate-frame constants of the cut stiffness matrix.

    Forms ``A_K_ell = h^{2-d} P_ell E^T (grad phi)^T W_ell (grad phi) E P_ell^T``
    on each graph aggregate and reports the worst lower and upper eigenvalues.
    The diagnostic uses the gradient values already captured by the volume
    quadrature representation; the assembled stiffness matrix is also
    reconstructed from the gradient values as an internal sanity check.
    """
    if volume.basis_grad_x is None or volume.basis_grad_y is None:
        raise ValueError(
            "stiffness aggregate-frame diagnostics require gradient values "
            "captured at the volume quadrature nodes"
        )
    if volume.basis_grad_x.shape[0] != local.E.shape[0]:
        raise ValueError(
            "volume gradient and extension matrix have inconsistent row counts"
        )
    if np.any(volume.weights <= 0.0):
        raise ValueError("stiffness aggregate-frame diagnostics require positive weights")

    gx = np.asarray(volume.basis_grad_x, dtype=float)
    gy = np.asarray(volume.basis_grad_y, dtype=float)
    V = np.asarray(volume.basis_values, dtype=float)
    weights = np.asarray(volume.weights, dtype=float)
    K_from_volume = volume.stiffness_matrix()
    stiffness_error = float(np.linalg.norm(K_from_volume - K, ord=np.inf))
    Gx_red = gx.T @ local.E
    Gy_red = gy.T @ local.E
    V_red = V.T @ local.E
    n_full, n_points = V.shape
    n_red = local.E.shape[1]
    active_set = set(int(i) for i in local.active)
    small_to_pos = {int(b): j for j, b in enumerate(local.small_support)}
    reduced_col = {int(g): j for j, g in enumerate(local.interior)}
    reduced_cover = np.zeros(n_red, dtype=int)
    node_cover = np.zeros(n_points, dtype=int)
    rows: list[dict[str, float | int | str]] = []
    lambda_mins: list[float] = []
    lambda_maxs: list[float] = []
    empty_aggregates = 0

    for aggregate_index, active_row in enumerate(int(i) for i in local.active):
        if active_row not in active_set:
            continue
        if active_row in small_to_pos:
            pos = small_to_pos[active_row]
            aggregate_rows = [active_row, *local.stencil_indices[pos]]
            aggregate_kind = "small_support"
            aggregate_radius = int(local.aggregate_radii[pos])
            stencil_size = int(local.stencil_sizes[pos])
        else:
            aggregate_rows = [active_row]
            aggregate_kind = "interior"
            aggregate_radius = 0
            stencil_size = 1
        aggregate_rows = [row for row in aggregate_rows if 0 <= row < n_full]
        q_mask = np.any(np.abs(V[aggregate_rows, :]) > support_tol, axis=0)
        q_idx = np.where(q_mask)[0]
        if q_idx.size == 0:
            empty_aggregates += 1
            continue
        col_idx = np.array(
            [reduced_col[row] for row in aggregate_rows if row in reduced_col], dtype=int
        )
        if col_idx.size == 0:
            empty_aggregates += 1
            continue
        gx_block = Gx_red[np.ix_(q_idx, col_idx)]
        gy_block = Gy_red[np.ix_(q_idx, col_idx)]
        wq = weights[q_idx]
        gram = (gx_block.T * wq) @ gx_block + (gy_block.T * wq) @ gy_block
        gram = 0.5 * (gram + gram.T)
        # Scale by h^{2-d}; we are in 2D so d = 2 and the scale factor is 1.
        scale = h ** (2 - 2)
        gram = gram * scale
        eigs = np.linalg.eigvalsh(gram)
        lam_min = float(eigs[0])
        lam_max = float(eigs[-1])
        lambda_mins.append(lam_min)
        lambda_maxs.append(lam_max)
        node_cover[q_idx] += 1
        reduced_cover[col_idx] += 1
        rows.append(
            {
                "aggregate_index": int(aggregate_index),
                "active_row": int(active_row),
                "aggregate_kind": aggregate_kind,
                "quadrature_node_count": int(q_idx.size),
                "reduced_index_count": int(col_idx.size),
                "lambda_min": lam_min,
                "lambda_max": lam_max,
                "aggregate_radius": int(aggregate_radius),
                "stencil_size": int(stencil_size),
            }
        )

    if not lambda_mins:
        raise ValueError("no nonempty stiffness aggregate-frame aggregate neighborhoods were found")
    positive_nodes = weights > 0.0
    max_node_cover = int(np.max(node_cover[positive_nodes])) if np.any(positive_nodes) else 0
    max_reduced_cover = int(np.max(reduced_cover)) if reduced_cover.size else 0
    c_K_F = float(np.min(lambda_mins))
    C_K_F = float(np.max(lambda_maxs))
    return StiffnessAggregateFrameDiagnostics2D(
        rows=tuple(rows),
        c_K_F=c_K_F,
        C_K_F=C_K_F,
        M_Omega=max_node_cover,
        M_G=max_reduced_cover,
        c_K_lower=c_K_F / max(max_node_cover, 1),
        c_K_upper=C_K_F * max(max_reduced_cover, 1),
        n_aggregates=len(rows),
        n_empty_aggregates=int(empty_aggregates),
        stiffness_reconstruction_error_inf=stiffness_error,
    )


def local_oblique_extension(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    matrices: ObliqueCutMatrices2D,
    *,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    active_tol: float = 1.0e-22,
    max_aggregate_radius: int | None = None,
    extension_rule: str = "distance_greedy",
    coefficient_threshold: float = 20.0,
) -> LocalAggregationDiagnostics2D:
    return local_polynomial_aggregation_2d(
        basis_x,
        basis_y,
        matrices.H,
        matrices.H_background,
        nx=matrices.nx,
        ny=matrices.ny,
        support_threshold=support_threshold,
        preserve_degree=preserve_degree,
        active_tol=active_tol,
        max_aggregate_radius=max_aggregate_radius,
        extension_rule=extension_rule,
        coefficient_threshold=coefficient_threshold,
    )


def local_circular_extension(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    matrices: CircularCornerMatrices2D,
    *,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    active_tol: float = 1.0e-12,
    max_aggregate_radius: int | None = None,
    extension_rule: str = "distance_greedy",
    coefficient_threshold: float = 20.0,
) -> LocalAggregationDiagnostics2D:
    return local_polynomial_aggregation_2d(
        basis_x,
        basis_y,
        matrices.H,
        matrices.H_background,
        nx=matrices.nx,
        ny=matrices.ny,
        support_threshold=support_threshold,
        preserve_degree=preserve_degree,
        active_tol=active_tol,
        max_aggregate_radius=max_aggregate_radius,
        extension_rule=extension_rule,
        coefficient_threshold=coefficient_threshold,
    )


def as_oblique_aggregation(local: LocalAggregationDiagnostics2D) -> ObliqueAggregation2D:
    """Return the common aggregation container used by older prototypes."""

    return ObliqueAggregation2D(
        E=local.E,
        interior=local.interior,
        small_support=local.small_support,
        active=local.active,
        gamma=local.gamma,
        polynomial_degree=local.polynomial_degree,
        polynomial_residual=local.polynomial_residual,
    )


def extension_matrix_stats(E: np.ndarray, *, small_support: np.ndarray | None = None) -> dict[str, float | int]:
    row_nnz = np.count_nonzero(np.abs(E) > 0.0, axis=1)
    col_nnz = np.count_nonzero(np.abs(E) > 0.0, axis=0)
    stats: dict[str, float | int] = {
        "extension_nnz": int(np.count_nonzero(np.abs(E) > 0.0)),
        "extension_density": float(np.count_nonzero(np.abs(E) > 0.0) / max(1, E.size)),
        "extension_row_nnz_max": int(np.max(row_nnz) if row_nnz.size else 0),
        "extension_row_nnz_mean": float(np.mean(row_nnz) if row_nnz.size else 0.0),
        "extension_col_nnz_max": int(np.max(col_nnz) if col_nnz.size else 0),
        "extension_col_nnz_mean": float(np.mean(col_nnz) if col_nnz.size else 0.0),
    }
    if small_support is not None and len(small_support):
        small_support_nnz = row_nnz[np.asarray(small_support, dtype=int)]
        stats["small_support_row_nnz_max"] = int(np.max(small_support_nnz))
        stats["small_support_row_nnz_mean"] = float(np.mean(small_support_nnz))
    else:
        stats["small_support_row_nnz_max"] = 0
        stats["small_support_row_nnz_mean"] = 0.0
    return stats


def local_diagnostics_as_dict(local: LocalAggregationDiagnostics2D, prefix: str = "local") -> dict[str, float | int]:
    out: dict[str, float | int] = {
        f"{prefix}_polynomial_degree": int(local.polynomial_degree),
        f"{prefix}_polynomial_residual_inf": float(local.polynomial_residual),
        f"{prefix}_max_stencil_size": int(local.max_stencil_size),
        f"{prefix}_mean_stencil_size": float(local.mean_stencil_size),
        f"{prefix}_max_aggregate_radius": int(local.max_aggregate_radius),
        f"{prefix}_mean_aggregate_radius": float(local.mean_aggregate_radius),
        f"{prefix}_min_stencil_singular_value": float(local.min_stencil_singular_value),
        f"{prefix}_min_row_normalized_stencil_singular_value": float(local.min_row_normalized_stencil_singular_value),
        f"{prefix}_max_stencil_condition_number": float(local.max_stencil_condition_number),
        f"{prefix}_max_coefficient_abs": float(local.max_coefficient_abs),
        f"{prefix}_mean_coefficient_abs": float(local.mean_coefficient_abs),
        f"{prefix}_fallback_count": int(local.fallback_count),
        f"{prefix}_extension_nnz": int(local.nnz),
        f"{prefix}_extension_density": float(local.density),
        f"{prefix}_extension_rule": str(local.extension_rule),
        f"{prefix}_n_recorded_stencils": int(len(local.stencil_indices)),
    }
    for (px, py), residual in zip(local.powers, local.monomial_residuals):
        out[f"{prefix}_monomial_x{px}_y{py}_residual_inf"] = float(residual)
    return out
