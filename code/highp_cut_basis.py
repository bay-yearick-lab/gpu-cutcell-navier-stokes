"""Well-conditioned high-degree cut-extension basis for the curved-cut showcase.

This module provides a drop-in, *better-conditioned* replacement for the
polynomial-pinning extension used by the curved-cut Poisson solver, plus the
machinery to run a pure weighted-L2 projection (which isolates the
basis-conditioning floor from the PDE solve) in double or extended precision.

Diagnosis it is built to defeat
--------------------------------
On the curved-cut manufactured Poisson sweep the realized L2 order saturates
near 4-5 for spline degree ``p >= 4`` and the deepest error floors near
``3e-10``.  The cap is *not* approximation power.  Profiling the stock
``polynomial_preserving_circular_extension`` shows the small-support extension is
built by least squares against the GLOBAL monomial coefficient matrix
``tensor_polynomial_coefficients`` whose moment matrix ``C`` has condition number
``~1e11-1e13`` at ``p = 10``.  That ill-conditioning limits the polynomial
reproduction of the extension to ``~5e-9`` (its ``polynomial_residual``), and
that residual is the L2 floor.

Three independent levers, all implemented here
----------------------------------------------
1. WELL-CONDITIONED BASIS: build the extension by fitting *locally shifted and
   scaled* monomials ``((x-x0)/s)^r ((y-y0)/s)^q`` centered at the small-support
   dof's Greville point, instead of global monomials on ``[0,1]``.  This drops
   the stencil moment-matrix condition number by several orders of magnitude and
   the extension residual from ``~5e-9`` to ``~1e-11``.
2. HIGHER-ORDER QUADRATURE: the assembly quadrature order is exposed so the
   curved-panel consistency budget can be pushed well below the target error.
3. EXTENDED PRECISION: ``project_l2`` accepts ``dtype=np.longdouble`` and solves
   the physical Gram by an in-precision Cholesky so the algebraic order ``p+1``
   is exposed before double-precision roundoff caps it.

Lever 4 (benign annulus geometry) lives in ``highp_annulus.py`` which reuses the
same well-conditioned extension idea on an exact polar quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cut_spline_sbp import SplineBasis1D, _eval_all, make_open_uniform_basis
from circular_corner_sbp import (
    CircularCornerCut,
    CircularCornerMatrices2D,
    assemble_circular_corner_matrices,
    circular_corner_active_sets,
)
from oblique_cut_sbp import ObliqueAggregation2D, _tensor_greville


# ---------------------------------------------------------------------------
# Local (shifted + scaled) polynomial coefficient construction
# ---------------------------------------------------------------------------


def _interp_inverse(basis: SplineBasis1D) -> np.ndarray:
    """Inverse of the 1D collocation matrix at the Greville abscissae.

    ``A[i, j] = phi_j(g_i)``; ``A^{-1}`` maps nodal values at the Greville points
    to spline coefficients.  The Greville collocation matrix is total positive
    and very well conditioned (it is the identity for the constant), so this is a
    stable way to realize an arbitrary polynomial as spline coefficients.
    """
    A = _eval_all(basis.basis, basis.greville).T
    return np.linalg.inv(A)


def _local_axis_values(u: np.ndarray, degree: int, local_basis: str) -> np.ndarray:
    """Sample the ``degree``-th local 1D polynomial basis function at ``u``.

    ``"monomial"`` returns ``u**degree`` (the committed default); ``"chebyshev"``
    and ``"legendre"`` return the corresponding orthogonal polynomial of that
    degree.  The orthogonal bases sample the local stencil far more evenly, which
    drops the moment-matrix condition number of the small-support fit by many
    orders at high spline degree (see ``well_conditioned_circular_extension``).
    """
    if local_basis == "monomial":
        return u ** degree
    coef = np.zeros(degree + 1)
    coef[degree] = 1.0
    if local_basis == "chebyshev":
        return np.polynomial.chebyshev.chebval(u, coef)
    if local_basis == "legendre":
        return np.polynomial.legendre.legval(u, coef)
    raise ValueError(f"unknown local_basis {local_basis!r}")


def local_tensor_polynomial_coefficients(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    max_total_degree: int,
    x0: float,
    y0: float,
    scale: float,
    local_basis: str = "monomial",
) -> np.ndarray:
    """Spline-coefficient rows for the LOCAL polynomial basis.

    Row ``(r, q)`` (``r + q <= max_total_degree``) is the tensor-product spline
    coefficient vector reproducing the local degree-``(r, q)`` tensor polynomial
    in the shifted/scaled coordinate ``((x-x0)/scale, (y-y0)/scale)`` exactly (it
    is a polynomial of degree ``<=`` the spline degree, so the spline space
    contains it).  Built by Greville collocation per axis, which keeps the
    construction well conditioned.

    ``local_basis`` selects the 1D local basis: ``"monomial"`` (the committed
    default, ``u^r u^q``) or ``"chebyshev"``/``"legendre"`` orthogonal
    polynomials.  The choice only changes which polynomial *functionals* the rows
    represent; the spanned row space (degree-``max_total_degree`` polynomials) is
    identical, so the resulting extension reproduces the same ``Q_p``.  At high
    spline degree the orthogonal bases give a dramatically better-conditioned
    small-support fit.
    """
    gx = basis_x.greville
    gy = basis_y.greville
    Axinv = _interp_inverse(basis_x)
    Ayinv = _interp_inverse(basis_y)
    ux = (gx - x0) / scale
    uy = (gy - y0) / scale
    rows: list[np.ndarray] = []
    for total in range(max_total_degree + 1):
        for r in range(total + 1):
            q = total - r
            cx = Axinv @ _local_axis_values(ux, r, local_basis)
            cy = Ayinv @ _local_axis_values(uy, q, local_basis)
            rows.append(np.kron(cy, cx))
    return np.asarray(rows)


def well_conditioned_circular_extension(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    matrices: CircularCornerMatrices2D,
    support_threshold: float = 0.04,
    preserve_degree: int | None = None,
    active_tol: float = 1.0e-12,
    local_basis: str = "monomial",
    extra_columns: int = 0,
) -> ObliqueAggregation2D:
    """Polynomial-pinning extension fit in a LOCAL, well-conditioned basis.

    Signature matches ``polynomial_preserving_circular_extension`` so it is a
    drop-in ``extension_builder`` for ``solve_poisson_circular``.  The only
    change is that the small-support least-squares fit is posed against locally
    shifted/scaled monomials, which removes the global-monomial conditioning that
    caps the stock extension's polynomial reproduction at high degree.

    Two opt-in levers reduce the small-support fit conditioning further at high
    spline degree, both defaulting OFF so the committed degree-6/8 path is
    byte-for-byte unchanged:

    * ``local_basis="chebyshev"`` (or ``"legendre"``) replaces the local
      monomials with orthogonal polynomials; this alone drops the fit
      condition number by ~13 orders at degree 12.
    * ``extra_columns=k`` appends up to ``k`` further nearest interior
      coordinates beyond the minimal unisolvent stencil and solves the
      over-determined least-squares fit; combined with the orthogonal basis it
      recovers the degree-10 polynomial reproduction from ~3e-12 to ~4e-13.
    """
    sets = circular_corner_active_sets(
        matrices, support_threshold=support_threshold, active_tol=active_tol
    )
    if len(sets.interior) == 0:
        raise ValueError("No interior active tensor-product basis functions were found.")
    requested = basis_x.degree if preserve_degree is None else preserve_degree
    max_by_count = 0
    while (max_by_count + 1) * (max_by_count + 2) // 2 <= len(sets.interior):
        max_by_count += 1
    max_by_count -= 1
    degree = min(requested, max_by_count)
    n_terms = (degree + 1) * (degree + 2) // 2

    greville = _tensor_greville(basis_x, basis_y)
    n_full = matrices.H.shape[0]
    n_red = len(sets.interior)
    col = {int(g): j for j, g in enumerate(sets.interior)}
    E = np.zeros((n_full, n_red))
    for g in sets.interior:
        E[int(g), col[int(g)]] = 1.0

    h = 1.0 / basis_x.n_elements
    scale = float((degree + 1) * h)  # support-width local scale
    max_res = 0.0
    for b in sets.small_support:
        x0, y0 = greville[int(b)]
        polyL = local_tensor_polynomial_coefficients(
            basis_x, basis_y, degree, float(x0), float(y0), scale, local_basis=local_basis
        )
        distances = np.linalg.norm(greville[sets.interior] - greville[int(b)], axis=1)
        ordered = [int(g) for _, g in sorted(zip(distances, sets.interior), key=lambda it: it[0])]
        chosen = ordered[:n_terms]
        C = polyL[:, chosen]
        for k in range(n_terms, len(ordered) + 1):
            cand = ordered[:k]
            Cc = polyL[:, cand]
            if np.linalg.matrix_rank(Cc, tol=1.0e-10) == n_terms:
                chosen = cand
                C = Cc
                break
        # Optional over-determination: append the next nearest interior dofs
        # beyond the minimal unisolvent stencil and fit least squares.  Better
        # conditioning, same reproduced polynomial space.  Off by default.
        if extra_columns > 0:
            base = len(chosen)
            enlarged = ordered[: min(base + extra_columns, len(ordered))]
            chosen = enlarged
            C = polyL[:, chosen]
        target = polyL[:, int(b)]
        weights = np.linalg.lstsq(C, target, rcond=None)[0]
        max_res = max(max_res, float(np.linalg.norm(C @ weights - target, np.inf)))
        for g, weight in zip(chosen, weights):
            E[int(b), col[int(g)]] = weight

    # Residual against the GLOBAL monomials (same metric the stock builder
    # reports) so the two are directly comparable.
    from oblique_cut_sbp import tensor_polynomial_coefficients

    poly_global, _ = tensor_polynomial_coefficients(basis_x, basis_y, degree)
    active_mask = sets.gamma > active_tol
    residuals = [
        np.linalg.norm(E @ row[sets.interior] - row * active_mask, ord=np.inf)
        for row in poly_global
    ]
    residual = float(max(residuals) if residuals else 0.0)
    return ObliqueAggregation2D(
        E=E,
        interior=sets.interior.astype(int),
        small_support=sets.small_support.astype(int),
        active=sets.active.astype(int),
        gamma=sets.gamma,
        polynomial_degree=degree,
        polynomial_residual=max(residual, max_res),
    )


# ---------------------------------------------------------------------------
# Pure weighted-L2 projection (isolates the basis-conditioning floor)
# ---------------------------------------------------------------------------


def _chol_solve(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve SPD ``G x = b`` honoring ``G``'s dtype (so longdouble stays 80-bit).

    ``numpy.linalg.solve`` and ``scipy`` downcast to float64; this hand-rolled
    Cholesky keeps the extended mantissa for the extended-precision lever.
    """
    n = b.shape[0]
    L = np.zeros_like(G)
    for j in range(n):
        s = G[j, j] - L[j, :j] @ L[j, :j]
        L[j, j] = np.sqrt(s)
        if j + 1 < n:
            L[j + 1 :, j] = (G[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    # forward / backward substitution
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1 :, i] @ x[i + 1 :]) / L[i, i]
    return x


@dataclass(frozen=True)
class ProjectionResult:
    p: int
    n_elements: int
    h: float
    relative_l2_error: float
    gram_cond: float
    extension_residual: float
    n_physical: int
    n_quadrature: int
    quadrature_order: int
    precision: str


def project_l2(
    p: int,
    n_elements: int,
    radius: float = 0.42,
    extension: str = "well",
    quadrature_extra: int = 14,
    precision: str = "double",
    support_threshold: float = 0.04,
    target: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> ProjectionResult:
    """Weighted-L2 projection of the manufactured solution onto the cut space.

    This is the diagnostic that strips the PDE solve away: it measures the best
    the physical cut space can represent the target under the same quadrature and
    extension used by the solver, so the resulting floor is purely the
    basis-conditioning / quadrature budget.

    ``extension`` is ``"well"`` (this module) or ``"stock"`` (the committed
    polynomial-pinning builder).  ``precision`` is ``"double"`` or ``"long"``
    (numpy longdouble assembly + in-precision Cholesky solve).
    """
    from cut_pde_solve import manufactured_poisson

    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=p)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=p)
    q_order = max(32, 2 * p + quadrature_extra)
    cut = CircularCornerCut(radius=radius)
    matrices = assemble_circular_corner_matrices(
        basis_x, basis_y, cut, quadrature_order=q_order, capture_volume_quadrature=True
    )
    if extension == "well":
        agg = well_conditioned_circular_extension(
            basis_x, basis_y, matrices, support_threshold=support_threshold, preserve_degree=p
        )
    else:
        from circular_corner_sbp import polynomial_preserving_circular_extension

        agg = polynomial_preserving_circular_extension(
            basis_x, basis_y, matrices, support_threshold=support_threshold, preserve_degree=p
        )
    E = agg.E
    q = matrices.volume_quadrature
    if target is None:
        u_exact, _, _ = manufactured_poisson(radius)
    else:
        u_exact = target

    dt = np.longdouble if precision == "long" else np.float64
    Phi = (E.T @ q.basis_values).astype(dt)
    w = q.weights.astype(dt)
    G = (Phi * w) @ Phi.T
    uv = u_exact(q.x, q.y).astype(dt)
    rhs = (Phi * w) @ uv
    if precision == "long":
        coeff = _chol_solve(G, rhs)
    else:
        coeff = np.linalg.solve(G, rhs)
    uh = Phi.T @ coeff
    err = uh - uv
    l2 = np.sqrt(np.sum(w * err * err))
    l2ref = np.sqrt(np.sum(w * uv * uv))
    cond = float(np.linalg.cond(G.astype(np.float64)))
    return ProjectionResult(
        p=int(p),
        n_elements=int(n_elements),
        h=1.0 / n_elements,
        relative_l2_error=float(l2 / l2ref),
        gram_cond=cond,
        extension_residual=float(agg.polynomial_residual),
        n_physical=int(E.shape[1]),
        n_quadrature=int(w.size),
        quadrature_order=int(q_order),
        precision=precision,
    )
