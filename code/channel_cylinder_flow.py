"""Vortex shedding past a sharply cut cylinder: the 2D-2 cylinder benchmark.

This module carries the mixed aggregated velocity-pressure construction of
``cut_saddle_flow`` to a realistic unsteady flow: the laminar vortex street
behind a circular cylinder in a channel at Re = 100 (the standard 2D-2 benchmark).  The geometry is the rectangle [0, 2.2] x [0, 0.41]
minus the disk of radius 0.05 centered at (0.2, 0.2); the disk is represented
sharply on a non-conforming tensor-product B-spline background grid, and the
small cut cells around the cylinder are handled by the same
polynomial-preserving aggregation as every other operator in the manuscript
(coordinate selection first, sharp assembly second, congruence restriction
third; no stabilization parameter anywhere).

Discretization
--------------
* Spline Taylor-Hood pair: velocity components in the degree-``p`` tensor
  space, pressure in the degree-``p-1`` space on the same knots, each with its
  own sparse polynomial-preserving extension (Marsden-exact monomial
  coefficients, coefficient-aware local stencils).
* Sharp assembly by the rectangle-minus-hole decomposition: tensor Kronecker
  products of exact 1D Grams minus a smooth polar rule on the removed disk
  (panels split at every knot crossing), so volume operators carry no
  vertical-panel tangency floor and the matrices are sparse by construction.
* Boundary conditions: parabolic inflow (weak-penalty) on the left, no-slip
  (weak-penalty) on the channel walls and on the sharp cylinder arc, do-nothing
  outflow on the right (no pressure pin needed).
* Time stepping: Crank-Nicolson diffusion + Adams-Bashforth-2 skew-symmetric
  convection against one sparse LU of the constant saddle matrix; the
  divergence constraint is enforced at every time level.
* Forces by the variationally consistent reaction functional (the discrete
  momentum residual against a partition-of-unity lift supported on the
  cylinder ring), plus the front/rear pressure difference of the benchmark.

Reference intervals (case 2D-2, Re = 100):
    c_D,max in [3.22, 3.24],  c_L,max in [0.99, 1.01],
    St in [0.295, 0.305],     Delta p(t_cLmax) in [2.46, 2.50].
"""

from __future__ import annotations

import os

for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "4")

from dataclasses import dataclass
from math import comb, pi, sqrt
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"


# ---------------------------------------------------------------------------
# Scaled 1D clamped basis and sparse tensor evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaledBasis1D:
    """Clamped open-uniform B-spline basis on [0, length]."""

    degree: int
    n_elements: int
    length: float
    knots: np.ndarray
    greville: np.ndarray

    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree - 1

    @property
    def h(self) -> float:
        return self.length / self.n_elements


def make_scaled_basis(n_elements: int, degree: int, length: float) -> ScaledBasis1D:
    h = length / n_elements
    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            np.arange(1, n_elements, dtype=float) * h,
            np.full(degree + 1, length),
        ]
    )
    n_basis = len(knots) - degree - 1
    greville = np.array(
        [np.sum(knots[i + 1 : i + degree + 1]) / degree for i in range(n_basis)]
    )
    return ScaledBasis1D(degree, n_elements, float(length), knots, greville)


def _derivative_map(basis: ScaledBasis1D) -> sp.csr_matrix:
    """Sparse map T with B'(x) coefficients: d/dx sum c_i B_i = sum (Tc)_j D_j.

    D_j are the degree-(p-1) basis functions on the interior knot vector
    t[1:-1]; T is the standard bidiagonal difference scaled by p / knot gaps.
    """
    p = basis.degree
    t = basis.knots
    n = basis.n_basis
    rows, cols, vals = [], [], []
    for j in range(n - 1):
        gap = t[j + p + 1] - t[j + 1]
        if gap <= 0:
            continue
        c = p / gap
        rows.append(j)
        cols.append(j + 1)
        vals.append(c)
        rows.append(j)
        cols.append(j)
        vals.append(-c)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n - 1, n))


def basis_values_banded(basis: ScaledBasis1D, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values of the p+1 nonzero basis functions at each point.

    Returns (vals, cols): vals[i, k] is the value of basis function
    cols[i, k] at x[i]; exactly p+1 entries per point.
    """
    x = np.asarray(x, dtype=float)
    dm = BSpline.design_matrix(np.clip(x, 0.0, basis.length), basis.knots, basis.degree)
    p = basis.degree
    vals = dm.data.reshape(x.size, p + 1)
    cols = dm.indices.reshape(x.size, p + 1)
    return vals, cols


def basis_derivative_banded(
    basis: ScaledBasis1D, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the p+1 nonzero basis functions at each point.

    Built from the degree-(p-1) design matrix and the bidiagonal derivative
    map, aligned to the same column window as ``basis_values_banded``.
    """
    x = np.asarray(x, dtype=float)
    p = basis.degree
    t = basis.knots
    dm = BSpline.design_matrix(np.clip(x, 0.0, basis.length), t[1:-1], p - 1)
    dvals = dm.data.reshape(x.size, p)
    dcols = dm.indices.reshape(x.size, p)
    # d/dx B_j = c_{j-1,j} D_{j-1} + c_{j,j} D_j with c from the derivative map:
    # D_k contributes to B_k (coefficient -p/gap_k) and B_{k+1} (+p/gap_k).
    n = basis.n_basis
    out_vals = np.zeros((x.size, p + 1))
    out_cols = np.zeros((x.size, p + 1), dtype=dm.indices.dtype)
    # Column window of the value design matrix: leftmost nonzero basis index.
    vals_cols = basis_values_banded(basis, x)[1]
    left = vals_cols[:, 0]
    out_cols[:] = left[:, None] + np.arange(p + 1)[None, :]
    gaps = t[1 + p : n + p] - t[1:n]  # gap_k = t[k+p+1] - t[k+1], k = 0..n-2
    coeff = np.where(gaps > 0, p / np.where(gaps > 0, gaps, 1.0), 0.0)
    for k_local in range(p):
        k_global = dcols[:, k_local]  # derivative basis index
        c = coeff[k_global]
        v = dvals[:, k_local]
        # contribution to B_{k_global} with -c
        idx = k_global - left
        valid = (idx >= 0) & (idx <= p)
        rows = np.nonzero(valid)[0]
        out_vals[rows, idx[valid]] -= c[valid] * v[valid]
        # contribution to B_{k_global + 1} with +c
        idx2 = k_global + 1 - left
        valid2 = (idx2 >= 0) & (idx2 <= p)
        rows2 = np.nonzero(valid2)[0]
        out_vals[rows2, idx2[valid2]] += c[valid2] * v[valid2]
    return out_vals, out_cols


def sparse_tensor_eval(
    basis_x: ScaledBasis1D,
    basis_y: ScaledBasis1D,
    x: np.ndarray,
    y: np.ndarray,
    derivatives: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix | None, sp.csr_matrix | None]:
    """Sparse (n_points x n_basis) tensor basis values and gradients.

    Row layout follows the repository convention: index = iy * nx + ix.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    npts = x.size
    nxb = basis_x.n_basis
    px = basis_x.degree
    py = basis_y.degree
    vx, cx = basis_values_banded(basis_x, x)
    vy, cy = basis_values_banded(basis_y, y)
    width = (px + 1) * (py + 1)
    cols = (cy[:, :, None] * nxb + cx[:, None, :]).reshape(npts, width)
    data = (vy[:, :, None] * vx[:, None, :]).reshape(npts, width)
    indptr = np.arange(0, npts * width + 1, width)
    n_total = nxb * basis_y.n_basis
    phi = sp.csr_matrix((data.ravel(), cols.ravel(), indptr), shape=(npts, n_total))
    if not derivatives:
        return phi, None, None
    dvx, dcx = basis_derivative_banded(basis_x, x)
    dvy, dcy = basis_derivative_banded(basis_y, y)
    data_dx = (vy[:, :, None] * dvx[:, None, :]).reshape(npts, width)
    cols_dx = (cy[:, :, None] * nxb + dcx[:, None, :]).reshape(npts, width)
    dphix = sp.csr_matrix((data_dx.ravel(), cols_dx.ravel(), indptr.copy()), shape=(npts, n_total))
    data_dy = (dvy[:, :, None] * vx[:, None, :]).reshape(npts, width)
    cols_dy = (dcy[:, :, None] * nxb + cx[:, None, :]).reshape(npts, width)
    dphiy = sp.csr_matrix((data_dy.ravel(), cols_dy.ravel(), indptr.copy()), shape=(npts, n_total))
    return phi, dphix, dphiy


# ---------------------------------------------------------------------------
# Exact 1D Grams (sparse) and Marsden monomial coefficients
# ---------------------------------------------------------------------------


def _gauss_panels(basis: ScaledBasis1D, order: int) -> tuple[np.ndarray, np.ndarray]:
    g, w = np.polynomial.legendre.leggauss(order)
    breaks = np.unique(basis.knots)
    xs, ws = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi <= lo:
            continue
        xs.append(0.5 * (hi - lo) * g + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * w)
    return np.concatenate(xs), np.concatenate(ws)


def sparse_1d_grams(
    basis_a: ScaledBasis1D, basis_b: ScaledBasis1D, order: int
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Exact sparse 1D mixed Grams M[a,b] = int Ba Bb, D[a,b] = int Ba' Bb."""
    x, w = _gauss_panels(basis_a, order)
    va, ca = basis_values_banded(basis_a, x)
    dva, dca = basis_derivative_banded(basis_a, x)
    vb, cb = basis_values_banded(basis_b, x)
    pa = basis_a.degree
    pb = basis_b.degree
    na, nb = basis_a.n_basis, basis_b.n_basis
    rows = np.repeat(ca, pb + 1, axis=1)
    cols = np.tile(cb, (1, pa + 1))
    vals = (va[:, :, None] * vb[:, None, :]).reshape(x.size, -1) * w[:, None]
    M = sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(na, nb))
    rows_d = np.repeat(dca, pb + 1, axis=1)
    vals_d = (dva[:, :, None] * vb[:, None, :]).reshape(x.size, -1) * w[:, None]
    D = sp.csr_matrix((vals_d.ravel(), (rows_d.ravel(), cols.ravel())), shape=(na, nb))
    return M, D


def marsden_monomial_coefficients(basis: ScaledBasis1D, max_degree: int) -> np.ndarray:
    """Exact B-spline coefficients of x^r, r = 0..max_degree (Marsden identity).

    lambda_i(x^r) = e_r(t_{i+1}, ..., t_{i+p}) / C(p, r), where e_r is the
    elementary symmetric polynomial of the p interior knots of basis i.
    """
    p = basis.degree
    t = basis.knots
    n = basis.n_basis
    out = np.zeros((max_degree + 1, n))
    for i in range(n):
        window = t[i + 1 : i + p + 1]
        # elementary symmetric polynomials e_0..e_max via the running product
        e = np.zeros(max_degree + 1)
        e[0] = 1.0
        for value in window:
            upper = min(max_degree, p)
            for r in range(upper, 0, -1):
                e[r] += value * e[r - 1]
        for r in range(max_degree + 1):
            out[r, i] = e[r] / comb(p, r) if r <= p else 0.0
    return out


# ---------------------------------------------------------------------------
# Geometry: rectangle minus interior disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelGeometry:
    length: float = 2.2
    height: float = 0.41
    cx: float = 0.2
    cy: float = 0.2
    radius: float = 0.05


def _unique_sorted(values, lo: float, hi: float, tol: float = 1.0e-13) -> np.ndarray:
    clipped = [lo, hi]
    for value in values:
        if lo + tol < value < hi - tol:
            clipped.append(float(value))
    arr = np.asarray(sorted(clipped), dtype=float)
    keep = [arr[0]]
    for value in arr[1:]:
        if abs(value - keep[-1]) > tol:
            keep.append(float(value))
    return np.asarray(keep, dtype=float)


def _disk_angle_breaks(geo: ChannelGeometry, knots_x: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    """Angles in [0, 2pi] where the circle crosses background knot lines."""
    values: list[float] = []
    for xk in np.unique(knots_x):
        c = (xk - geo.cx) / geo.radius
        if -1.0 < c < 1.0:
            a = float(np.arccos(c))
            values.extend([a, 2.0 * pi - a])
    for yk in np.unique(knots_y):
        s = (yk - geo.cy) / geo.radius
        if -1.0 < s < 1.0:
            a = float(np.arcsin(s))
            values.extend([a % (2.0 * pi), (pi - a) % (2.0 * pi)])
    return _unique_sorted(values, 0.0, 2.0 * pi)


def disk_polar_quadrature(
    geo: ChannelGeometry, knots_x: np.ndarray, knots_y: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth polar rule over the removed disk, split at knot crossings.

    Angular panels split where the circle crosses knot lines; radial panels
    split where each ray crosses knot lines, so spline factors stay on single
    polynomial pieces and the rule converges spectrally (no tangency floor).
    """
    gt, wt = np.polynomial.legendre.leggauss(order)
    gr, wr = np.polynomial.legendre.leggauss(order)
    theta_breaks = _disk_angle_breaks(geo, knots_x, knots_y)
    # Sub-divide every angular panel with Chebyshev clustering toward both
    # endpoints: where two crossing angles collide (a knot line near tangency)
    # the angular integrand carries an endpoint square-root kink, and graded
    # panels restore fast convergence that uniform Gauss panels lose.
    refined: list[float] = []
    n_sub = 10
    cheb = 0.5 * (1.0 - np.cos(pi * np.arange(n_sub + 1) / n_sub))
    for ta, tb in zip(theta_breaks[:-1], theta_breaks[1:]):
        if tb <= ta + 1.0e-15:
            continue
        for frac in cheb[:-1]:
            refined.append(float(ta + (tb - ta) * frac))
    refined.append(float(theta_breaks[-1]))
    theta_breaks = np.asarray(refined)
    xs, ys, ws = [], [], []
    ux = np.unique(knots_x)
    uy = np.unique(knots_y)
    for ta, tb in zip(theta_breaks[:-1], theta_breaks[1:]):
        if tb <= ta + 1.0e-15:
            continue
        theta = 0.5 * (tb - ta) * gt + 0.5 * (tb + ta)
        wtheta = 0.5 * (tb - ta) * wt
        for th, wth in zip(theta, wtheta):
            c = float(np.cos(th))
            s = float(np.sin(th))
            rvals: list[float] = []
            if abs(c) > 1.0e-14:
                for xk in ux:
                    r = (xk - geo.cx) / c
                    if 0.0 < r < geo.radius:
                        rvals.append(float(r))
            if abs(s) > 1.0e-14:
                for yk in uy:
                    r = (yk - geo.cy) / s
                    if 0.0 < r < geo.radius:
                        rvals.append(float(r))
            rb = _unique_sorted(rvals, 0.0, geo.radius)
            for ra, rb_hi in zip(rb[:-1], rb[1:]):
                if rb_hi <= ra:
                    continue
                r = 0.5 * (rb_hi - ra) * gr + 0.5 * (rb_hi + ra)
                w = 0.5 * (rb_hi - ra) * wr
                xs.append(geo.cx + r * c)
                ys.append(geo.cy + r * s)
                ws.append(wth * w * r)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def arc_quadrature(
    geo: ChannelGeometry, knots_x: np.ndarray, knots_y: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cylinder-arc nodes, weights, and outward (into-disk) domain normals."""
    g, w = np.polynomial.legendre.leggauss(order)
    breaks = _disk_angle_breaks(geo, knots_x, knots_y)
    xs, ys, ws, nxs, nys = [], [], [], [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a + 1.0e-15:
            continue
        theta = 0.5 * (b - a) * g + 0.5 * (b + a)
        wt = 0.5 * (b - a) * w * geo.radius
        c = np.cos(theta)
        s = np.sin(theta)
        xs.append(geo.cx + geo.radius * c)
        ys.append(geo.cy + geo.radius * s)
        ws.append(wt)
        nxs.append(-c)
        nys.append(-s)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ws),
        np.concatenate(nxs),
        np.concatenate(nys),
    )


def _side_quadrature_1d(basis: ScaledBasis1D, order: int) -> tuple[np.ndarray, np.ndarray]:
    return _gauss_panels(basis, order)


def positive_channel_volume_quadrature(
    basis_x: ScaledBasis1D,
    basis_y: ScaledBasis1D,
    geo: ChannelGeometry,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positive cell-local rule over (rectangle minus disk); returns x, y, w, cell_id.

    Full cells use the tensor Gauss rule; cells meeting the circle integrate
    the fluid part by vertical panels split at circle extremes and crossings
    (used for convection, loads, norms; the operators use the
    rectangle-minus-hole decomposition instead).
    """
    g, w = np.polynomial.legendre.leggauss(order)
    hx = basis_x.h
    hy = basis_y.h
    r2 = geo.radius**2
    xs, ys, ws, cells = [], [], [], []
    ux = np.unique(basis_x.knots)
    uy = np.unique(basis_y.knots)

    def cell_outside_disk(x0, x1, y0, y1) -> bool:
        dx = max(x0 - geo.cx, 0.0, geo.cx - x1)
        dy = max(y0 - geo.cy, 0.0, geo.cy - y1)
        return dx * dx + dy * dy >= r2 - 1.0e-15

    def cell_inside_disk(x0, x1, y0, y1) -> bool:
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        return all((cx_ - geo.cx) ** 2 + (cy_ - geo.cy) ** 2 <= r2 + 1.0e-15 for cx_, cy_ in corners)

    for iy in range(basis_y.n_elements):
        y0, y1 = iy * hy, (iy + 1) * hy
        for ix in range(basis_x.n_elements):
            x0, x1 = ix * hx, (ix + 1) * hx
            cell_id = ix + basis_x.n_elements * iy
            if cell_outside_disk(x0, x1, y0, y1):
                xq = 0.5 * (x1 - x0) * g + 0.5 * (x1 + x0)
                yq = 0.5 * (y1 - y0) * g + 0.5 * (y1 + y0)
                wx = 0.5 * (x1 - x0) * w
                wy = 0.5 * (y1 - y0) * w
                XX, YY = np.meshgrid(xq, yq, indexing="xy")
                WW = wy[:, None] * wx[None, :]
                xs.append(XX.ravel())
                ys.append(YY.ravel())
                ws.append(WW.ravel())
                cells.append(np.full(XX.size, cell_id, dtype=int))
                continue
            if cell_inside_disk(x0, x1, y0, y1):
                continue
            # Cut cell: vertical panels with the disk band removed.
            xb = [x0, x1]
            for xc in (geo.cx - geo.radius, geo.cx + geo.radius):
                if x0 < xc < x1:
                    xb.append(xc)
            for yk in uy:
                if abs(yk - geo.cy) < geo.radius:
                    s = sqrt(max(r2 - (yk - geo.cy) ** 2, 0.0))
                    for xc in (geo.cx - s, geo.cx + s):
                        if x0 < xc < x1:
                            xb.append(xc)
            xbreaks = _unique_sorted(xb, x0, x1)
            for xa, xb_hi in zip(xbreaks[:-1], xbreaks[1:]):
                if xb_hi <= xa:
                    continue
                xq = 0.5 * (xb_hi - xa) * g + 0.5 * (xb_hi + xa)
                wx = 0.5 * (xb_hi - xa) * w
                for xnode, wxnode in zip(xq, wx):
                    d2 = r2 - (xnode - geo.cx) ** 2
                    if d2 <= 0.0:
                        bands = [(y0, y1)]
                    else:
                        s = sqrt(d2)
                        lo, hi = geo.cy - s, geo.cy + s
                        bands = []
                        if y0 < min(lo, y1):
                            bands.append((y0, min(lo, y1)))
                        if max(hi, y0) < y1:
                            bands.append((max(hi, y0), y1))
                    for ya, yb in bands:
                        if yb <= ya:
                            continue
                        ybreaks = _unique_sorted([yk for yk in uy if ya < yk < yb], ya, yb)
                        for yya, yyb in zip(ybreaks[:-1], ybreaks[1:]):
                            if yyb <= yya:
                                continue
                            yq = 0.5 * (yyb - yya) * g + 0.5 * (yyb + yya)
                            wy = 0.5 * (yyb - yya) * w
                            xs.append(np.full_like(yq, xnode))
                            ys.append(yq)
                            ws.append(wxnode * wy)
                            cells.append(np.full(yq.size, cell_id, dtype=int))
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ws),
        np.concatenate(cells),
    )


# ---------------------------------------------------------------------------
# Sparse 1D stiffness and sparse Marsden extension
# ---------------------------------------------------------------------------


def sparse_1d_stiffness(basis: ScaledBasis1D, order: int) -> sp.csr_matrix:
    """Exact sparse 1D stiffness S[a,b] = int Ba' Bb'."""
    x, w = _gauss_panels(basis, order)
    dva, dca = basis_derivative_banded(basis, x)
    p = basis.degree
    n = basis.n_basis
    rows = np.repeat(dca, p + 1, axis=1)
    cols = np.tile(dca, (1, p + 1))
    vals = (dva[:, :, None] * dva[:, None, :]).reshape(x.size, -1) * w[:, None]
    return sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n, n))


def sparse_marsden_extension(
    basis_x: ScaledBasis1D,
    basis_y: ScaledBasis1D,
    gamma: np.ndarray,
    support_threshold: float,
    preserve_degree: int,
    active_tol: float = 1.0e-12,
    coefficient_candidates: int = 8,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict]:
    """Sparse polynomial-preserving extension by exact Marsden coefficients.

    Interior rows (gamma >= threshold) are identity columns; small-support
    active rows are filled from a local interior stencil that reproduces every
    tensor monomial of total degree <= preserve_degree exactly (coefficients
    from the Marsden identity in locally scaled variables); inactive rows
    (gamma <= active_tol) are zero.  The stencil search is distance-greedy
    with a coefficient-aware refinement: among rank-complete candidate sets it
    keeps the one with the smallest max |weight|.

    Returns (E, interior, small_support, diagnostics).
    """
    nxb = basis_x.n_basis
    nyb = basis_y.n_basis
    n_full = nxb * nyb
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    col_of = -np.ones(n_full, dtype=int)
    col_of[interior] = np.arange(interior.size)

    ix_all = np.arange(n_full) % nxb
    iy_all = np.arange(n_full) // nxb
    gx = basis_x.greville
    gy = basis_y.greville
    interior_set = set(int(i) for i in interior)

    # tensor monomials of total degree <= preserve_degree
    powers = [(a, b) for a in range(preserve_degree + 1) for b in range(preserve_degree + 1 - a)]
    n_terms = len(powers)

    rows_E, cols_E, vals_E = list(interior), list(range(interior.size)), [1.0] * interior.size
    max_weight = 0.0
    max_stencil = 0
    max_radius = 0

    def scaled_marsden(basis: ScaledBasis1D, center: float, scale: float, indices: np.ndarray) -> np.ndarray:
        """lambda_i(((x - center)/scale)^r), r = 0..preserve_degree, for given i."""
        p = basis.degree
        t = (basis.knots - center) / scale
        out = np.zeros((preserve_degree + 1, indices.size))
        for k, i in enumerate(indices):
            window = t[i + 1 : i + p + 1]
            e = np.zeros(preserve_degree + 1)
            e[0] = 1.0
            for value in window:
                for r in range(min(preserve_degree, p), 0, -1):
                    e[r] += value * e[r - 1]
            for r in range(preserve_degree + 1):
                out[r, k] = e[r] / comb(p, r) if r <= p else 0.0
        return out

    for b in small:
        bx_i = int(ix_all[b])
        by_i = int(iy_all[b])
        cx0 = gx[bx_i]
        cy0 = gy[by_i]
        found = None
        for radius in range(2, max(nxb, nyb)):
            cand = [
                int(i)
                for i in interior
                if abs(int(ix_all[i]) - bx_i) <= radius and abs(int(iy_all[i]) - by_i) <= radius
            ]
            if len(cand) < n_terms:
                continue
            cand = sorted(
                cand,
                key=lambda i: (gx[ix_all[i]] - cx0) ** 2 + (gy[iy_all[i]] - cy0) ** 2,
            )
            # Grow the candidate pool with the search radius: the nearest
            # interior Grevilles along a locally flat cut can be nearly
            # collinear, so a fixed-size nearest-only pool can stay
            # rank-deficient at every radius.
            pool = n_terms + coefficient_candidates + 6 * (radius - 2)
            cand = np.array(cand[:pool], dtype=int)
            lx = scaled_marsden(basis_x, cx0, basis_x.h, ix_all[cand])
            ly = scaled_marsden(basis_y, cy0, basis_y.h, iy_all[cand])
            lxb = scaled_marsden(basis_x, cx0, basis_x.h, np.array([bx_i]))
            lyb = scaled_marsden(basis_y, cy0, basis_y.h, np.array([by_i]))
            C = np.array([lx[a] * ly[bb] for a, bb in powers])  # (n_terms, n_cand)
            target = np.array([lxb[a, 0] * lyb[bb, 0] for a, bb in powers])
            if np.linalg.matrix_rank(C, tol=1.0e-10) < n_terms:
                continue
            best = None
            # coefficient-aware: try nested prefixes, keep smallest max |w|
            for k in range(n_terms, len(cand) + 1):
                Ck = C[:, :k]
                if np.linalg.matrix_rank(Ck, tol=1.0e-10) < n_terms:
                    continue
                wk, *_ = np.linalg.lstsq(Ck, target, rcond=None)
                wk += np.linalg.lstsq(Ck, target - Ck @ wk, rcond=None)[0]
                resid = float(np.linalg.norm(Ck @ wk - target, ord=np.inf))
                if resid > 1.0e-9:
                    continue
                score = float(np.max(np.abs(wk)))
                if best is None or score < best[0]:
                    best = (score, k, wk)
            if best is not None:
                found = (cand[: best[1]], best[2], radius, best[0])
                break
        if found is None:
            raise RuntimeError(f"no polynomial-complete stencil found for mode {b}")
        sel, weights, radius, score = found
        for i, wgt in zip(sel, weights):
            rows_E.append(int(b))
            cols_E.append(int(col_of[i]))
            vals_E.append(float(wgt))
        max_weight = max(max_weight, score)
        max_stencil = max(max_stencil, len(sel))
        max_radius = max(max_radius, radius)

    E = sp.csr_matrix((vals_E, (rows_E, cols_E)), shape=(n_full, interior.size))
    diag = {
        "n_small_support": int(small.size),
        "n_interior": int(interior.size),
        "n_active": int(active.size),
        "max_stencil_weight": float(max_weight),
        "max_stencil_size": int(max_stencil),
        "max_stencil_radius_cells": int(max_radius),
    }
    return E, interior, small, diag


def sparse_merge_extension(
    basis_x: ScaledBasis1D,
    basis_y: ScaledBasis1D,
    gamma: np.ndarray,
    support_threshold: float,
    preserve_degree: int,
    active_tol: float = 1.0e-14,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict]:
    """Cell-agglomeration remedy: merge each small coordinate to its nearest
    interior neighbor with unit weight.

    The finite-volume merging analog inside the identical sharp assembly: the
    unit-weight row reproduces constants (so mass balances survive) but no
    higher monomial, which is exactly the accuracy mechanism the ablation
    isolates.  ``preserve_degree`` is accepted for signature parity and
    ignored.
    """
    nxb = basis_x.n_basis
    n_full = nxb * basis_y.n_basis
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    col_of = -np.ones(n_full, dtype=int)
    col_of[interior] = np.arange(interior.size)
    gx = basis_x.greville
    gy = basis_y.greville
    ix_all = np.arange(n_full) % nxb
    iy_all = np.arange(n_full) // nxb
    px = gx[ix_all[interior]]
    py = gy[iy_all[interior]]
    rows = list(interior)
    cols = list(range(interior.size))
    vals = [1.0] * interior.size
    for b in small:
        d2 = (px - gx[ix_all[b]]) ** 2 + (py - gy[iy_all[b]]) ** 2
        j = int(np.argmin(d2))
        rows.append(int(b))
        cols.append(j)
        vals.append(1.0)
    E = sp.csr_matrix((vals, (rows, cols)), shape=(n_full, interior.size))
    diag = {
        "n_small_support": int(small.size),
        "n_interior": int(interior.size),
        "n_active": int(active.size),
        "remedy": "merge",
    }
    return E, interior, small, diag


def sparse_delete_extension(
    basis_x: ScaledBasis1D,
    basis_y: ScaledBasis1D,
    gamma: np.ndarray,
    support_threshold: float,
    preserve_degree: int,
    active_tol: float = 1.0e-14,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict]:
    """Plain-deletion remedy: drop every small-support coordinate.

    The naive thresholding baseline: small rows are zero, so the partition of
    unity over the cut is broken and no polynomial is reproduced near the
    boundary.  ``preserve_degree`` is accepted for signature parity and
    ignored.
    """
    nxb = basis_x.n_basis
    n_full = nxb * basis_y.n_basis
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    E = sp.csr_matrix(
        (np.ones(interior.size), (interior, np.arange(interior.size))),
        shape=(n_full, interior.size),
    )
    diag = {
        "n_small_support": int(small.size),
        "n_interior": int(interior.size),
        "n_active": int(active.size),
        "remedy": "delete",
    }
    return E, interior, small, diag


# ---------------------------------------------------------------------------
# Channel operator bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelOperators:
    """Sparse aggregated mixed operator blocks for the channel-cylinder flow."""

    p: int
    nx_elements: int
    ny_elements: int
    hx: float
    hy: float
    geo: ChannelGeometry
    beta: float
    q_op: int
    q_cap: int
    basis_ux: ScaledBasis1D
    basis_uy: ScaledBasis1D
    basis_px: ScaledBasis1D
    basis_py: ScaledBasis1D
    E_u: sp.csr_matrix
    E_p: sp.csr_matrix
    H_u: sp.csr_matrix
    Diff: sp.csr_matrix
    Gx: sp.csr_matrix
    Gy: sp.csr_matrix
    # Dirichlet boundary (inflow + walls + arc) data for weak-penalty RHS
    bx: np.ndarray
    by: np.ndarray
    bw: np.ndarray
    bnx: np.ndarray
    bny: np.ndarray
    phi_b: sp.csr_matrix
    dphin_b: sp.csr_matrix
    psi_b: sp.csr_matrix
    # full-boundary rule (adds outflow) for the skew boundary mass
    fb_x: np.ndarray
    fb_y: np.ndarray
    fb_w: np.ndarray
    fb_nx: np.ndarray
    fb_ny: np.ndarray
    phi_fb: sp.csr_matrix
    # cylinder-arc rule for forces
    arc_x: np.ndarray
    arc_y: np.ndarray
    arc_w: np.ndarray
    arc_nx: np.ndarray
    arc_ny: np.ndarray
    phi_arc: sp.csr_matrix
    dphix_arc: sp.csr_matrix
    dphiy_arc: sp.csr_matrix
    psi_arc: sp.csr_matrix
    # captured positive volume rule
    qx: np.ndarray
    qy: np.ndarray
    qw: np.ndarray
    phi_q: sp.csr_matrix
    gx_q: sp.csr_matrix
    gy_q: sp.csr_matrix
    psi_q: sp.csr_matrix
    # reduced evaluation matrices used by the time-stepper
    phi_q_E: sp.csr_matrix
    gx_q_E: sp.csr_matrix
    gy_q_E: sp.csr_matrix
    phi_fb_E: sp.csr_matrix
    phi_arc_E: sp.csr_matrix
    dphix_arc_E: sp.csr_matrix
    dphiy_arc_E: sp.csr_matrix
    psi_arc_Ep: sp.csr_matrix
    # bookkeeping
    n_u_red: int
    n_p_red: int
    min_cell_volume_fraction: float
    extension_diag_u: dict
    extension_diag_p: dict

    @property
    def h_min(self) -> float:
        return min(self.hx, self.hy)


def assemble_channel_operators(
    p: int = 3,
    nx_elements: int = 220,
    ny_elements: int = 41,
    geo: ChannelGeometry | None = None,
    beta_factor: float = 10.0,
    q_order: int | None = None,
    q_vol: int | None = None,
    support_threshold: float = 0.005,
    support_threshold_p: float = 0.04,
    extension_kind: str = "polynomial",
) -> ChannelOperators:
    """Assemble all sparse blocks for the channel-cylinder mixed pair.

    ``extension_kind`` selects the small-cell remedy applied to both spaces
    inside the otherwise identical pipeline: ``"polynomial"`` is the
    polynomial-preserving aggregation of this work, ``"merge"`` the
    cell-agglomeration analog (unit-weight nearest neighbor), and
    ``"delete"`` the naive thresholding baseline.
    """
    geo = geo or ChannelGeometry()
    basis_ux = make_scaled_basis(nx_elements, p, geo.length)
    basis_uy = make_scaled_basis(ny_elements, p, geo.height)
    basis_px = make_scaled_basis(nx_elements, p - 1, geo.length)
    basis_py = make_scaled_basis(ny_elements, p - 1, geo.height)
    hx = basis_ux.h
    hy = basis_uy.h
    q_op = q_order if q_order is not None else (2 * p + 12)
    q_cap = q_vol if q_vol is not None else (p + 2)

    g1d = 2 * p + 3  # exact for all 1D Gram integrands

    # --- 1D sparse Grams ---------------------------------------------------
    Mx_uu, _ = sparse_1d_grams(basis_ux, basis_ux, g1d)
    My_uu, _ = sparse_1d_grams(basis_uy, basis_uy, g1d)
    Kx_uu = sparse_1d_stiffness(basis_ux, g1d)
    Ky_uu = sparse_1d_stiffness(basis_uy, g1d)
    Mx_up, Dx_up = sparse_1d_grams(basis_ux, basis_px, g1d)
    My_up, Dy_up = sparse_1d_grams(basis_uy, basis_py, g1d)
    Mx_pp, _ = sparse_1d_grams(basis_px, basis_px, g1d)
    My_pp, _ = sparse_1d_grams(basis_py, basis_py, g1d)

    # --- rectangle Kron parts ------------------------------------------------
    H_full = sp.kron(My_uu, Mx_uu, format="csr")
    K_full = sp.kron(My_uu, Kx_uu, format="csr") + sp.kron(Ky_uu, Mx_uu, format="csr")
    Gx_full = -sp.kron(My_up, Dx_up, format="csr")
    Gy_full = -sp.kron(Dy_up, Mx_up, format="csr")
    Hp_full = sp.kron(My_pp, Mx_pp, format="csr")
    H_bg_diag = H_full.diagonal().copy()
    Hp_bg_diag = Hp_full.diagonal().copy()

    # --- hole corrections (smooth polar rule) -------------------------------
    xh, yh, wh = disk_polar_quadrature(geo, basis_ux.knots, basis_uy.knots, q_op)
    phi_h, dphix_h, dphiy_h = sparse_tensor_eval(basis_ux, basis_uy, xh, yh)
    psi_h = sparse_tensor_eval(basis_px, basis_py, xh, yh, derivatives=False)[0]
    Wh = sp.diags(wh)
    H_full = (H_full - (phi_h.T @ Wh @ phi_h)).tocsr()
    K_full = (K_full - (dphix_h.T @ Wh @ dphix_h) - (dphiy_h.T @ Wh @ dphiy_h)).tocsr()
    Gx_full = (Gx_full + (dphix_h.T @ Wh @ psi_h)).tocsr()
    Gy_full = (Gy_full + (dphiy_h.T @ Wh @ psi_h)).tocsr()
    Hp_full = (Hp_full - (psi_h.T @ Wh @ psi_h)).tocsr()

    # --- boundary rules ------------------------------------------------------
    # Dirichlet parts: inflow x=0 (n=(-1,0)), walls y=0 (n=(0,-1)) and y=H
    # (n=(0,1)), and the cylinder arc (n = -e_r).  Outflow x=L is free.
    parts = []
    ty, wy_1d = _side_quadrature_1d(basis_uy, q_op)
    parts.append((np.zeros_like(ty), ty, wy_1d, -np.ones_like(ty), np.zeros_like(ty)))
    tx, wx_1d = _side_quadrature_1d(basis_ux, q_op)
    parts.append((tx, np.zeros_like(tx), wx_1d, np.zeros_like(tx), -np.ones_like(tx)))
    parts.append((tx, np.full_like(tx, geo.height), wx_1d, np.zeros_like(tx), np.ones_like(tx)))
    ax, ay, aw, anx, any_ = arc_quadrature(geo, basis_ux.knots, basis_uy.knots, q_op)
    parts.append((ax, ay, aw, anx, any_))
    bx = np.concatenate([part[0] for part in parts])
    by = np.concatenate([part[1] for part in parts])
    bw = np.concatenate([part[2] for part in parts])
    bnx = np.concatenate([part[3] for part in parts])
    bny = np.concatenate([part[4] for part in parts])
    phi_b, dphix_b, dphiy_b = sparse_tensor_eval(basis_ux, basis_uy, bx, by)
    dphin_b = (sp.diags(bnx) @ dphix_b + sp.diags(bny) @ dphiy_b).tocsr()
    psi_b = sparse_tensor_eval(basis_px, basis_py, bx, by, derivatives=False)[0]

    # full boundary (adds outflow) for the skew convection boundary mass
    fb_parts = parts + [
        (np.full_like(ty, geo.length), ty, wy_1d, np.ones_like(ty), np.zeros_like(ty))
    ]
    fb_x = np.concatenate([part[0] for part in fb_parts])
    fb_y = np.concatenate([part[1] for part in fb_parts])
    fb_w = np.concatenate([part[2] for part in fb_parts])
    fb_nx = np.concatenate([part[3] for part in fb_parts])
    fb_ny = np.concatenate([part[4] for part in fb_parts])
    phi_fb = sparse_tensor_eval(basis_ux, basis_uy, fb_x, fb_y, derivatives=False)[0]

    # --- weak-penalty blocks ------------------------------------------------------
    Wb = sp.diags(bw)
    M_bnd = (phi_b.T @ Wb @ phi_b).tocsr()
    N_flux = (phi_b.T @ Wb @ dphin_b).tocsr()
    beta = beta_factor * (p + 1) ** 2
    h_min = min(hx, hy)
    Diff_full = (K_full - N_flux - N_flux.T + (beta / h_min) * M_bnd).tocsr()

    # --- G boundary part on the Dirichlet boundary --------------------------
    Gx_full = (Gx_full + phi_b.T @ sp.diags(bw * bnx) @ psi_b).tocsr()
    Gy_full = (Gy_full + phi_b.T @ sp.diags(bw * bny) @ psi_b).tocsr()

    # --- active sets and sparse extensions -----------------------------------
    gamma_u = np.zeros(H_full.shape[0])
    pos = H_bg_diag > 0
    gamma_u[pos] = H_full.diagonal()[pos] / H_bg_diag[pos]
    gamma_p = np.zeros(Hp_full.shape[0])
    pos_p = Hp_bg_diag > 0
    gamma_p[pos_p] = Hp_full.diagonal()[pos_p] / Hp_bg_diag[pos_p]
    # Every strictly positive-mass row is kept (extended); only rows at the
    # cancellation-noise floor are excluded, so the discrete partition of
    # unity over the cut stays complete and the assembled Gauss identities
    # hold at quadrature accuracy independently of the active tolerance.
    builder = {
        "polynomial": sparse_marsden_extension,
        "merge": sparse_merge_extension,
        "delete": sparse_delete_extension,
    }[extension_kind]
    E_u, _, _, diag_u = builder(
        basis_ux, basis_uy, gamma_u, support_threshold, p, active_tol=1.0e-14
    )
    E_p, _, _, diag_p = builder(
        basis_px, basis_py, gamma_p, support_threshold_p, p - 1, active_tol=1.0e-14
    )

    # --- pulled-back blocks ---------------------------------------------------
    H_u = (E_u.T @ H_full @ E_u).tocsr()
    Diff = (E_u.T @ Diff_full @ E_u).tocsr()
    Diff = (0.5 * (Diff + Diff.T)).tocsr()
    H_u = (0.5 * (H_u + H_u.T)).tocsr()
    Gx = (E_u.T @ Gx_full @ E_p).tocsr()
    Gy = (E_u.T @ Gy_full @ E_p).tocsr()

    # --- captured positive volume rule (convection, loads, norms) -----------
    qx, qy, qw, qcell = positive_channel_volume_quadrature(basis_ux, basis_uy, geo, q_cap)
    phi_q, gx_q, gy_q = sparse_tensor_eval(basis_ux, basis_uy, qx, qy)
    psi_q = sparse_tensor_eval(basis_px, basis_py, qx, qy, derivatives=False)[0]
    cell_area = np.zeros(nx_elements * ny_elements)
    np.add.at(cell_area, qcell, qw)
    nz = cell_area > 1.0e-22
    fractions = cell_area[nz] / (hx * hy)
    min_frac = float(np.min(fractions[fractions < 1.0 - 1.0e-9])) if np.any(fractions < 1.0 - 1.0e-9) else 1.0

    # arc rule for forces (reuse the operator-order arc points)
    psi_arc = sparse_tensor_eval(basis_px, basis_py, ax, ay, derivatives=False)[0]
    phi_arc, dphix_arc, dphiy_arc = sparse_tensor_eval(basis_ux, basis_uy, ax, ay)
    phi_q_E = (phi_q @ E_u).tocsr()
    gx_q_E = (gx_q @ E_u).tocsr()
    gy_q_E = (gy_q @ E_u).tocsr()
    phi_fb_E = (phi_fb @ E_u).tocsr()
    phi_arc_E = (phi_arc @ E_u).tocsr()
    dphix_arc_E = (dphix_arc @ E_u).tocsr()
    dphiy_arc_E = (dphiy_arc @ E_u).tocsr()
    psi_arc_Ep = (psi_arc @ E_p).tocsr()

    return ChannelOperators(
        p=p,
        nx_elements=nx_elements,
        ny_elements=ny_elements,
        hx=hx,
        hy=hy,
        geo=geo,
        beta=beta,
        q_op=q_op,
        q_cap=q_cap,
        basis_ux=basis_ux,
        basis_uy=basis_uy,
        basis_px=basis_px,
        basis_py=basis_py,
        E_u=E_u,
        E_p=E_p,
        H_u=H_u,
        Diff=Diff,
        Gx=Gx,
        Gy=Gy,
        bx=bx,
        by=by,
        bw=bw,
        bnx=bnx,
        bny=bny,
        phi_b=phi_b,
        dphin_b=dphin_b,
        psi_b=psi_b,
        fb_x=fb_x,
        fb_y=fb_y,
        fb_w=fb_w,
        fb_nx=fb_nx,
        fb_ny=fb_ny,
        phi_fb=phi_fb,
        arc_x=ax,
        arc_y=ay,
        arc_w=aw,
        arc_nx=anx,
        arc_ny=any_,
        phi_arc=phi_arc,
        dphix_arc=dphix_arc,
        dphiy_arc=dphiy_arc,
        psi_arc=psi_arc,
        qx=qx,
        qy=qy,
        qw=qw,
        phi_q=phi_q,
        gx_q=gx_q,
        gy_q=gy_q,
        psi_q=psi_q,
        phi_q_E=phi_q_E,
        gx_q_E=gx_q_E,
        gy_q_E=gy_q_E,
        phi_fb_E=phi_fb_E,
        phi_arc_E=phi_arc_E,
        dphix_arc_E=dphix_arc_E,
        dphiy_arc_E=dphiy_arc_E,
        psi_arc_Ep=psi_arc_Ep,
        n_u_red=int(E_u.shape[1]),
        n_p_red=int(E_p.shape[1]),
        min_cell_volume_fraction=min_frac,
        extension_diag_u=diag_u,
        extension_diag_p=diag_p,
    )


# ---------------------------------------------------------------------------
# Time-dependent solver: AB2 predictor / trapezoidal corrector + CN diffusion
# ---------------------------------------------------------------------------


def inflow_profile(geo: ChannelGeometry, um: float, y: np.ndarray) -> np.ndarray:
    return 4.0 * um * y * (geo.height - y) / geo.height**2


def _smooth_ramp(t: float, t_ramp: float) -> float:
    if t_ramp <= 0.0 or t >= t_ramp:
        return 1.0
    return 0.5 * (1.0 - np.cos(pi * t / t_ramp))


@dataclass
class CylinderSeries:
    times: np.ndarray
    drag: np.ndarray
    lift: np.ndarray
    delta_p: np.ndarray
    kinetic_energy: np.ndarray
    divergence_residual: np.ndarray


@dataclass
class CylinderResult:
    p: int
    nx_elements: int
    ny_elements: int
    dt: float
    t_final: float
    reynolds: float
    n_velocity: int
    n_pressure: int
    min_cell_volume_fraction: float
    strouhal: float
    cd_max: float
    cd_mean: float
    cl_max: float
    cl_min: float
    delta_p_at_clmax: float
    max_divergence_residual: float
    n_periods_measured: int
    series: CylinderSeries
    net_mass_flux_final: float = float("nan")


class ChannelCylinderSolver:
    """AB2/trapezoid-corrector + Crank-Nicolson stepping of the channel flow."""

    def __init__(
        self,
        ops: ChannelOperators,
        nu: float = 1.0e-3,
        um: float = 1.5,
        dt: float = 5.0e-4,
        t_ramp: float = 1.0,
        correctors: int = 1,
    ) -> None:
        self.ops = ops
        self.nu = float(nu)
        self.um = float(um)
        self.dt = float(dt)
        self.t_ramp = float(t_ramp)
        self.correctors = int(correctors)
        nu_red = ops.n_u_red
        np_red = ops.n_p_red
        self.nu_red = nu_red
        self.np_red = np_red

        A = (ops.H_u / dt + 0.5 * nu * ops.Diff).tocsr()
        S = sp.bmat(
            [
                [A, None, ops.Gx],
                [None, A, ops.Gy],
                [ops.Gx.T, ops.Gy.T, None],
            ],
            format="csc",
        )
        self.S = S
        self.solver = spla.splu(S)
        self.B_explicit = (ops.H_u / dt - 0.5 * nu * ops.Diff).tocsr()

        # Steady boundary-data shapes (scaled by the ramp factor in time).
        g_in = np.where(ops.bx < 1.0e-12, inflow_profile(ops.geo, um, ops.by), 0.0)
        self.gu_shape_b = g_in  # v-data is zero everywhere
        pen = ops.phi_b.T @ (ops.bw * g_in) * (ops.beta / ops.h_min)
        cons = ops.dphin_b.T @ (ops.bw * g_in)
        self.rx_shape = np.asarray(ops.E_u.T @ (nu * (pen - cons))).ravel()
        self.hp_shape = np.asarray(ops.E_p.T @ (ops.psi_b.T @ (ops.bw * g_in * ops.bnx))).ravel()
        # full-boundary inflow samples for the skew boundary mass
        self.gu_shape_fb = np.where(ops.fb_x < 1.0e-12, inflow_profile(ops.geo, um, ops.fb_y), 0.0)

        # state
        self.cu = np.zeros(nu_red)
        self.cv = np.zeros(nu_red)
        self.cp = np.zeros(np_red)
        self.t = 0.0
        self.conv_prev: tuple[np.ndarray, np.ndarray] | None = None

        # pressure probes (front / rear of the cylinder)
        geo = ops.geo
        probe_x = np.array([geo.cx - geo.radius, geo.cx + geo.radius])
        probe_y = np.array([geo.cy, geo.cy])
        psi_probe = sparse_tensor_eval(ops.basis_px, ops.basis_py, probe_x, probe_y, derivatives=False)[0]
        self.psi_probe_Ep = (psi_probe @ ops.E_p).tocsr()

    # -- skew convection (matrix-free) ------------------------------------
    def _convection(self, cu: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ops = self.ops
        wx = ops.phi_q_E @ cu
        wy = ops.phi_q_E @ cv
        dux = ops.gx_q_E @ cu
        duy = ops.gy_q_E @ cu
        dvx = ops.gx_q_E @ cv
        dvy = ops.gy_q_E @ cv
        qw = ops.qw
        # C(w) u and C(w) v
        cu_adv = ops.phi_q_E.T @ (qw * (wx * dux + wy * duy))
        cv_adv = ops.phi_q_E.T @ (qw * (wx * dvx + wy * dvy))
        # C(w)^T u and C(w)^T v
        pu = wx
        pv = wy
        cu_adj = ops.gx_q_E.T @ (qw * wx * pu) + ops.gy_q_E.T @ (qw * wy * pu)
        cv_adj = ops.gx_q_E.T @ (qw * wx * pv) + ops.gy_q_E.T @ (qw * wy * pv)
        # boundary mass weighted by w.n on the full boundary
        ub = ops.phi_fb_E @ cu
        vb = ops.phi_fb_E @ cv
        wn = ub * ops.fb_nx + vb * ops.fb_ny
        bu = ops.phi_fb_E.T @ (ops.fb_w * wn * ub)
        bv = ops.phi_fb_E.T @ (ops.fb_w * wn * vb)
        fu = 0.5 * (cu_adv - cu_adj) + 0.5 * bu
        fv = 0.5 * (cv_adv - cv_adj) + 0.5 * bv
        return np.asarray(fu).ravel(), np.asarray(fv).ravel()

    def _solve_saddle(self, conv_u, conv_v, ramp_mid, ramp_new, refine: bool = True):
        rhs_u = self.B_explicit @ self.cu + ramp_mid * self.rx_shape - conv_u
        rhs_v = self.B_explicit @ self.cv - conv_v
        rhs = np.concatenate([rhs_u, rhs_v, ramp_new * self.hp_shape])
        sol = self.solver.solve(rhs)
        if refine:
            sol += self.solver.solve(rhs - self.S @ sol)
        return sol

    def step(self) -> None:
        dt = self.dt
        t_new = self.t + dt
        ramp_mid = _smooth_ramp(self.t + 0.5 * dt, self.t_ramp)
        ramp_new = _smooth_ramp(t_new, self.t_ramp)

        conv_n = self._convection(self.cu, self.cv)
        if self.conv_prev is None:
            conv_u, conv_v = conv_n
        else:
            conv_u = 1.5 * conv_n[0] - 0.5 * self.conv_prev[0]
            conv_v = 1.5 * conv_n[1] - 0.5 * self.conv_prev[1]
        sol = self._solve_saddle(conv_u, conv_v, ramp_mid, ramp_new, refine=self.correctors == 0)
        cu_new = sol[: self.nu_red]
        cv_new = sol[self.nu_red : 2 * self.nu_red]

        for _ in range(self.correctors):
            conv_np1 = self._convection(cu_new, cv_new)
            conv_u = 0.5 * (conv_n[0] + conv_np1[0])
            conv_v = 0.5 * (conv_n[1] + conv_np1[1])
            sol = self._solve_saddle(conv_u, conv_v, ramp_mid, ramp_new)
            cu_new = sol[: self.nu_red]
            cv_new = sol[self.nu_red : 2 * self.nu_red]

        self.cu = cu_new
        self.cv = cv_new
        self.cp = sol[2 * self.nu_red :]
        self.conv_prev = conv_n
        self.t = t_new

    # -- diagnostics --------------------------------------------------------
    def kinetic_energy(self) -> float:
        return 0.5 * float(self.cu @ (self.ops.H_u @ self.cu) + self.cv @ (self.ops.H_u @ self.cv))

    def divergence_residual(self) -> float:
        ramp = _smooth_ramp(self.t, self.t_ramp)
        r = self.ops.Gx.T @ self.cu + self.ops.Gy.T @ self.cv - ramp * self.hp_shape
        return float(np.linalg.norm(r, ord=np.inf))

    def forces(self) -> tuple[float, float]:
        """Force on the cylinder from the consistent weak-penalty traction.

        Domain outward normal n_d points into the cylinder; the force on the
        body is the negative of the integrated discrete traction
        nu (n_d . grad) u_h - p_h n_d - (nu beta / h)(u_h - g), with g = 0 on
        the arc.
        """
        ops = self.ops
        dun = (
            ops.arc_nx * (ops.dphix_arc_E @ self.cu)
            + ops.arc_ny * (ops.dphiy_arc_E @ self.cu)
        )
        dvn = (
            ops.arc_nx * (ops.dphix_arc_E @ self.cv)
            + ops.arc_ny * (ops.dphiy_arc_E @ self.cv)
        )
        p_arc = ops.psi_arc_Ep @ self.cp
        u_arc = ops.phi_arc_E @ self.cu
        v_arc = ops.phi_arc_E @ self.cv
        pen = self.nu * ops.beta / ops.h_min
        tx = self.nu * dun - p_arc * ops.arc_nx - pen * u_arc
        ty = self.nu * dvn - p_arc * ops.arc_ny - pen * v_arc
        fx = -float(np.sum(ops.arc_w * tx))
        fy = -float(np.sum(ops.arc_w * ty))
        return fx, fy

    def net_mass_flux(self) -> float:
        """Net boundary mass flux of the current state over the full boundary.

        Incompressibility plus the boundary data make the exact value zero
        after the ramp; a small-cell remedy that breaks the partition of
        unity (deletion) loses the constant-pressure test function and with
        it the discrete global mass balance, so this diagnostic separates
        conservative from non-conservative remedies.
        """
        ops = self.ops
        au = ops.E_u @ self.cu
        av = ops.E_u @ self.cv
        un = (ops.phi_fb @ au) * ops.fb_nx + (ops.phi_fb @ av) * ops.fb_ny
        return float(np.sum(ops.fb_w * un))

    def pressure_probes(self) -> tuple[float, float]:
        vals = self.psi_probe_Ep @ self.cp
        return float(vals[0]), float(vals[1])


def _strouhal_from_lift(times: np.ndarray, lift: np.ndarray, geo: ChannelGeometry, u_mean: float) -> tuple[float, int]:
    """Shedding frequency from upward zero crossings of the lift over the tail."""
    if times.size < 16:
        return float("nan"), 0
    t0 = times[0] + 0.5 * (times[-1] - times[0])
    mask = times >= t0
    t = times[mask]
    c = lift[mask] - np.mean(lift[mask])
    crossings = []
    for i in range(1, t.size):
        if c[i - 1] < 0.0 <= c[i]:
            frac = -c[i - 1] / (c[i] - c[i - 1])
            crossings.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    if len(crossings) < 3:
        return float("nan"), max(0, len(crossings) - 1)
    periods = np.diff(crossings)
    period = float(np.mean(periods))
    freq = 1.0 / period
    return freq * (2.0 * geo.radius) / u_mean, len(periods)


def run_cylinder_benchmark(
    p: int = 3,
    nx_elements: int = 220,
    ny_elements: int = 41,
    nu: float = 1.0e-3,
    um: float = 1.5,
    dt: float = 5.0e-4,
    t_final: float = 8.0,
    t_ramp: float = 1.0,
    correctors: int = 1,
    record_every: int = 4,
    q_order: int | None = None,
    q_vol: int | None = None,
    verbose: bool = True,
    ops: ChannelOperators | None = None,
    state_path: Path | None = None,
    checkpoint_every: int = 2000,
    resume_from: Path | None = None,
    geo: ChannelGeometry | None = None,
    extension_kind: str = "polynomial",
) -> CylinderResult:
    """Run the 2D-2 cylinder benchmark and extract the headline numbers."""
    import time as _time

    if ops is None:
        ops = assemble_channel_operators(
            p=p,
            nx_elements=nx_elements,
            ny_elements=ny_elements,
            geo=geo,
            q_order=q_order,
            q_vol=q_vol,
            extension_kind=extension_kind,
        )
    solver = ChannelCylinderSolver(ops, nu=nu, um=um, dt=dt, t_ramp=t_ramp, correctors=correctors)

    u_mean = 2.0 * um / 3.0
    force_scale = 2.0 / (u_mean**2 * 2.0 * ops.geo.radius)

    times, drags, lifts, dps, kes, divs = [], [], [], [], [], []
    if resume_from is not None and resume_from.exists():
        state = np.load(resume_from)
        if int(state["p"]) != p or int(state["nx_elements"]) != nx_elements or int(state["ny_elements"]) != ny_elements:
            raise ValueError("checkpoint grid does not match the requested cylinder run")
        solver.cu = np.asarray(state["cu"])
        solver.cv = np.asarray(state["cv"])
        solver.cp = np.asarray(state["cp"])
        solver.t = float(state["t"])
        solver.conv_prev = None
        for name, target in (
            ("times", times),
            ("drag", drags),
            ("lift", lifts),
            ("delta_p", dps),
            ("kinetic_energy", kes),
            ("divergence_residual", divs),
        ):
            if name in state.files:
                target.extend(np.asarray(state[name], dtype=float).tolist())
        if verbose:
            print(f"[cylinder] resumed from t={solver.t:.3f} with {len(times)} samples", flush=True)

    def _save_state() -> None:
        if state_path is not None:
            np.savez_compressed(
                state_path,
                cu=solver.cu,
                cv=solver.cv,
                cp=solver.cp,
                t=solver.t,
                p=p,
                nx_elements=nx_elements,
                ny_elements=ny_elements,
                times=np.asarray(times),
                drag=np.asarray(drags),
                lift=np.asarray(lifts),
                delta_p=np.asarray(dps),
                kinetic_energy=np.asarray(kes),
                divergence_residual=np.asarray(divs),
            )

    n_steps = max(0, int(round((t_final - solver.t) / dt)))
    t_wall = _time.time()
    for step in range(1, n_steps + 1):
        solver.step()
        if step % record_every == 0 or step == n_steps:
            fx, fy = solver.forces()
            pf, pr = solver.pressure_probes()
            times.append(solver.t)
            drags.append(force_scale * fx)
            lifts.append(force_scale * fy)
            dps.append(pf - pr)
            kes.append(solver.kinetic_energy())
            divs.append(solver.divergence_residual())
            if not np.isfinite(kes[-1]) or kes[-1] > 1.0e8:
                if verbose:
                    print(f"[cylinder] blow-up at t={solver.t:.3f}")
                break
        if step % checkpoint_every == 0:
            _save_state()
        if verbose and step % max(1, n_steps // 20) == 0:
            print(
                f"[cylinder] t={solver.t:6.3f}  cD={drags[-1] if drags else 0:7.4f} "
                f"cL={lifts[-1] if lifts else 0:8.4f}  KE={kes[-1] if kes else 0:8.5f} "
                f"div={divs[-1] if divs else 0:.2e}  [{_time.time()-t_wall:6.1f}s]",
                flush=True,
            )

    _save_state()
    times_a = np.asarray(times)
    drags_a = np.asarray(drags)
    lifts_a = np.asarray(lifts)
    dps_a = np.asarray(dps)
    st, n_periods = _strouhal_from_lift(times_a, lifts_a, ops.geo, u_mean)

    # tail metrics over the measured shedding periods
    tail = times_a >= times_a[0] + 0.5 * (times_a[-1] - times_a[0])
    cd_max = float(np.max(drags_a[tail])) if np.any(tail) else float("nan")
    cd_mean = float(np.mean(drags_a[tail])) if np.any(tail) else float("nan")
    cl_max = float(np.max(lifts_a[tail])) if np.any(tail) else float("nan")
    cl_min = float(np.min(lifts_a[tail])) if np.any(tail) else float("nan")
    if np.any(tail):
        i_clmax = np.argmax(np.where(tail, lifts_a, -np.inf))
        dp_at = float(dps_a[i_clmax])
    else:
        dp_at = float("nan")

    series = CylinderSeries(
        times=times_a,
        drag=drags_a,
        lift=lifts_a,
        delta_p=dps_a,
        kinetic_energy=np.asarray(kes),
        divergence_residual=np.asarray(divs),
    )
    return CylinderResult(
        p=p,
        nx_elements=nx_elements,
        ny_elements=ny_elements,
        dt=dt,
        t_final=float(times_a[-1]) if times_a.size else 0.0,
        reynolds=u_mean * 2.0 * ops.geo.radius / nu,
        n_velocity=2 * ops.n_u_red,
        n_pressure=ops.n_p_red,
        min_cell_volume_fraction=ops.min_cell_volume_fraction,
        strouhal=st,
        cd_max=cd_max,
        cd_mean=cd_mean,
        cl_max=cl_max,
        cl_min=cl_min,
        delta_p_at_clmax=dp_at,
        max_divergence_residual=float(np.max(divs)) if divs else float("nan"),
        n_periods_measured=n_periods,
        net_mass_flux_final=solver.net_mass_flux(),
        series=series,
    )


REFERENCE_INTERVALS = {
    "cd_max": (3.22, 3.24),
    "cl_max": (0.99, 1.01),
    "strouhal": (0.295, 0.305),
    "delta_p": (2.46, 2.50),
}


def write_cylinder_artifacts(result: CylinderResult, tag: str = "") -> dict:
    """Write the benchmark series, summary, and snapshot inputs to results/."""
    import json

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    series = result.series
    payload = {
        "times": series.times.tolist(),
        "drag": series.drag.tolist(),
        "lift": series.lift.tolist(),
        "delta_p": series.delta_p.tolist(),
        "kinetic_energy": series.kinetic_energy.tolist(),
        "divergence_residual": series.divergence_residual.tolist(),
    }
    summary = {
        "p": result.p,
        "nx_elements": result.nx_elements,
        "ny_elements": result.ny_elements,
        "dt": result.dt,
        "t_final": result.t_final,
        "n_steps": int(round(result.t_final / result.dt)) if result.dt > 0.0 else 0,
        "n_diagnostic_samples": int(series.times.size),
        "reynolds": result.reynolds,
        "n_velocity": result.n_velocity,
        "n_pressure": result.n_pressure,
        "min_cell_volume_fraction": result.min_cell_volume_fraction,
        "strouhal": result.strouhal,
        "cd_max": result.cd_max,
        "cd_mean": result.cd_mean,
        "cl_max": result.cl_max,
        "cl_min": result.cl_min,
        "cl_max_abs": max(abs(result.cl_max), abs(result.cl_min)),
        "delta_p_at_clmax": result.delta_p_at_clmax,
        "max_divergence_residual": result.max_divergence_residual,
        "n_periods_measured": result.n_periods_measured,
        "reference_intervals": REFERENCE_INTERVALS,
    }
    (RESULTS / f"cylinder_benchmark{suffix}.json").write_text(json.dumps(payload, indent=2))
    (RESULTS / f"cylinder_benchmark{suffix}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--nx", type=int, default=220)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--nu", type=float, default=1.0e-3)
    parser.add_argument("--um", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--t-final", type=float, default=9.0)
    parser.add_argument("--t-ramp", type=float, default=1.0)
    parser.add_argument("--correctors", type=int, default=1)
    parser.add_argument("--q-order", type=int, default=None)
    parser.add_argument("--q-vol", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run_cylinder_benchmark(
        p=args.p,
        nx_elements=args.nx,
        ny_elements=args.ny,
        nu=args.nu,
        um=args.um,
        dt=args.dt,
        t_final=args.t_final,
        t_ramp=args.t_ramp,
        correctors=args.correctors,
        record_every=args.record_every,
        q_order=args.q_order,
        q_vol=args.q_vol,
        verbose=True,
        state_path=RESULTS / f"cylinder_state{suffix}.npz",
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
    )
    summary = write_cylinder_artifacts(result, tag=args.tag)
    print(json.dumps(summary, indent=2))
