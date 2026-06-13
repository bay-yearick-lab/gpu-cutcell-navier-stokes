"""Sparse 3D machinery for the hollow-cube acoustic bell benchmark.

The geometry is a thick-walled hollow cube (every wall at least degree+2
background cells thick) with a square mouth opening through one wall, embedded
in the closed unit-box cavity and translated by a common sub-cell offset
epsilon*h.  The walls bury background B-splines exactly as in the 2D Block-I
bell, the concave corner pockets carry fluid fraction epsilon^3, and the
volume quadrature is exact because every cut is rectilinear: each background
cell splits into fluid sub-boxes on which all forms factor into 1D Gauss
integrals per axis.

Everything is assembled sparse.  The aggregated operators stay explicit CSR;
the BMF^cut stiffness K = sum_i C_i^T H_{1,i}^{-1} C_i is applied matrix-free
through one sparse LU factorization per directional flux mass.  Time
integration, ledgers, and spectral diagnostics mirror the 2D bell: one
velocity-Verlet loop for every method, the staggered discrete energy, the
fluid-mass/wall-flux functional, and the probe resonance fingerprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cut_assembly import basis_value_table, element_support
from cut_mimetic_routes import make_derivative_basis
from cut_spline_sbp import SplineBasis1D, make_open_uniform_basis, polynomial_coefficients


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float

    def contains(self, x: float, y: float, z: float, *, tol: float = 1.0e-14) -> bool:
        return (
            (self.x0 - tol <= x <= self.x1 + tol)
            and (self.y0 - tol <= y <= self.y1 + tol)
            and (self.z0 - tol <= z <= self.z1 + tol)
        )


@dataclass(frozen=True)
class Bell3DGeometry:
    """Hollow thick-walled cube with a mouth, anchored to a reference grid."""

    n_geometry: int
    epsilon: float
    boxes: tuple[Box, ...]

    @staticmethod
    def from_grid(n_geometry: int, epsilon: float) -> "Bell3DGeometry":
        if not (0.0 < epsilon < 0.35):
            raise ValueError("epsilon must be in (0, 0.35)")
        n = int(n_geometry)
        if n < 20:
            raise ValueError("the 3D bell needs n_geometry >= 20")
        h = 1.0 / n
        s = epsilon * h
        # Shell outer extent [4h, 16h]^3 scaled to n=20 proportions, wall
        # thickness 4 cells, inner chamber 4 cells across, mouth 2x2 cells
        # through the x-min wall.
        a = round(0.2 * n)
        w = round(0.2 * n)
        b = n - a
        c0 = a + w
        c1 = b - w
        m0 = round(0.45 * n)
        m1 = n - m0

        def bx(x0, x1, y0, y1, z0, z1) -> Box:
            return Box(x0 * h + s, x1 * h + s, y0 * h + s, y1 * h + s, z0 * h + s, z1 * h + s)

        boxes = (
            # x-min wall split into four boxes around the mouth.
            bx(a, c0, a, m0, a, b),
            bx(a, c0, m1, b, a, b),
            bx(a, c0, m0, m1, a, m0),
            bx(a, c0, m0, m1, m1, b),
            # x-max wall.
            bx(c1, b, a, b, a, b),
            # y walls.
            bx(c0, c1, a, c0, a, b),
            bx(c0, c1, c1, b, a, b),
            # z walls.
            bx(c0, c1, c0, c1, a, c0),
            bx(c0, c1, c0, c1, c1, b),
        )
        return Bell3DGeometry(n_geometry=n, epsilon=float(epsilon), boxes=boxes)

    def edges(self, axis: int) -> list[float]:
        vals: set[float] = set()
        for box in self.boxes:
            lo = (box.x0, box.y0, box.z0)[axis]
            hi = (box.x1, box.y1, box.z1)[axis]
            vals.add(lo)
            vals.add(hi)
        return sorted(vals)

    def is_solid(self, x: float, y: float, z: float) -> bool:
        return any(box.contains(x, y, z) for box in self.boxes)

    def is_fluid(self, x: float, y: float, z: float) -> bool:
        inside = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and (0.0 <= z <= 1.0)
        return inside and not self.is_solid(x, y, z)

    @property
    def chamber_center(self) -> tuple[float, float, float]:
        h = 1.0 / self.n_geometry
        s = self.epsilon * h
        return (0.5 + s, 0.5 + s, 0.5 + s)


def cell_subboxes(
    geom: Bell3DGeometry, n_elements: int, ix: int, iy: int, iz: int
) -> Iterable[tuple[float, float, float, float, float, float]]:
    h = 1.0 / n_elements
    bounds = [(ix * h, (ix + 1) * h), (iy * h, (iy + 1) * h), (iz * h, (iz + 1) * h)]
    cuts = []
    for axis in range(3):
        lo, hi = bounds[axis]
        vals = [lo, hi] + [v for v in geom.edges(axis) if lo < v < hi]
        cuts.append(sorted(set(vals)))
    for xa, xb in zip(cuts[0][:-1], cuts[0][1:]):
        for ya, yb in zip(cuts[1][:-1], cuts[1][1:]):
            for za, zb in zip(cuts[2][:-1], cuts[2][1:]):
                if xb <= xa or yb <= ya or zb <= za:
                    continue
                if geom.is_fluid(0.5 * (xa + xb), 0.5 * (ya + yb), 0.5 * (za + zb)):
                    yield (xa, xb, ya, yb, za, zb)


def cell_fluid_fraction(geom: Bell3DGeometry, n_elements: int, ix: int, iy: int, iz: int) -> float:
    vol = sum(
        (xb - xa) * (yb - ya) * (zb - za)
        for xa, xb, ya, yb, za, zb in cell_subboxes(geom, n_elements, ix, iy, iz)
    )
    return float(vol * n_elements**3)


# ---------------------------------------------------------------------------
# 1D interval forms (the tensor factors of every 3D integral).
# ---------------------------------------------------------------------------


class AxisForms:
    """Per-axis 1D interval integrals for primal and flux bases."""

    def __init__(self, basis: SplineBasis1D, flux: SplineBasis1D | None, quadrature_order: int):
        self.basis = basis
        self.flux = flux
        q = quadrature_order
        g, w = np.polynomial.legendre.leggauss(q)
        self._g = g
        self._w = w
        self._cache: dict[tuple[int, float, float], tuple] = {}

    def interval(self, element: int, a: float, b: float):
        """Local 1D matrices on [a, b] inside one element.

        Returns (lo, M, K, D, flo, F, C, xq, wq, vals) with
        M = mass, K = stiffness, D = first-derivative values gram is not
        needed; F = flux-basis mass, C = flux x primal-derivative coupling.
        """
        key = (element, round(a, 14), round(b, 14))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        xq = 0.5 * (b - a) * self._g + 0.5 * (b + a)
        wq = 0.5 * (b - a) * self._w
        lo, hi = element_support(self.basis, element)
        v = basis_value_table(self.basis, xq, nu=0)[lo:hi]
        d = basis_value_table(self.basis, xq, nu=1)[lo:hi]
        M = (v * wq) @ v.T
        K = (d * wq) @ d.T
        flo = 0
        F = None
        C = None
        if self.flux is not None:
            flo, fhi = element_support(self.flux, element)
            fv = basis_value_table(self.flux, xq, nu=0)[flo:fhi]
            F = (fv * wq) @ fv.T
            C = (fv * wq) @ d.T
        out = (lo, M, K, flo, F, C, xq, wq, v)
        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# Sparse triplet helpers.
# ---------------------------------------------------------------------------


class _Triplets:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []

    def add(self, rows: np.ndarray, cols: np.ndarray, block: np.ndarray) -> None:
        self.rows.append(np.repeat(rows, len(cols)))
        self.cols.append(np.tile(cols, len(rows)))
        self.vals.append(np.asarray(block, dtype=float).reshape(-1))

    def tocsr(self) -> sp.csr_matrix:
        if not self.rows:
            return sp.csr_matrix(self.shape)
        return sp.coo_matrix(
            (np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
            shape=self.shape,
        ).tocsr()


def _kron3_block(az: np.ndarray, ay: np.ndarray, ax: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kl,mn->ikmjln", az, ay, ax).reshape(
        az.shape[0] * ay.shape[0] * ax.shape[0], az.shape[1] * ay.shape[1] * ax.shape[1]
    )


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


@dataclass
class Bell3DAssembly:
    geom: Bell3DGeometry
    n_elements: int
    degree: int
    bases: tuple[SplineBasis1D, SplineBasis1D, SplineBasis1D]
    H: sp.csr_matrix
    K: sp.csr_matrix
    H_cellfull: sp.csr_matrix
    K_cellfull: sp.csr_matrix
    gamma: np.ndarray
    active: np.ndarray
    load_full: np.ndarray
    probe_full: np.ndarray
    cut_fractions: np.ndarray
    flux_mass: tuple[sp.csr_matrix, ...] | None
    flux_coupling: tuple[sp.csr_matrix, ...] | None

    @property
    def n_full(self) -> int:
        nx, ny, nz = (b.n_basis for b in self.bases)
        return nx * ny * nz

    @property
    def min_cut_fraction(self) -> float:
        f = self.cut_fractions
        cut = f[(f > 1.0e-15) & (f < 1.0 - 1.0e-12)]
        return float(cut.min()) if cut.size else 1.0


def _gaussian_1d_load(xq: np.ndarray, wq: np.ndarray, vals: np.ndarray, center: float, sigma: float) -> np.ndarray:
    g = np.exp(-((xq - center) ** 2) / (2.0 * sigma**2))
    return (vals * wq) @ g


def assemble_bell_3d(
    geom: Bell3DGeometry,
    *,
    n_elements: int,
    degree: int,
    with_flux: bool = True,
    quadrature_order: int | None = None,
    pulse_center: tuple[float, float, float] = (0.10, 0.50, 0.50),
    pulse_sigma: float = 0.06,
    probe: tuple[float, float, float] | None = None,
) -> Bell3DAssembly:
    q = quadrature_order or max(5, degree + 3)
    bases = tuple(make_open_uniform_basis(n_elements=n_elements, degree=degree) for _ in range(3))
    fluxes = tuple(make_derivative_basis(b) for b in bases) if with_flux else (None, None, None)
    axes = [AxisForms(bases[i], fluxes[i], q) for i in range(3)]
    nx, ny, nz = (b.n_basis for b in bases)
    n = nx * ny * nz
    p1 = degree + 1

    if probe is None:
        probe = geom.chamber_center

    tH = _Triplets((n, n))
    tK = _Triplets((n, n))
    tHf = _Triplets((n, n))
    tKf = _Triplets((n, n))
    load = np.zeros(n)
    probe_load = np.zeros(n)
    cut_fractions = np.zeros((n_elements, n_elements, n_elements))

    if with_flux:
        nfx = fluxes[0].n_basis
        nfy = fluxes[1].n_basis
        nfz = fluxes[2].n_basis
        flux_dims = (nfx * ny * nz, nx * nfy * nz, nx * ny * nfz)
        tF = [_Triplets((flux_dims[i], flux_dims[i])) for i in range(3)]
        tC = [_Triplets((flux_dims[i], n)) for i in range(3)]

    def primal_rows(lox: int, loy: int, loz: int) -> np.ndarray:
        gx = np.arange(lox, lox + p1)
        gy = np.arange(loy, loy + p1)
        gz = np.arange(loz, loz + p1)
        return ((gz[:, None] * ny + gy[None, :])[:, :, None] * nx + gx[None, None, :]).reshape(-1)

    def flux_rows(axis: int, lof: int, lo_others: tuple[int, int]) -> np.ndarray:
        # Flux space along `axis` replaces that axis basis with the
        # derivative basis; index layout mirrors the primal (z, y, x) order.
        if axis == 0:
            gx = np.arange(lof, lof + p1 - 1)
            gy = np.arange(lo_others[0], lo_others[0] + p1)
            gz = np.arange(lo_others[1], lo_others[1] + p1)
            return ((gz[:, None] * ny + gy[None, :])[:, :, None] * nfx + gx[None, None, :]).reshape(-1)
        if axis == 1:
            gx = np.arange(lo_others[0], lo_others[0] + p1)
            gy = np.arange(lof, lof + p1 - 1)
            gz = np.arange(lo_others[1], lo_others[1] + p1)
            return ((gz[:, None] * nfy + gy[None, :])[:, :, None] * nx + gx[None, None, :]).reshape(-1)
        gx = np.arange(lo_others[0], lo_others[0] + p1)
        gy = np.arange(lo_others[1], lo_others[1] + p1)
        gz = np.arange(lof, lof + p1 - 1)
        return ((gz[:, None] * ny + gy[None, :])[:, :, None] * nx + gx[None, None, :]).reshape(-1)

    h = 1.0 / n_elements
    for iz in range(n_elements):
        for iy in range(n_elements):
            for ix in range(n_elements):
                cell = (ix, iy, iz)
                # Full-cell (background) forms, one per cell.
                fx = axes[0].interval(ix, ix * h, (ix + 1) * h)
                fy = axes[1].interval(iy, iy * h, (iy + 1) * h)
                fz = axes[2].interval(iz, iz * h, (iz + 1) * h)
                rows = primal_rows(fx[0], fy[0], fz[0])
                Mfull = _kron3_block(fz[1], fy[1], fx[1])
                Kfull = (
                    _kron3_block(fz[1], fy[1], fx[2])
                    + _kron3_block(fz[1], fy[2], fx[1])
                    + _kron3_block(fz[2], fy[1], fx[1])
                )
                tHf.add(rows, rows, Mfull)
                tKf.add(rows, rows, Kfull)

                vol = 0.0
                for xa, xb, ya, yb, za, zb in cell_subboxes(geom, n_elements, ix, iy, iz):
                    vol += (xb - xa) * (yb - ya) * (zb - za)
                    sx = axes[0].interval(ix, xa, xb)
                    sy = axes[1].interval(iy, ya, yb)
                    sz = axes[2].interval(iz, za, zb)
                    M = _kron3_block(sz[1], sy[1], sx[1])
                    Kb = (
                        _kron3_block(sz[1], sy[1], sx[2])
                        + _kron3_block(sz[1], sy[2], sx[1])
                        + _kron3_block(sz[2], sy[1], sx[1])
                    )
                    tH.add(rows, rows, M)
                    tK.add(rows, rows, Kb)
                    lx = _gaussian_1d_load(sx[6], sx[7], sx[8], pulse_center[0], pulse_sigma)
                    ly = _gaussian_1d_load(sy[6], sy[7], sy[8], pulse_center[1], pulse_sigma)
                    lz = _gaussian_1d_load(sz[6], sz[7], sz[8], pulse_center[2], pulse_sigma)
                    load[rows] += np.einsum("k,j,i->kji", lz, ly, lx).reshape(-1)
                    px = _gaussian_1d_load(sx[6], sx[7], sx[8], probe[0], 1.15 * pulse_sigma)
                    py = _gaussian_1d_load(sy[6], sy[7], sy[8], probe[1], 1.15 * pulse_sigma)
                    pz = _gaussian_1d_load(sz[6], sz[7], sz[8], probe[2], 1.15 * pulse_sigma)
                    probe_load[rows] += np.einsum("k,j,i->kji", pz, py, px).reshape(-1)

                    if with_flux:
                        # axis 0: flux-x mass and coupling.
                        rows_f0 = flux_rows(0, sx[3], (fy[0], fz[0]))
                        tF[0].add(rows_f0, rows_f0, _kron3_block(sz[1], sy[1], sx[4]))
                        tC[0].add(rows_f0, rows, _kron3_block(sz[1], sy[1], sx[5]))
                        rows_f1 = flux_rows(1, sy[3], (fx[0], fz[0]))
                        tF[1].add(rows_f1, rows_f1, _kron3_block(sz[1], sy[4], sx[1]))
                        tC[1].add(rows_f1, rows, _kron3_block(sz[1], sy[5], sx[1]))
                        rows_f2 = flux_rows(2, sz[3], (fx[0], fy[0]))
                        tF[2].add(rows_f2, rows_f2, _kron3_block(sz[4], sy[1], sx[1]))
                        tC[2].add(rows_f2, rows, _kron3_block(sz[5], sy[1], sx[1]))
                cut_fractions[iz, iy, ix] = vol * n_elements**3

    H = tH.tocsr()
    K = tK.tocsr()
    H_cellfull = tHf.tocsr()
    K_cellfull = tKf.tocsr()
    H = (H + H.T) * 0.5
    K = (K + K.T) * 0.5
    H_cellfull = (H_cellfull + H_cellfull.T) * 0.5
    K_cellfull = (K_cellfull + K_cellfull.T) * 0.5

    diag_fluid = np.asarray(H.diagonal())
    diag_bg = np.asarray(H_cellfull.diagonal())
    gamma = np.zeros(n)
    pos = diag_bg > 0.0
    gamma[pos] = diag_fluid[pos] / diag_bg[pos]
    active = np.where(gamma > 0.0)[0]

    flux_mass = None
    flux_coupling = None
    if with_flux:
        flux_mass = tuple(0.5 * (t.tocsr() + t.tocsr().T) for t in tF)
        flux_coupling = tuple(t.tocsr() for t in tC)

    return Bell3DAssembly(
        geom=geom,
        n_elements=n_elements,
        degree=degree,
        bases=bases,
        H=H,
        K=K,
        H_cellfull=H_cellfull,
        K_cellfull=K_cellfull,
        gamma=gamma,
        active=active,
        load_full=load,
        probe_full=probe_load,
        cut_fractions=cut_fractions,
        flux_mass=flux_mass,
        flux_coupling=flux_coupling,
    )


# ---------------------------------------------------------------------------
# 3D polynomial-preserving aggregation (sparse).
# ---------------------------------------------------------------------------


def _greville_3d(bases) -> np.ndarray:
    gx, gy, gz = (b.greville for b in bases)
    nx, ny, nz = (b.n_basis for b in bases)
    X = np.tile(np.tile(gx, ny), nz)
    Y = np.tile(np.repeat(gy, nx), nz)
    Z = np.repeat(gz, nx * ny)
    return np.column_stack([X, Y, Z])


def _monomial_rows_3d(bases, degree: int) -> np.ndarray:
    """Spline coefficient vectors of all total-degree <= degree monomials."""
    cx = polynomial_coefficients(bases[0], degree)
    cy = polynomial_coefficients(bases[1], degree)
    cz = polynomial_coefficients(bases[2], degree)
    rows = []
    for total in range(degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                c = total - a - b
                rows.append(np.einsum("k,j,i->kji", cz[c], cy[b], cx[a]).reshape(-1))
    return np.vstack(rows)


@dataclass
class Aggregation3D:
    E: sp.csr_matrix
    interior: np.ndarray
    small_support: np.ndarray
    active: np.ndarray
    gamma: np.ndarray
    polynomial_residual: float


def polynomial_preserving_extension_3d(
    asm: Bell3DAssembly,
    *,
    support_threshold: float = 0.04,
    preserve_degree: int | None = None,
) -> Aggregation3D:
    bases = asm.bases
    degree = asm.degree if preserve_degree is None else int(preserve_degree)
    gamma = asm.gamma
    active = asm.active
    interior = np.where(gamma >= support_threshold)[0]
    small = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    if interior.size == 0:
        raise ValueError("no interior coordinates")

    poly = _monomial_rows_3d(bases, degree)
    n_terms = poly.shape[0]
    greville = _greville_3d(bases)
    col = {int(g): j for j, g in enumerate(interior)}
    n_full = asm.n_full

    rows_t: list[int] = []
    cols_t: list[int] = []
    vals_t: list[float] = []
    for g in interior:
        rows_t.append(int(g))
        cols_t.append(col[int(g)])
        vals_t.append(1.0)

    residual = 0.0
    if degree == 0:
        # Constants-only aggregation: mass-weighted average of the nearest
        # interior neighbors, weights summing to one.
        m_diag = np.asarray(asm.H.diagonal())
        for b in small:
            dist = np.linalg.norm(greville[interior] - greville[int(b)], axis=1)
            nb = interior[np.argsort(dist)[:8]]
            w = np.maximum(m_diag[nb], 0.0)
            w = w / max(float(w.sum()), 1.0e-300)
            for g, weight in zip(nb, w):
                rows_t.append(int(b))
                cols_t.append(col[int(g)])
                vals_t.append(float(weight))
        E = sp.csr_matrix((vals_t, (rows_t, cols_t)), shape=(n_full, interior.size))
        return Aggregation3D(E=E, interior=interior, small_support=small, active=active, gamma=gamma, polynomial_residual=0.0)

    order_cache = np.argsort(
        np.linalg.norm(greville[interior][None, :, :] - greville[small][:, None, :], axis=2), axis=1
    ) if small.size else np.empty((0, 0), dtype=int)
    for row_idx, b in enumerate(small):
        ordered = interior[order_cache[row_idx]]
        k = n_terms
        chosen = ordered[: k + 8]
        C = poly[:, chosen]
        while np.linalg.matrix_rank(C, tol=1.0e-10) < n_terms and len(chosen) < min(len(ordered), 6 * n_terms):
            k += 8
            chosen = ordered[: k + 8]
            C = poly[:, chosen]
        target = poly[:, int(b)]
        weights, *_ = np.linalg.lstsq(C, target, rcond=None)
        for g, weight in zip(chosen, weights):
            if abs(weight) > 1.0e-14:
                rows_t.append(int(b))
                cols_t.append(col[int(g)])
                vals_t.append(float(weight))

    E = sp.csr_matrix((vals_t, (rows_t, cols_t)), shape=(n_full, interior.size))
    active_mask = (gamma > 0.0).astype(float)
    res = 0.0
    for row in poly:
        res = max(res, float(np.max(np.abs(E @ row[interior] - row * active_mask))))
    return Aggregation3D(E=E, interior=interior, small_support=small, active=active, gamma=gamma, polynomial_residual=res)


# ---------------------------------------------------------------------------
# Operators and integrators (sparse / matrix-free).
# ---------------------------------------------------------------------------


def restrict_spd(A: sp.csr_matrix, E: sp.csr_matrix) -> sp.csr_matrix:
    B = (E.T @ A @ E).tocsr()
    return (B + B.T) * 0.5


class BMFStiffness:
    """Matrix-free K = sum_i C_i^T H_{1,i}^{-1} C_i on a reduced space.

    Each directional flux mass is restricted to its mass-visible rows and
    factorized once; applications cost three sparse matvec pairs and three
    sparse triangular solves.
    """

    def __init__(
        self,
        E: sp.csr_matrix,
        flux_mass: Sequence[sp.csr_matrix],
        flux_coupling: Sequence[sp.csr_matrix],
        *,
        visibility: float = 1.0e-10,
    ):
        self.E = E.tocsc()
        self.parts = []
        for Hf, C in zip(flux_mass, flux_coupling):
            d = np.asarray(Hf.diagonal())
            scale = float(np.max(d)) if d.size else 1.0
            keep = np.where(d > visibility * scale)[0]
            Hk = Hf[keep][:, keep].tocsc()
            Ck = (C[keep] @ self.E).tocsr()
            lu = spla.splu(Hk)
            self.parts.append((lu, Ck))
        self.shape = (self.E.shape[1], self.E.shape[1])
        self.dtype = np.float64

    def matvec(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape[0])
        for lu, Ck in self.parts:
            c = Ck @ v
            out += Ck.T @ lu.solve(c)
        return out

    def __matmul__(self, v):
        v = np.asarray(v)
        if v.ndim == 1:
            return self.matvec(v)
        return np.column_stack([self.matvec(v[:, j]) for j in range(v.shape[1])])

    def aslinearoperator(self) -> spla.LinearOperator:
        return spla.LinearOperator(self.shape, matvec=self.matvec, dtype=float)


def omega_max(K, M: sp.csr_matrix, *, iters: int = 120, seed: int = 5) -> float:
    """Largest generalized frequency sqrt(lambda_max(M^{-1}K)) by power iteration."""
    lu = spla.splu(M.tocsc())
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = K @ v if not isinstance(K, BMFStiffness) else K.matvec(v)
        w = lu.solve(w)
        nrm = np.linalg.norm(w)
        if nrm <= 0.0:
            return 0.0
        v = w / nrm
        lam = nrm
    return math.sqrt(max(lam, 0.0))


def nearest_mode_shift_sparse(K: sp.csr_matrix, M: sp.csr_matrix, omega_ref: float, *, k: int = 6) -> float:
    """Relative distance from omega_ref to the nearest exact eigenfrequency.

    Shift-inverted Lanczos around the reference resonance; explicit sparse
    operators only (the matrix-free BMF stiffness is certified through the
    measured fingerprint instead).
    """
    sigma = omega_ref**2
    vals = spla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=sigma, which="LM", return_eigenvectors=False)
    omegas = np.sqrt(np.maximum(vals, 0.0))
    return float(np.min(np.abs(omegas - omega_ref)) / max(omega_ref, 1.0e-300))


def verlet_ledger_sparse(
    M: sp.csr_matrix,
    K,
    load: np.ndarray,
    probe_load: np.ndarray,
    *,
    dt: float,
    t_final: float,
    n_samples: int,
    post_step: Callable[[np.ndarray], np.ndarray] | None = None,
    phys: tuple[sp.csr_matrix, object, np.ndarray] | None = None,
    fail_at: float = 1.0e6,
) -> dict:
    """Velocity-Verlet ledgers with sparse mass solves (mirrors the 2D bell)."""
    lu = spla.splu(M.tocsc())

    def kmul(v: np.ndarray) -> np.ndarray:
        return K.matvec(v) if isinstance(K, BMFStiffness) else K @ v

    p = lu.solve(load)
    ones = np.ones_like(p)
    mean = float(ones @ (M @ p)) / float(ones @ (M @ ones))
    p = p - mean * ones
    v = np.zeros_like(p)
    m_vec = M @ ones
    if phys is not None:
        M_phys, K_phys, m_phys = phys
    else:
        M_phys, K_phys, m_phys = M, K, m_vec

    def kmul_phys(w: np.ndarray) -> np.ndarray:
        return K_phys.matvec(w) if isinstance(K_phys, BMFStiffness) else K_phys @ w

    n_steps = max(1, int(math.ceil(t_final / dt)))
    sample_every = max(1, n_steps // max(1, n_samples - 1))

    times: list[float] = []
    stag_energy: list[float] = []
    phys_energy: list[float] = []
    mass: list[float] = []
    signal: list[float] = []

    mass0 = float(m_phys @ p)
    mass_scale = max(float(np.sum(np.abs(m_phys * p))), 1.0e-30)
    Kp = kmul(p)
    accel = lu.solve(Kp)
    failed = False
    E_star0: float | None = None
    same_metric = phys is None
    E_phys0 = 0.5 * float(v @ (M_phys @ v) + (p @ Kp if same_metric else p @ kmul_phys(p)))
    for nstep in range(n_steps):
        v_half = v - 0.5 * dt * accel
        p_old = p
        p = p + dt * v_half
        Kp = kmul(p)
        E_star = 0.5 * float(v_half @ (M @ v_half)) + 0.5 * float(p_old @ Kp)
        accel = lu.solve(Kp)
        v = v_half - 0.5 * dt * accel
        if post_step is not None:
            p = post_step(p)
            v = post_step(v)
            Kp = kmul(p)
            accel = lu.solve(Kp)
        if E_star0 is None:
            E_star0 = E_star
        if nstep % sample_every == 0 or nstep == n_steps - 1:
            times.append((nstep + 1) * dt)
            stag_energy.append(E_star)
            phys_energy.append(0.5 * float(v @ (M_phys @ v) + (p @ Kp if same_metric else p @ kmul_phys(p))))
            mass.append(float(m_phys @ p))
            signal.append(float(probe_load @ p))
            if not np.isfinite(E_star) or abs(E_star) > fail_at * max(abs(E_star0), 1.0e-30):
                failed = True
                break

    stag = np.asarray(stag_energy)
    e_scale = max(abs(E_star0 or 0.0), 1.0e-30)
    finite = stag[np.isfinite(stag)]
    energy_defect = float(np.max(np.abs(finite - (E_star0 or 0.0))) / e_scale) if finite.size else float("inf")
    mass_arr = np.asarray(mass)
    finite_m = mass_arr[np.isfinite(mass_arr)]
    mass_defect = float(np.max(np.abs(finite_m - mass0)) / mass_scale) if finite_m.size else float("inf")
    phys_arr = np.asarray(phys_energy)
    finite_p = phys_arr[np.isfinite(phys_arr)]
    phys_defect = (
        float(np.max(np.abs(finite_p - E_phys0)) / max(abs(E_phys0), 1.0e-30)) if finite_p.size else float("inf")
    )
    return {
        "times": np.asarray(times),
        "staggered_energy": stag,
        "phys_energy": phys_arr,
        "mass": mass_arr,
        "signal": np.asarray(signal),
        "energy0": float(E_star0 or 0.0),
        "phys_energy0": E_phys0,
        "mass0": mass0,
        "mass_scale": mass_scale,
        "failed": failed or energy_defect == float("inf"),
        "energy_defect": energy_defect,
        "mass_defect": mass_defect,
        "phys_energy_defect": phys_defect,
        "dt": dt,
        "state": p,
    }
