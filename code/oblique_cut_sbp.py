from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np
from scipy.linalg import eigvalsh, svdvals

from cut_spline_sbp import (
    SplineBasis1D,
    assemble_cut_matrices,
    make_open_uniform_basis,
    polynomial_coefficients,
)


@dataclass(frozen=True)
class HalfPlaneCut:
    """A square cut by the half-plane normal . x >= offset."""

    normal: np.ndarray
    offset: float

    @staticmethod
    def from_angle(angle: float, offset: float) -> "HalfPlaneCut":
        n = np.array([cos(angle), sin(angle)], dtype=float)
        n = n / np.linalg.norm(n)
        return HalfPlaneCut(normal=n, offset=float(offset))


@dataclass(frozen=True)
class VolumeQuadrature2D:
    """Positive volume quadrature representation for a cut mass matrix.

    When ``basis_grad_x`` and ``basis_grad_y`` are present (one column per
    quadrature node), the stiffness matrix is also recoverable as the
    sum of weighted directional Gram matrices.  This is the same positive
    representation used by the mass aggregate-frame theorem, extended to the
    stiffness aggregate-frame analog.
    """

    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    cell_ids: np.ndarray
    basis_values: np.ndarray
    nx: int
    ny: int
    basis_grad_x: np.ndarray | None = None
    basis_grad_y: np.ndarray | None = None

    def mass_matrix(self) -> np.ndarray:
        return (self.basis_values * self.weights) @ self.basis_values.T

    def stiffness_matrix(self) -> np.ndarray:
        if self.basis_grad_x is None or self.basis_grad_y is None:
            raise ValueError(
                "stiffness_matrix requires gradient values captured at quadrature nodes"
            )
        gx = self.basis_grad_x
        gy = self.basis_grad_y
        return (gx * self.weights) @ gx.T + (gy * self.weights) @ gy.T


@dataclass(frozen=True)
class ObliqueCutMatrices2D:
    """Direct non-tensor 2D cut-domain matrices.

    Coefficients are flattened with x as the fast index:
    index = ix + nx * iy.
    """

    H: np.ndarray
    Qx: np.ndarray
    Qy: np.ndarray
    K: np.ndarray
    H_background: np.ndarray
    Sx: np.ndarray
    Sy: np.ndarray
    nx: int
    ny: int
    domain_polygon: np.ndarray
    cut: HalfPlaneCut
    volume_quadrature: VolumeQuadrature2D | None = None


@dataclass(frozen=True)
class ObliqueAggregation2D:
    E: np.ndarray
    interior: np.ndarray
    small_support: np.ndarray
    active: np.ndarray
    gamma: np.ndarray
    polynomial_degree: int
    polynomial_residual: float


@dataclass(frozen=True)
class ObliqueActiveSets2D:
    active: np.ndarray
    interior: np.ndarray
    small_support: np.ndarray
    gamma: np.ndarray


def _inside(point: np.ndarray, cut: HalfPlaneCut, tol: float = 1.0e-14) -> bool:
    return bool(np.dot(cut.normal, point) >= cut.offset - tol)


def _segment_intersection(a: np.ndarray, b: np.ndarray, cut: HalfPlaneCut) -> np.ndarray:
    fa = float(np.dot(cut.normal, a) - cut.offset)
    fb = float(np.dot(cut.normal, b) - cut.offset)
    denom = fa - fb
    if abs(denom) < 1.0e-30:
        return 0.5 * (a + b)
    t = fa / denom
    t = min(1.0, max(0.0, t))
    return a + t * (b - a)


def clip_polygon_halfplane(poly: np.ndarray, cut: HalfPlaneCut) -> np.ndarray:
    """Clip a CCW polygon against normal . x >= offset."""
    if len(poly) == 0:
        return np.empty((0, 2), dtype=float)
    out: list[np.ndarray] = []
    prev = np.asarray(poly[-1], dtype=float)
    prev_inside = _inside(prev, cut)
    for cur in np.asarray(poly, dtype=float):
        cur_inside = _inside(cur, cut)
        if cur_inside:
            if not prev_inside:
                out.append(_segment_intersection(prev, cur, cut))
            out.append(cur.copy())
        elif prev_inside:
            out.append(_segment_intersection(prev, cur, cut))
        prev = cur
        prev_inside = cur_inside
    if not out:
        return np.empty((0, 2), dtype=float)
    # Remove adjacent duplicates introduced by clipping through corners.
    cleaned: list[np.ndarray] = []
    for point in out:
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > 1.0e-13:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= 1.0e-13:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _square_polygon() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)


def _cell_polygon(ix: int, iy: int, n_elements: int) -> np.ndarray:
    h = 1.0 / n_elements
    x0 = ix * h
    x1 = (ix + 1) * h
    y0 = iy * h
    y1 = (iy + 1) * h
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _eval_all(functions, x: np.ndarray) -> np.ndarray:
    return np.vstack([np.nan_to_num(f(x), nan=0.0) for f in functions])


def _tensor_outer(vy: np.ndarray, vx: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Row-wise tensor product (row = iy*nx + ix), evaluated in point-chunks.

    Bit-identical to ``(vy[:, None, :] * vx[None, :, :]).reshape(ny*nx, nq)`` but
    never materializes the full ``(ny, nx, nq)`` transient, whose size explodes on
    the finest cut meshes (a multi-GB intermediate at ``n=32``).  Chunking over the
    quadrature points caps that transient without changing any output value.
    """
    ny, nq = vy.shape
    nx = vx.shape[0]
    out = np.empty((ny * nx, nq), dtype=float)
    for start in range(0, nq, chunk):
        stop = min(start + chunk, nq)
        block = vy[:, None, start:stop] * vx[None, :, start:stop]
        out[:, start:stop] = block.reshape(ny * nx, stop - start)
    return out


def _tensor_values(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vx = _eval_all(basis_x.basis, x)
    vy = _eval_all(basis_y.basis, y)
    dvx = _eval_all(basis_x.dbasis, x)
    dvy = _eval_all(basis_y.dbasis, y)
    # y-major rows, x-fast columns: row = iy * nx + ix.
    phi = _tensor_outer(vy, vx)
    dphix = _tensor_outer(vy, dvx)
    dphiy = _tensor_outer(dvy, vx)
    return phi, dphix, dphiy


def _triangle_quadrature(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Product Gauss-Duffy quadrature on a physical triangle."""
    gu, wu = np.polynomial.legendre.leggauss(order)
    gv, wv = np.polynomial.legendre.leggauss(order)
    u = 0.5 * (gu + 1.0)
    v = 0.5 * (gv + 1.0)
    wu = 0.5 * wu
    wv = 0.5 * wv
    e1 = v1 - v0
    e2 = v2 - v0
    det = abs(float(e1[0] * e2[1] - e1[1] * e2[0]))
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    for ui, wui in zip(u, wu):
        one_minus_u = 1.0 - ui
        for vi, wvi in zip(v, wv):
            point = v0 + ui * e1 + one_minus_u * vi * e2
            xs.append(float(point[0]))
            ys.append(float(point[1]))
            ws.append(float(det * one_minus_u * wui * wvi))
    return np.asarray(xs), np.asarray(ys), np.asarray(ws)


def _polygon_quadrature(poly: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(poly) < 3:
        return np.empty(0), np.empty(0), np.empty(0)
    if polygon_area(poly) < 0.0:
        poly = poly[::-1]
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    ws_all: list[np.ndarray] = []
    v0 = poly[0]
    for k in range(1, len(poly) - 1):
        xs, ys, ws = _triangle_quadrature(v0, poly[k], poly[k + 1], order)
        xs_all.append(xs)
        ys_all.append(ys)
        ws_all.append(ws)
    return np.concatenate(xs_all), np.concatenate(ys_all), np.concatenate(ws_all)


def _edge_quadrature(a: np.ndarray, b: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g, w = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (g + 1.0)
    length = float(np.linalg.norm(b - a))
    points = a[:, None] + (b - a)[:, None] * t[None, :]
    weights = 0.5 * length * w
    if length <= 1.0e-15:
        normal = np.array([0.0, 0.0])
    else:
        dx, dy = (b - a) / length
        normal = np.array([dy, -dx])
    return points[0], points[1], weights, normal


def tensor_background_mass(basis_x: SplineBasis1D, basis_y: SplineBasis1D) -> np.ndarray:
    mx = assemble_cut_matrices(basis_x, left=0.0)
    my = assemble_cut_matrices(basis_y, left=0.0)
    return np.kron(my.H, mx.H)


def assemble_oblique_cut_matrices(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: HalfPlaneCut,
    quadrature_order: int | None = None,
    *,
    capture_volume_quadrature: bool = False,
) -> ObliqueCutMatrices2D:
    """Assemble H, Qx, Qy, K, Sx, Sy by clipped-cell quadrature.

    Identical quadrature to the closure-based reference assembly, evaluated
    through vectorized local-block basis tables: only the (p+1)^2 basis
    functions supported on a cell enter that cell's Gram updates, which
    removes the O(n^2 m) per-cell cost of the reference path.
    """
    from cut_assembly import basis_value_table, element_support

    if basis_x.n_elements != basis_y.n_elements:
        raise ValueError("The prototype expects the same element count in x and y.")
    if basis_x.degree != basis_y.degree:
        raise ValueError("The prototype expects the same spline degree in x and y.")
    p = basis_x.degree
    q = quadrature_order or max(10, p + 5)
    nx = basis_x.n_basis
    ny = basis_y.n_basis
    n = nx * ny
    H = np.zeros((n, n))
    Qx = np.zeros((n, n))
    Qy = np.zeros((n, n))
    K = np.zeros((n, n))
    Sx = np.zeros((n, n))
    Sy = np.zeros((n, n))
    volume_x: list[np.ndarray] = []
    volume_y: list[np.ndarray] = []
    volume_w: list[np.ndarray] = []
    volume_cells: list[np.ndarray] = []
    volume_rows: list[np.ndarray] = []
    volume_phi: list[np.ndarray] = []
    volume_dphix: list[np.ndarray] = []
    volume_dphiy: list[np.ndarray] = []

    def is_physical_boundary_edge(a: np.ndarray, b: np.ndarray) -> bool:
        mid = 0.5 * (a + b)
        on_cut = abs(float(np.dot(cut.normal, mid) - cut.offset)) <= 1.0e-10
        on_square = (
            abs(mid[0]) <= 1.0e-12
            or abs(mid[0] - 1.0) <= 1.0e-12
            or abs(mid[1]) <= 1.0e-12
            or abs(mid[1] - 1.0) <= 1.0e-12
        )
        return bool(on_cut or on_square)

    def local_tensor(ix: int, iy: int, xq: np.ndarray, yq: np.ndarray):
        lox, hix = element_support(basis_x, ix)
        loy, hiy = element_support(basis_y, iy)
        vx = basis_value_table(basis_x, xq, nu=0)[lox:hix]
        dvx = basis_value_table(basis_x, xq, nu=1)[lox:hix]
        vy = basis_value_table(basis_y, yq, nu=0)[loy:hiy]
        dvy = basis_value_table(basis_y, yq, nu=1)[loy:hiy]
        rows = (np.arange(loy, hiy)[:, None] * nx + np.arange(lox, hix)[None, :]).reshape(-1)
        phi = (vy[:, None, :] * vx[None, :, :]).reshape(rows.size, xq.size)
        dphix = (vy[:, None, :] * dvx[None, :, :]).reshape(rows.size, xq.size)
        dphiy = (dvy[:, None, :] * vx[None, :, :]).reshape(rows.size, xq.size)
        return rows, phi, dphix, dphiy

    for iy in range(basis_y.n_elements):
        for ix in range(basis_x.n_elements):
            cell = _cell_polygon(ix, iy, basis_x.n_elements)
            clipped = clip_polygon_halfplane(cell, cut)
            if len(clipped) < 3 or abs(polygon_area(clipped)) <= 1.0e-16:
                continue
            if polygon_area(clipped) < 0.0:
                clipped = clipped[::-1]
            xq, yq, wq = _polygon_quadrature(clipped, q)
            rows, phi, dphix, dphiy = local_tensor(ix, iy, xq, yq)
            idx = np.ix_(rows, rows)
            H[idx] += (phi * wq) @ phi.T
            Qx[idx] += (phi * wq) @ dphix.T
            Qy[idx] += (phi * wq) @ dphiy.T
            K[idx] += (dphix * wq) @ dphix.T + (dphiy * wq) @ dphiy.T
            if capture_volume_quadrature:
                volume_x.append(xq)
                volume_y.append(yq)
                volume_w.append(wq)
                volume_cells.append(np.full(xq.size, ix + basis_x.n_elements * iy, dtype=int))
                volume_rows.append(rows)
                volume_phi.append(phi)
                volume_dphix.append(dphix)
                volume_dphiy.append(dphiy)

            # Boundary quadrature must be split at every knot/grid crossing.
            for k in range(len(clipped)):
                a = clipped[k]
                b = clipped[(k + 1) % len(clipped)]
                if not is_physical_boundary_edge(a, b):
                    continue
                xqe, yqe, wqe, normal = _edge_quadrature(a, b, q)
                if np.allclose(normal, 0.0):
                    continue
                rows_e, phie, _dpe_x, _dpe_y = local_tensor(ix, iy, xqe, yqe)
                idx_e = np.ix_(rows_e, rows_e)
                boundary_mass = (phie * wqe) @ phie.T
                Sx[idx_e] += normal[0] * boundary_mass
                Sy[idx_e] += normal[1] * boundary_mass

    domain_polygon = clip_polygon_halfplane(_square_polygon(), cut)
    if len(domain_polygon) < 3 or abs(polygon_area(domain_polygon)) <= 1.0e-16:
        raise ValueError("The half-plane cut produced an empty or degenerate domain.")
    if polygon_area(domain_polygon) < 0.0:
        domain_polygon = domain_polygon[::-1]

    H_background = tensor_background_mass(basis_x, basis_y)
    volume_quadrature = None
    if capture_volume_quadrature:
        if volume_x:
            n_pts = int(sum(x.size for x in volume_x))
            basis_values = np.zeros((n, n_pts))
            basis_grad_x = np.zeros((n, n_pts))
            basis_grad_y = np.zeros((n, n_pts))
            offset = 0
            for rows, phi, dpx, dpy in zip(volume_rows, volume_phi, volume_dphix, volume_dphiy):
                m = phi.shape[1]
                basis_values[rows, offset : offset + m] = phi
                basis_grad_x[rows, offset : offset + m] = dpx
                basis_grad_y[rows, offset : offset + m] = dpy
                offset += m
            volume_quadrature = VolumeQuadrature2D(
                x=np.concatenate(volume_x),
                y=np.concatenate(volume_y),
                weights=np.concatenate(volume_w),
                cell_ids=np.concatenate(volume_cells),
                basis_values=basis_values,
                nx=nx,
                ny=ny,
                basis_grad_x=basis_grad_x,
                basis_grad_y=basis_grad_y,
            )
        else:
            volume_quadrature = VolumeQuadrature2D(
                x=np.empty(0),
                y=np.empty(0),
                weights=np.empty(0),
                cell_ids=np.empty(0, dtype=int),
                basis_values=np.empty((n, 0)),
                nx=nx,
                ny=ny,
                basis_grad_x=np.empty((n, 0)),
                basis_grad_y=np.empty((n, 0)),
            )
    return ObliqueCutMatrices2D(
        H=H,
        Qx=Qx,
        Qy=Qy,
        K=K,
        H_background=H_background,
        Sx=Sx,
        Sy=Sy,
        nx=nx,
        ny=ny,
        domain_polygon=domain_polygon,
        cut=cut,
        volume_quadrature=volume_quadrature,
    )


def assemble_oblique_cut_matrices_reference(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: HalfPlaneCut,
    quadrature_order: int | None = None,
    *,
    capture_volume_quadrature: bool = False,
) -> ObliqueCutMatrices2D:
    """Closure-based reference assembly (kept for equivalence tests).

    Volume integration clips each background element against the half-plane and
    triangulates the clipped polygon. Boundary forms are integrated over the
    boundary of the clipped square, including both outer square faces and the
    oblique embedded line segment.  The production path is
    ``assemble_oblique_cut_matrices``, which performs the identical quadrature
    with vectorized local-block evaluation.
    """
    if basis_x.n_elements != basis_y.n_elements:
        raise ValueError("The prototype expects the same element count in x and y.")
    if basis_x.degree != basis_y.degree:
        raise ValueError("The prototype expects the same spline degree in x and y.")
    p = basis_x.degree
    q = quadrature_order or max(10, p + 5)
    nx = basis_x.n_basis
    ny = basis_y.n_basis
    n = nx * ny
    H = np.zeros((n, n))
    Qx = np.zeros((n, n))
    Qy = np.zeros((n, n))
    K = np.zeros((n, n))
    Sx = np.zeros((n, n))
    Sy = np.zeros((n, n))
    volume_x: list[np.ndarray] = []
    volume_y: list[np.ndarray] = []
    volume_w: list[np.ndarray] = []
    volume_cells: list[np.ndarray] = []
    volume_phi: list[np.ndarray] = []
    volume_dphix: list[np.ndarray] = []
    volume_dphiy: list[np.ndarray] = []

    def is_physical_boundary_edge(a: np.ndarray, b: np.ndarray) -> bool:
        mid = 0.5 * (a + b)
        on_cut = abs(float(np.dot(cut.normal, mid) - cut.offset)) <= 1.0e-10
        on_square = (
            abs(mid[0]) <= 1.0e-12
            or abs(mid[0] - 1.0) <= 1.0e-12
            or abs(mid[1]) <= 1.0e-12
            or abs(mid[1] - 1.0) <= 1.0e-12
        )
        return bool(on_cut or on_square)

    for iy in range(basis_y.n_elements):
        for ix in range(basis_x.n_elements):
            cell = _cell_polygon(ix, iy, basis_x.n_elements)
            clipped = clip_polygon_halfplane(cell, cut)
            if len(clipped) < 3 or abs(polygon_area(clipped)) <= 1.0e-16:
                continue
            if polygon_area(clipped) < 0.0:
                clipped = clipped[::-1]
            xq, yq, wq = _polygon_quadrature(clipped, q)
            phi, dphix, dphiy = _tensor_values(basis_x, basis_y, xq, yq)
            H += (phi * wq) @ phi.T
            Qx += (phi * wq) @ dphix.T
            Qy += (phi * wq) @ dphiy.T
            K += (dphix * wq) @ dphix.T + (dphiy * wq) @ dphiy.T
            if capture_volume_quadrature:
                volume_x.append(xq)
                volume_y.append(yq)
                volume_w.append(wq)
                volume_cells.append(np.full(xq.size, ix + basis_x.n_elements * iy, dtype=int))
                volume_phi.append(phi)
                volume_dphix.append(dphix)
                volume_dphiy.append(dphiy)

            # Boundary quadrature must be split at every knot/grid crossing.
            # Integrating a whole outer side or oblique segment at once would
            # cross B-spline breakpoints and would only converge algebraically.
            for k in range(len(clipped)):
                a = clipped[k]
                b = clipped[(k + 1) % len(clipped)]
                if not is_physical_boundary_edge(a, b):
                    continue
                xqe, yqe, wqe, normal = _edge_quadrature(a, b, q)
                if np.allclose(normal, 0.0):
                    continue
                phie, _dphixe, _dphiye = _tensor_values(basis_x, basis_y, xqe, yqe)
                boundary_mass = (phie * wqe) @ phie.T
                Sx += normal[0] * boundary_mass
                Sy += normal[1] * boundary_mass

    domain_polygon = clip_polygon_halfplane(_square_polygon(), cut)
    if len(domain_polygon) < 3 or abs(polygon_area(domain_polygon)) <= 1.0e-16:
        raise ValueError("The half-plane cut produced an empty or degenerate domain.")
    if polygon_area(domain_polygon) < 0.0:
        domain_polygon = domain_polygon[::-1]

    H_background = tensor_background_mass(basis_x, basis_y)
    volume_quadrature = None
    if capture_volume_quadrature:
        if volume_x:
            volume_quadrature = VolumeQuadrature2D(
                x=np.concatenate(volume_x),
                y=np.concatenate(volume_y),
                weights=np.concatenate(volume_w),
                cell_ids=np.concatenate(volume_cells),
                basis_values=np.concatenate(volume_phi, axis=1),
                nx=nx,
                ny=ny,
                basis_grad_x=np.concatenate(volume_dphix, axis=1),
                basis_grad_y=np.concatenate(volume_dphiy, axis=1),
            )
        else:
            volume_quadrature = VolumeQuadrature2D(
                x=np.empty(0),
                y=np.empty(0),
                weights=np.empty(0),
                cell_ids=np.empty(0, dtype=int),
                basis_values=np.empty((n, 0)),
                nx=nx,
                ny=ny,
                basis_grad_x=np.empty((n, 0)),
                basis_grad_y=np.empty((n, 0)),
            )
    return ObliqueCutMatrices2D(
        H=H,
        Qx=Qx,
        Qy=Qy,
        K=K,
        H_background=H_background,
        Sx=Sx,
        Sy=Sy,
        nx=nx,
        ny=ny,
        domain_polygon=domain_polygon,
        cut=cut,
        volume_quadrature=volume_quadrature,
    )


def oblique_active_sets(
    matrices: ObliqueCutMatrices2D,
    support_threshold: float = 0.05,
    active_tol: float = 1.0e-22,
) -> ObliqueActiveSets2D:
    background = np.diag(matrices.H_background)
    physical = np.diag(matrices.H)
    gamma = np.zeros_like(physical)
    positive = background > 0.0
    gamma[positive] = physical[positive] / background[positive]
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small_support = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    return ObliqueActiveSets2D(active=active, interior=interior, small_support=small_support, gamma=gamma)


def _tensor_greville(basis_x: SplineBasis1D, basis_y: SplineBasis1D) -> np.ndarray:
    xx, yy = np.meshgrid(basis_x.greville, basis_y.greville, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def tensor_polynomial_coefficients(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    max_total_degree: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Coefficient rows for monomials x^r y^s with r+s <= max_total_degree."""
    cx = polynomial_coefficients(basis_x, max_total_degree)
    cy = polynomial_coefficients(basis_y, max_total_degree)
    rows: list[np.ndarray] = []
    powers: list[tuple[int, int]] = []
    for total in range(max_total_degree + 1):
        for sx in range(total + 1):
            sy = total - sx
            rows.append(np.kron(cy[sy], cx[sx]))
            powers.append((sx, sy))
    return np.asarray(rows), powers


def polynomial_preserving_oblique_extension(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    matrices: ObliqueCutMatrices2D,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    active_tol: float = 1.0e-22,
) -> ObliqueAggregation2D:
    sets = oblique_active_sets(matrices, support_threshold=support_threshold, active_tol=active_tol)
    if len(sets.interior) == 0:
        raise ValueError("No interior active tensor-product basis functions were found.")
    requested_degree = basis_x.degree if preserve_degree is None else preserve_degree
    # Total-degree polynomial preservation needs (m+1)(m+2)/2 interior modes.
    max_degree_by_count = 0
    while (max_degree_by_count + 1) * (max_degree_by_count + 2) // 2 <= len(sets.interior):
        max_degree_by_count += 1
    max_degree_by_count -= 1
    degree = min(requested_degree, max_degree_by_count)
    poly, _powers = tensor_polynomial_coefficients(basis_x, basis_y, degree)
    n_terms = poly.shape[0]
    n_full = matrices.H.shape[0]
    n_red = len(sets.interior)
    col = {int(g): j for j, g in enumerate(sets.interior)}
    E = np.zeros((n_full, n_red))
    for g in sets.interior:
        E[int(g), col[int(g)]] = 1.0

    greville = _tensor_greville(basis_x, basis_y)
    for b in sets.small_support:
        distances = np.linalg.norm(greville[sets.interior] - greville[int(b)], axis=1)
        ordered = [int(g) for _, g in sorted(zip(distances, sets.interior), key=lambda item: item[0])]
        chosen = ordered[:n_terms]
        C = poly[:, chosen]
        for k in range(n_terms, len(ordered) + 1):
            candidate = ordered[:k]
            C_candidate = poly[:, candidate]
            if np.linalg.matrix_rank(C_candidate, tol=1.0e-12) == n_terms:
                chosen = candidate
                C = C_candidate
                break
        target = poly[:, int(b)]
        weights = np.linalg.lstsq(C, target, rcond=None)[0]
        for g, weight in zip(chosen, weights):
            E[int(b), col[int(g)]] = weight

    residuals = []
    active_mask = sets.gamma > active_tol
    for row in poly:
        residuals.append(np.linalg.norm(E @ row[sets.interior] - row * active_mask, ord=np.inf))
    residual = float(max(residuals) if residuals else 0.0)
    return ObliqueAggregation2D(
        E=E,
        interior=sets.interior.astype(int),
        small_support=sets.small_support.astype(int),
        active=sets.active.astype(int),
        gamma=sets.gamma,
        polynomial_degree=degree,
        polynomial_residual=residual,
    )


def restrict_oblique_2d(m: ObliqueCutMatrices2D, active: np.ndarray) -> ObliqueCutMatrices2D:
    idx = np.ix_(active, active)
    return ObliqueCutMatrices2D(
        H=m.H[idx],
        Qx=m.Qx[idx],
        Qy=m.Qy[idx],
        K=m.K[idx],
        H_background=m.H_background[idx],
        Sx=m.Sx[idx],
        Sy=m.Sy[idx],
        nx=m.nx,
        ny=m.ny,
        domain_polygon=m.domain_polygon,
        cut=m.cut,
        volume_quadrature=None,
    )


def pullback_oblique_2d(m: ObliqueCutMatrices2D, E: np.ndarray) -> ObliqueCutMatrices2D:
    return ObliqueCutMatrices2D(
        H=E.T @ m.H @ E,
        Qx=E.T @ m.Qx @ E,
        Qy=E.T @ m.Qy @ E,
        K=E.T @ m.K @ E,
        H_background=E.T @ m.H_background @ E,
        Sx=E.T @ m.Sx @ E,
        Sy=E.T @ m.Sy @ E,
        nx=m.nx,
        ny=m.ny,
        domain_polygon=m.domain_polygon,
        cut=m.cut,
        volume_quadrature=None,
    )


def directional_sbp_residuals_oblique(m: ObliqueCutMatrices2D) -> tuple[float, float]:
    rx = np.linalg.norm(m.Qx + m.Qx.T - m.Sx, ord=np.inf)
    ry = np.linalg.norm(m.Qy + m.Qy.T - m.Sy, ord=np.inf)
    return float(rx), float(ry)


def mass_condition_number(H: np.ndarray) -> float:
    s = svdvals(0.5 * (H + H.T))
    return float(s[0] / s[-1])


def heat_lambda_max(K: np.ndarray, H: np.ndarray) -> float:
    vals = eigvalsh(0.5 * (K + K.T), 0.5 * (H + H.T))
    return float(vals[-1])


def make_oblique_cut_pair(
    n_elements: int = 8,
    degree: int = 3,
    epsilon: float = 0.05,
    angle: float = 0.71,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    quadrature_order: int | None = None,
) -> tuple[ObliqueCutMatrices2D, ObliqueCutMatrices2D, ObliqueAggregation2D, SplineBasis1D, SplineBasis1D]:
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    normal = np.array([cos(angle), sin(angle)], dtype=float)
    normal = normal / np.linalg.norm(normal)
    h = 1.0 / n_elements
    offset = float((normal[0] + normal[1]) * h * (1.0 - epsilon))
    cut = HalfPlaneCut(normal=normal, offset=offset)
    m_all = assemble_oblique_cut_matrices(basis_x, basis_y, cut, quadrature_order=quadrature_order)
    agg = polynomial_preserving_oblique_extension(
        basis_x,
        basis_y,
        m_all,
        support_threshold=support_threshold,
        preserve_degree=preserve_degree,
    )
    full = restrict_oblique_2d(m_all, agg.active)
    red = pullback_oblique_2d(m_all, agg.E)
    return full, red, agg, basis_x, basis_y
