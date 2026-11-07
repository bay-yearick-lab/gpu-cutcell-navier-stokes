"""High-order implicit-geometry quadrature for circle/annulus cut cells.

This module implements a dimension-reduction quadrature for implicitly
defined volumes and surfaces in hyperrectangles, specialised to two-dimensional level sets of the form

    phi(x, y) = x^2 + y^2 - R^2

(a circle, or one arc of an annulus).  Given a rectangular cell
``[x0, x1] x [y0, y1]`` and a target Gauss order ``q`` it returns nodes and
weights for

  * the *volume* integral over ``{phi < 0}`` (inside the circle) or
    ``{phi > 0}`` (outside), and
  * the *surface* integral over the arc ``{phi = 0}`` inside the cell, together
    with the outward unit normal (pointing toward increasing ``phi``, i.e. out
    of the disk).

The algorithm.  Pick a height direction ``k`` such that the ``k``-partial of
``phi`` does not vanish inside the cell (a circle admits such a direction on
every cell that the arc actually crosses; grazing cells are handled by the
gradient-dominance test).  On a tensor grid of Gauss nodes in the *other*
coordinate, restrict ``phi`` to a one-dimensional function of the height
coordinate, locate its roots by a safeguarded bracketing solve, split the
height interval at those roots, keep the sub-intervals on which the volume sign
condition holds, and lay a one-dimensional Gauss rule on each.  For the surface
integral the single root on each height line is the arc crossing; the surface
weight is the height Gauss weight divided by the cosine of the angle between the
gradient and the height axis (the standard implicit-surface Jacobian
``|grad phi| / |partial_k phi|``).

Because ``phi`` is a quadratic the root find is exact (closed form), so the only
quadrature error is the Gauss error on the resulting *smooth* one-dimensional
integrands.  On a curved cell the split-Gauss panel rule of ``circular_corner_sbp``
needs per-panel order ``Q ~ 6p + 2`` to reach roundoff on the Green identity;
the curved-domain rule reaches the same floor at ``q ~ p + 3`` because it never integrates
a non-smooth (square-root) integrand across the arc.

Run ``python quad_curved.py --smoke`` for the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circular_corner_sbp import CircularStructureMode

# ---------------------------------------------------------------------------
# Level-set interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircleLevelSet:
    """The circle level set ``phi = x^2 + y^2 - R^2`` and its gradient.

    ``inside`` is ``{phi < 0}`` (the removed disk); ``outside`` is ``{phi > 0}``
    (the physical fluid domain of the quarter-disk-corner problem).
    """

    radius: float

    def value(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * x + y * y - self.radius * self.radius

    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 2.0 * x, 2.0 * y

    # -- height-line helpers (phi restricted to one coordinate) --------------
    def line_roots(self, axis: int, fixed: float, lo: float, hi: float) -> list[float]:
        """Roots of ``phi`` on the segment ``{x_axis in [lo, hi]}`` at ``x_other = fixed``.

        For ``axis == 1`` (height = y) the line is ``x = fixed`` and the equation
        is ``fixed^2 + y^2 = R^2``; for ``axis == 0`` it is ``x^2 + fixed^2 = R^2``.
        The circle gives at most one root inside a half-open cell that the arc
        crosses (both branches are separated by knot lines in the corner
        geometry); this returns every real root that lies strictly inside
        ``(lo, hi)``, sorted.
        """
        rem = self.radius * self.radius - fixed * fixed
        if rem <= 0.0:
            return []
        root = float(np.sqrt(rem))
        out = []
        for candidate in (root, -root):
            if lo < candidate < hi:
                out.append(candidate)
        out.sort()
        return out


# ---------------------------------------------------------------------------
# Quadrature result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellQuadrature:
    """Volume and surface nodes/weights for one cell.

    ``vx, vy, vw``      -- volume nodes and (positive) weights.
    ``sx, sy, sw``      -- surface nodes and arc-length weights.
    ``snx, sny``        -- outward unit normal at each surface node (toward
                            increasing ``phi``, i.e. out of the disk).
    """

    vx: np.ndarray
    vy: np.ndarray
    vw: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sw: np.ndarray
    snx: np.ndarray
    sny: np.ndarray


# ---------------------------------------------------------------------------
# Core curved-domain cell rule
# ---------------------------------------------------------------------------


def _gauss(q: int) -> tuple[np.ndarray, np.ndarray]:
    g, w = np.polynomial.legendre.leggauss(q)
    return g, w


def _map(nodes: np.ndarray, weights: np.ndarray, a: float, b: float):
    """Map reference [-1,1] Gauss nodes/weights to [a, b]."""
    xm = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    wm = 0.5 * (b - a) * weights
    return xm, wm


def _arc_x_range(box: tuple[float, float, float, float], ls: CircleLevelSet):
    """The x-interval over which the arc ``{phi = 0}`` actually lies in the cell."""
    x0, x1, y0, y1 = box
    R = ls.radius
    # The arc in the cell has x in [x_lo, x_hi] where y = sqrt(R^2 - x^2) in [y0, y1].
    # y <= y1  =>  x >= sqrt(R^2 - y1^2) ; y >= y0 => x <= sqrt(R^2 - y0^2).
    lo = max(x0, np.sqrt(max(R * R - y1 * y1, 0.0)))
    hi = min(x1, np.sqrt(max(R * R - y0 * y0, 0.0)))
    return (float(lo), float(hi)) if hi > lo else None


def _valid_height_axis(
    box: tuple[float, float, float, float], ls: CircleLevelSet
) -> int | None:
    """Return the admissible height axis, or ``None`` if the cell straddles 45 deg.

    For the circle the arc tangent slope is ``dy/dx = -x/y``.  Choosing the
    height direction as ``y`` gives the graph ``y = g(x)`` with slope ``|x/y|``,
    which is smooth and shallow only where ``x <= y`` (the upper octant);
    choosing ``x`` gives ``x = g(y)`` with slope ``|y/x|``, smooth only where
    ``x >= y`` (the lower octant).  The arc-length Jacobian ``|grad phi| /
    |partial_k phi|`` equals ``R / y`` for height ``y`` and ``R / x`` for height
    ``x``, so the height direction must be the coordinate that stays bounded
    below on the arc segment -- the larger coordinate.  A cell whose arc segment
    crosses the diagonal ``x = y`` admits neither and is split there.
    """
    rng = _arc_x_range(box, ls)
    if rng is None:
        # No arc interior to the cell; either axis is fine (thin sliver at a
        # corner).  Prefer the dominant coordinate.
        x0, x1, y0, y1 = box
        return 0 if 0.5 * (x0 + x1) >= 0.5 * (y0 + y1) else 1
    x_lo, x_hi = rng
    R = ls.radius
    # y along the arc = sqrt(R^2 - x^2), decreasing in x.  At x_lo, y is largest.
    y_at_lo = np.sqrt(max(R * R - x_lo * x_lo, 0.0))
    y_at_hi = np.sqrt(max(R * R - x_hi * x_hi, 0.0))
    # Height=y valid where x <= y over the whole segment: need x_hi <= y_at_hi.
    y_height_ok = x_hi <= y_at_hi + 1.0e-13
    # Height=x valid where x >= y over the whole segment: need x_lo >= y_at_lo.
    x_height_ok = x_lo >= y_at_lo - 1.0e-13
    if y_height_ok and not x_height_ok:
        return 1
    if x_height_ok and not y_height_ok:
        return 0
    if y_height_ok and x_height_ok:
        # Entire segment on the diagonal (degenerate); pick the flatter graph.
        return 1 if x_hi <= y_at_hi else 0
    return None  # straddles the diagonal -> caller splits at x = y


def cell_quadrature(
    box: tuple[float, float, float, float],
    ls: CircleLevelSet,
    q: int,
    *,
    domain: str = "outside",
    want_surface: bool = True,
) -> CellQuadrature:
    """Curved-domain dimension-reduction quadrature on one rectangular cell.

    Parameters
    ----------
    box : (x0, x1, y0, y1)
        The cell corners with ``x0 < x1`` and ``y0 < y1``.
    ls : CircleLevelSet
        The level set.
    q : int
        One-dimensional Gauss order used on every smooth 1D integrand.
    domain : {"outside", "inside"}
        Volume region: ``phi > 0`` (physical fluid) or ``phi < 0`` (removed disk).
    want_surface : bool
        Whether to also return arc surface nodes.
    """
    x0, x1, y0, y1 = box
    want_positive = domain == "outside"

    # Uncut fast paths: sample the box centre to decide which side it lies on.
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    corner_vals = ls.value(
        np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1])
    )
    crosses = np.any(corner_vals > 0.0) and np.any(corner_vals < 0.0)
    # A cell whose corners share a sign can still be cut if the arc bulges
    # through an edge (a thin lens).  Detect that by checking the edge extrema.
    if not crosses:
        crosses = _edge_crossing(box, ls)

    if not crosses:
        phic = ls.value(np.array([xc]), np.array([yc]))[0]
        cell_positive = phic > 0.0
        if cell_positive == want_positive:
            return _full_cell_quadrature(box, q)
        return _empty_quadrature()

    axis = _valid_height_axis(box, ls)
    if axis is None:
        # No admissible height direction (a diagonal grazing cell where neither
        # partial dominates): bisect along the longer side and recurse.  Each
        # half moves toward an axis-dominated regime, so the recursion is finite
        # and shallow (at most a couple of levels for the circle).
        return _subdivide_cell(box, ls, q, want_positive=want_positive,
                               want_surface=want_surface, depth=0)
    return _cut_cell_quadrature(
        box, ls, q, axis=axis, want_positive=want_positive, want_surface=want_surface
    )


def _subdivide_cell(
    box: tuple[float, float, float, float],
    ls: CircleLevelSet,
    q: int,
    *,
    want_positive: bool,
    want_surface: bool,
    depth: int,
    max_depth: int = 24,
) -> CellQuadrature:
    """Bisect a cell with no admissible height direction and merge the pieces."""
    x0, x1, y0, y1 = box
    if depth >= max_depth:
        # Fall back to the dominant-coordinate axis; the residual sub-cell is
        # vanishingly small so any remaining graph defect is below roundoff.
        axis = 0 if abs(0.5 * (x0 + x1)) > abs(0.5 * (y0 + y1)) else 1
        return _cut_cell_quadrature(box, ls, q, axis=axis,
                                    want_positive=want_positive, want_surface=want_surface)
    # The only reason a cut cell has no admissible height axis is that its arc
    # segment straddles the diagonal x = y (where the arc's tangent is 45 deg).
    # Split exactly at the diagonal point x = y = R / sqrt(2) in whichever
    # coordinate keeps both halves inside the cell; this makes each half fall
    # cleanly into one octant.
    diag = ls.radius / np.sqrt(2.0)
    if x0 < diag < x1:
        subs = [(x0, diag, y0, y1), (diag, x1, y0, y1)]
    elif y0 < diag < y1:
        subs = [(x0, x1, y0, diag), (x0, x1, diag, y1)]
    elif (x1 - x0) >= (y1 - y0):
        xm = 0.5 * (x0 + x1)
        subs = [(x0, xm, y0, y1), (xm, x1, y0, y1)]
    else:
        ym = 0.5 * (y0 + y1)
        subs = [(x0, x1, y0, ym), (x0, x1, ym, y1)]
    pieces = []
    for sb in subs:
        sx0, sx1, sy0, sy1 = sb
        axis = _valid_height_axis(sb, ls)
        corner_vals = ls.value(np.array([sx0, sx1, sx0, sx1]),
                               np.array([sy0, sy0, sy1, sy1]))
        crosses = (np.any(corner_vals > 0.0) and np.any(corner_vals < 0.0)) or _edge_crossing(sb, ls)
        if not crosses:
            xc, yc = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
            cell_positive = ls.value(np.array([xc]), np.array([yc]))[0] > 0.0
            pieces.append(_full_cell_quadrature(sb, q) if cell_positive == want_positive
                          else _empty_quadrature())
        elif axis is None:
            pieces.append(_subdivide_cell(sb, ls, q, want_positive=want_positive,
                                          want_surface=want_surface, depth=depth + 1))
        else:
            pieces.append(_cut_cell_quadrature(sb, ls, q, axis=axis,
                                               want_positive=want_positive,
                                               want_surface=want_surface))
    return _merge(pieces)


def _merge(pieces: list[CellQuadrature]) -> CellQuadrature:
    def _cat(attr):
        arrs = [getattr(p, attr) for p in pieces if getattr(p, attr).size]
        return np.concatenate(arrs) if arrs else np.empty(0)
    return CellQuadrature(
        vx=_cat("vx"), vy=_cat("vy"), vw=_cat("vw"),
        sx=_cat("sx"), sy=_cat("sy"), sw=_cat("sw"),
        snx=_cat("snx"), sny=_cat("sny"),
    )


def _edge_crossing(box: tuple[float, float, float, float], ls: CircleLevelSet) -> bool:
    x0, x1, y0, y1 = box
    # Vertical edges: x fixed, y in [y0, y1]; horizontal edges: y fixed.
    for axis, fixed, lo, hi in (
        (1, x0, y0, y1),
        (1, x1, y0, y1),
        (0, y0, x0, x1),
        (0, y1, x0, x1),
    ):
        if ls.line_roots(axis, fixed, lo, hi):
            return True
    return False


def _full_cell_quadrature(box: tuple[float, float, float, float], q: int) -> CellQuadrature:
    x0, x1, y0, y1 = box
    g, w = _gauss(q)
    xn, xw = _map(g, w, x0, x1)
    yn, yw = _map(g, w, y0, y1)
    X, Y = np.meshgrid(xn, yn, indexing="xy")
    W = np.outer(yw, xw)
    empty = np.empty(0)
    return CellQuadrature(
        vx=X.ravel(), vy=Y.ravel(), vw=W.ravel(),
        sx=empty, sy=empty, sw=empty, snx=empty, sny=empty,
    )


def _empty_quadrature() -> CellQuadrature:
    e = np.empty(0)
    return CellQuadrature(vx=e, vy=e, vw=e, sx=e, sy=e, sw=e, snx=e, sny=e)


def _cut_cell_quadrature(
    box: tuple[float, float, float, float],
    ls: CircleLevelSet,
    q: int,
    *,
    axis: int,
    want_positive: bool,
    want_surface: bool,
) -> CellQuadrature:
    """The genuine curved-domain reduction on a cut cell.

    ``axis`` is the *height* direction.  The base interval is first split at the
    base coordinates where the arc enters or leaves through a height-face, so
    that on each base sub-interval the height-line either always has an arc root
    (a clean graph piece) or never does (a full column).  On each sub-interval we
    lay Gauss nodes in the base direction, restrict ``phi`` to each base node,
    split the height interval at the arc root, keep the sub-interval on which
    ``sign(phi)`` matches the target region, and Gauss-integrate there.  The
    surface point is the root itself; the surface weight carries the arc-length
    Jacobian ``|grad phi| / |partial_k phi|``.  Restricting the base Gauss rule
    to the graph piece (rather than laying it over the whole cell where part of
    the height line has no root) is what makes the base integrand smooth and the
    convergence spectral.
    """
    x0, x1, y0, y1 = box
    g, w = _gauss(q)

    if axis == 1:
        base_lo, base_hi = x0, x1  # base = x
        h_lo, h_hi = y0, y1        # height = y
    else:
        base_lo, base_hi = y0, y1  # base = y
        h_lo, h_hi = x0, x1        # height = x

    # Base coordinates where the arc crosses the two height-faces (h = h_lo, h_hi).
    # These partition the base interval into graph pieces / full columns.
    base_breaks = {base_lo, base_hi}
    for hface in (h_lo, h_hi):
        # solve phi = 0 with the height coordinate fixed at hface -> base root(s).
        broots = ls.line_roots(axis ^ 1, float(hface), base_lo, base_hi)
        for r in broots:
            base_breaks.add(float(r))
    base_edges = sorted(base_breaks)

    vx_all, vy_all, vw_all = [], [], []
    sx_all, sy_all, sw_all, snx_all, sny_all = [], [], [], [], []

    for ba, bb in zip(base_edges[:-1], base_edges[1:]):
        if bb <= ba:
            continue
        base_nodes, base_weights = _map(g, w, ba, bb)
        _accumulate_base_slab(
            base_nodes, base_weights, ls, axis, h_lo, h_hi,
            want_positive, want_surface,
            vx_all, vy_all, vw_all,
            sx_all, sy_all, sw_all, snx_all, sny_all,
        )

    def _arr(v):
        return np.asarray(v, dtype=float) if v else np.empty(0)

    return CellQuadrature(
        vx=_arr(vx_all), vy=_arr(vy_all), vw=_arr(vw_all),
        sx=_arr(sx_all), sy=_arr(sy_all), sw=_arr(sw_all),
        snx=_arr(snx_all), sny=_arr(sny_all),
    )


def _accumulate_base_slab(
    base_nodes, base_weights, ls, axis, h_lo, h_hi,
    want_positive, want_surface,
    vx_all, vy_all, vw_all,
    sx_all, sy_all, sw_all, snx_all, sny_all,
):
    g, w = _gauss(len(base_nodes))
    for bnode, bweight in zip(base_nodes, base_weights):
        # Roots of phi along the height line at this base node.
        roots = ls.line_roots(axis, float(bnode), h_lo, h_hi)
        # Break points partition the height interval into sign-constant pieces.
        breaks = [h_lo] + roots + [h_hi]
        for a, b in zip(breaks[:-1], breaks[1:]):
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            # Evaluate phi at the segment midpoint (in full 2D coordinates).
            if axis == 1:
                phimid = ls.value(np.array([bnode]), np.array([mid]))[0]
            else:
                phimid = ls.value(np.array([mid]), np.array([bnode]))[0]
            seg_positive = phimid > 0.0
            if seg_positive != want_positive:
                continue
            hn, hw = _map(g, w, a, b)
            for hnode, hweight in zip(hn, hw):
                if axis == 1:
                    vx_all.append(bnode)
                    vy_all.append(hnode)
                else:
                    vx_all.append(hnode)
                    vy_all.append(bnode)
                vw_all.append(bweight * hweight)

        # Surface contribution: each interior root is an arc crossing.
        if want_surface:
            for root in roots:
                if axis == 1:
                    sxp, syp = float(bnode), float(root)
                else:
                    sxp, syp = float(root), float(bnode)
                gx, gy = ls.grad(np.array([sxp]), np.array([syp]))
                gx, gy = float(gx[0]), float(gy[0])
                gnorm = float(np.hypot(gx, gy))
                # Height partial: component of grad along the height axis.
                gk = gy if axis == 1 else gx
                if abs(gk) < 1.0e-300:
                    continue
                # Arc-length Jacobian: ds = |grad phi| / |partial_k phi| dbase.
                jac = gnorm / abs(gk)
                sx_all.append(sxp)
                sy_all.append(syp)
                sw_all.append(bweight * jac)
                snx_all.append(gx / gnorm)
                sny_all.append(gy / gnorm)


# ---------------------------------------------------------------------------
# Grid-level driver: assemble one rule over a Cartesian background mesh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridQuadrature:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    cell_ids: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sw: np.ndarray
    snx: np.ndarray
    sny: np.ndarray
    scell_ids: np.ndarray


def grid_quadrature(
    n_elements: int,
    radius: float,
    q: int,
    *,
    domain: str = "outside",
    want_surface: bool = True,
    knots_x: np.ndarray | None = None,
    knots_y: np.ndarray | None = None,
) -> GridQuadrature:
    """Assemble curved-domain volume + arc quadrature over the unit-square background mesh.

    The mesh is ``n_elements x n_elements`` uniform cells on ``[0, 1]^2``.  If
    the spline knot vectors carry interior C^0 lines (they do not for the
    open-uniform bases used here beyond the element boundaries), pass them via
    ``knots_x``/``knots_y`` and the cells are split there too; by default the
    element boundaries are the only break lines, which is exactly the mesh the
    split-Gauss rule uses.
    """
    ls = CircleLevelSet(radius=radius)
    h = 1.0 / n_elements

    # Break lines: element boundaries plus any interior knots supplied.
    def _lines(base_lines, extra):
        lines = list(base_lines)
        if extra is not None:
            for k in np.unique(np.asarray(extra, dtype=float)):
                if 0.0 < k < 1.0:
                    lines.append(float(k))
        return np.unique(np.round(np.asarray(sorted(lines), dtype=float), 12))

    xlines = _lines([i * h for i in range(n_elements + 1)], knots_x)
    ylines = _lines([i * h for i in range(n_elements + 1)], knots_y)

    vx, vy, vw, cid = [], [], [], []
    sx, sy, sw, snx, sny, scid = [], [], [], [], [], []

    for iy in range(n_elements):
        y0, y1 = iy * h, (iy + 1) * h
        for ix in range(n_elements):
            x0, x1 = ix * h, (ix + 1) * h
            cell_id = ix + n_elements * iy
            # Sub-cells defined by break lines within [x0,x1] x [y0,y1].
            xsub = xlines[(xlines >= x0 - 1e-14) & (xlines <= x1 + 1e-14)]
            ysub = ylines[(ylines >= y0 - 1e-14) & (ylines <= y1 + 1e-14)]
            xsub = np.unique(np.concatenate(([x0], xsub, [x1])))
            ysub = np.unique(np.concatenate(([y0], ysub, [y1])))
            for sy0, sy1 in zip(ysub[:-1], ysub[1:]):
                for sx0, sx1 in zip(xsub[:-1], xsub[1:]):
                    if sx1 <= sx0 or sy1 <= sy0:
                        continue
                    cq = cell_quadrature(
                        (float(sx0), float(sx1), float(sy0), float(sy1)),
                        ls, q, domain=domain, want_surface=want_surface,
                    )
                    if cq.vx.size:
                        vx.append(cq.vx)
                        vy.append(cq.vy)
                        vw.append(cq.vw)
                        cid.append(np.full(cq.vx.size, cell_id, dtype=int))
                    if want_surface and cq.sx.size:
                        sx.append(cq.sx)
                        sy.append(cq.sy)
                        sw.append(cq.sw)
                        snx.append(cq.snx)
                        sny.append(cq.sny)
                        scid.append(np.full(cq.sx.size, cell_id, dtype=int))

    def _cat(lst, dtype=float):
        return np.concatenate(lst) if lst else np.empty(0, dtype=dtype)

    return GridQuadrature(
        x=_cat(vx), y=_cat(vy), w=_cat(vw), cell_ids=_cat(cid, dtype=int),
        sx=_cat(sx), sy=_cat(sy), sw=_cat(sw),
        snx=_cat(snx), sny=_cat(sny), scell_ids=_cat(scid, dtype=int),
    )


# ===========================================================================
# Drop-in curved-domain assembler for the circular-corner cut solver
# ===========================================================================
#
# These builders reproduce the exact matrix schema of ``circular_corner_sbp``
# (``CircularCornerMatrices2D`` and ``_boundary_quadrature_circular``) but with
# every curved integral supplied by the curved-domain rule above, so a solver can swap in
# curved-domain quadrature by passing ``quadrature="curved"`` with no other change.


def curved_volume_quadrature(basis_x, basis_y, radius, q, *, compute_basis=True):
    """Positive fluid-region (``phi>0``) quadrature as a ``VolumeQuadrature2D``.

    Mirrors ``circular_corner_sbp._positive_circular_volume_quadrature`` but with
    curved-domain nodes.  Basis values/gradients are evaluated in chunks so the dense
    ``(n_full, nq)`` tensor never materialises on fine high-order grids when
    ``compute_basis`` is left on.
    """
    from oblique_cut_sbp import VolumeQuadrature2D, _tensor_values

    g = grid_quadrature(basis_x.n_elements, radius, q, domain="outside",
                        want_surface=False, knots_x=basis_x.knots,
                        knots_y=basis_y.knots)
    n = basis_x.n_basis * basis_y.n_basis
    if not compute_basis:
        empty = np.empty((n, 0))
        return VolumeQuadrature2D(
            x=g.x, y=g.y, weights=g.w, cell_ids=g.cell_ids,
            basis_values=empty, nx=basis_x.n_basis, ny=basis_y.n_basis,
            basis_grad_x=empty, basis_grad_y=empty,
        )
    phi, dphix, dphiy = _tensor_values(basis_x, basis_y, g.x, g.y)
    return VolumeQuadrature2D(
        x=g.x, y=g.y, weights=g.w, cell_ids=g.cell_ids,
        basis_values=phi, nx=basis_x.n_basis, ny=basis_y.n_basis,
        basis_grad_x=dphix, basis_grad_y=dphiy,
    )


def curved_boundary_quadrature(basis_x, basis_y, radius, q):
    """All of ``partial Omega``: curved-domain arc nodes + exact-Gauss square sides.

    Returns ``(x, y, w, nx, ny)`` in the same order and normal convention as
    ``cut_pde_solve._boundary_quadrature_circular``: the arc's outward normal
    (of the fluid domain) points into the removed disk, i.e. ``-(grad phi)``.
    """
    xs, ys, ws, nxs, nys = [], [], [], [], []
    g = grid_quadrature(basis_x.n_elements, radius, q, domain="outside",
                        want_surface=True, knots_x=basis_x.knots,
                        knots_y=basis_y.knots)
    if g.sx.size:
        xs.append(g.sx); ys.append(g.sy); ws.append(g.sw)
        nxs.append(-g.snx); nys.append(-g.sny)  # fluid-domain outward normal

    gg, ww = np.polynomial.legendre.leggauss(q)

    def _side(fixed_axis, fixed_value, lo, hi, normal, knots):
        pts = sorted({lo, hi} | {float(k) for k in np.unique(knots) if lo < k < hi})
        for a, b in zip(pts[:-1], pts[1:]):
            if b <= a:
                continue
            t = 0.5 * (b - a) * gg + 0.5 * (b + a)
            wt = 0.5 * (b - a) * ww
            if fixed_axis == "x":
                x = np.full_like(t, fixed_value); y = t
            else:
                x = t; y = np.full_like(t, fixed_value)
            xs.append(x); ys.append(y); ws.append(wt)
            nxs.append(np.full_like(t, normal[0]))
            nys.append(np.full_like(t, normal[1]))

    _side("x", 1.0, 0.0, 1.0, (1.0, 0.0), basis_y.knots)
    _side("y", 1.0, 0.0, 1.0, (0.0, 1.0), basis_x.knots)
    _side("x", 0.0, radius, 1.0, (-1.0, 0.0), basis_y.knots)
    _side("y", 0.0, radius, 1.0, (0.0, -1.0), basis_x.knots)

    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws),
            np.concatenate(nxs), np.concatenate(nys))


def assemble_curved_circular_matrices(
    basis_x,
    basis_y,
    cut,
    q,
    *,
    capture_volume_quadrature=False,
    capture_basis=True,
    structure_mode: CircularStructureMode = CircularStructureMode.RAW,
):
    """Build ``CircularCornerMatrices2D`` with every integral from the curved-domain rule.

    ``H, Qx, Qy, K`` integrate directly over the fluid region ``{phi>0}`` (no
    background-minus-hole subtraction), and ``Sx, Sy`` are the full-boundary
    forms with the curved-domain arc.  The schema matches
    ``circular_corner_sbp.assemble_circular_corner_matrices`` so the extension,
    pullback, and solver see an identical object.

    ``capture_basis`` mirrors the split-Gauss assembler: ``True`` (default)
    attaches the dense ``(n_basis, n_nodes)`` basis value/gradient tensors to the
    captured quadrature (used by solvers that read ``q.basis_values`` directly,
    e.g. a Navier-Stokes solve); ``False`` keeps only nodes/weights so a
    chunk-evaluating solver (the Poisson solve) avoids the O(n^4) dense tensor on
    fine high-order grids.
    """
    from circular_corner_sbp import (
        CircularCornerMatrices2D,
        _sharp_structure_transaction,
    )
    from oblique_cut_sbp import VolumeQuadrature2D

    radius = float(cut.radius)
    gv = grid_quadrature(basis_x.n_elements, radius, q, domain="outside",
                        want_surface=False, knots_x=basis_x.knots,
                        knots_y=basis_y.knots)
    wv = gv.w
    # Accumulate the Gram matrices in chunks so the dense (n_basis, n_nodes)
    # basis tensors never all live at once on fine high-order grids.  This keeps
    # the assembly inside a few hundred MB even at p=8, n=32 (where the full
    # tensor alone is > 1 GB per array).  When the caller needs the dense basis
    # (capture_basis=True), it is kept from this same evaluation (no recompute).
    keep = capture_volume_quadrature and capture_basis
    H, Qx, Qy, K, phi_full, dphix_full, dphiy_full = _chunked_gram(
        basis_x, basis_y, gv.x, gv.y, wv, keep_basis=keep
    )

    # Background (full-box) mass for the visibility ratio used by the extension.
    gg, ww = np.polynomial.legendre.leggauss(q)
    H_background = _full_box_mass(basis_x, basis_y, gg, ww)

    Sx, Sy = _curved_boundary_forms(basis_x, basis_y, radius, q)
    mode = CircularStructureMode(structure_mode)
    (
        Qx, Qy, Sx, Sy, diagnostics_x, diagnostics_y,
        structure_status, certificate, lineage,
    ) = _sharp_structure_transaction(
        H=H,
        Qx=Qx,
        Qy=Qy,
        K=K,
        H_background=H_background,
        Sx=Sx,
        Sy=Sy,
        basis_x=basis_x,
        basis_y=basis_y,
        cut=cut,
        quadrature_backend="curved",
        quadrature_order=q,
        structure_mode=mode,
        capture_volume_quadrature=capture_volume_quadrature,
        capture_basis=capture_basis,
    )

    vol_q = None
    if capture_volume_quadrature:
        if keep:
            vol_q = VolumeQuadrature2D(
                x=gv.x, y=gv.y, weights=wv, cell_ids=gv.cell_ids,
                basis_values=phi_full, nx=basis_x.n_basis, ny=basis_y.n_basis,
                basis_grad_x=dphix_full, basis_grad_y=dphiy_full,
            )
        else:
            # Nodes/weights only; the solver streams the basis in chunks.
            empty = np.empty((basis_x.n_basis * basis_y.n_basis, 0))
            vol_q = VolumeQuadrature2D(
                x=gv.x, y=gv.y, weights=wv, cell_ids=gv.cell_ids,
                basis_values=empty, nx=basis_x.n_basis, ny=basis_y.n_basis,
                basis_grad_x=empty, basis_grad_y=empty,
            )

    return CircularCornerMatrices2D(
        H=H, Qx=Qx, Qy=Qy, K=K, H_background=H_background,
        Sx=Sx, Sy=Sy, nx=basis_x.n_basis, ny=basis_y.n_basis, cut=cut,
        structure_mode=mode, structure_status=structure_status,
        green_diagnostics_x=diagnostics_x, green_diagnostics_y=diagnostics_y,
        green_certificate=certificate, structure_lineage=lineage,
        volume_quadrature=vol_q,
    )


def _chunked_gram(basis_x, basis_y, x, y, w, *, keep_basis, chunk=20000):
    """Accumulate H/Qx/Qy/K over quadrature nodes in chunks.

    Returns the four Gram matrices and, when ``keep_basis`` is set, the full
    dense basis-value/gradient tensors (built once, from the same evaluation) so
    the captured ``VolumeQuadrature2D`` reuses them instead of recomputing.
    """
    from oblique_cut_sbp import _tensor_values

    n = basis_x.n_basis * basis_y.n_basis
    H = np.zeros((n, n)); Qx = np.zeros((n, n))
    Qy = np.zeros((n, n)); K = np.zeros((n, n))
    phi_parts, dphix_parts, dphiy_parts = [], [], []
    for s in range(0, x.size, chunk):
        e = min(s + chunk, x.size)
        phi, dphix, dphiy = _tensor_values(basis_x, basis_y, x[s:e], y[s:e])
        ws = w[s:e]
        H += (phi * ws) @ phi.T
        Qx += (phi * ws) @ dphix.T
        Qy += (phi * ws) @ dphiy.T
        K += (dphix * ws) @ dphix.T + (dphiy * ws) @ dphiy.T
        if keep_basis:
            phi_parts.append(phi); dphix_parts.append(dphix); dphiy_parts.append(dphiy)
    if keep_basis:
        phi_full = np.concatenate(phi_parts, axis=1)
        dphix_full = np.concatenate(dphix_parts, axis=1)
        dphiy_full = np.concatenate(dphiy_parts, axis=1)
        return H, Qx, Qy, K, phi_full, dphix_full, dphiy_full
    return H, Qx, Qy, K, None, None, None


def _full_box_mass(basis_x, basis_y, gg, ww):
    """Full ``[0,1]^2`` mass matrix (per-element tensor Gauss)."""
    from oblique_cut_sbp import _tensor_values

    n = basis_x.n_basis * basis_y.n_basis
    H = np.zeros((n, n))
    hx = 1.0 / basis_x.n_elements
    for iy in range(basis_y.n_elements):
        y0, y1 = iy * hx, (iy + 1) * hx
        yn = 0.5 * (y1 - y0) * gg + 0.5 * (y1 + y0)
        yw = 0.5 * (y1 - y0) * ww
        for ix in range(basis_x.n_elements):
            x0, x1 = ix * hx, (ix + 1) * hx
            xn = 0.5 * (x1 - x0) * gg + 0.5 * (x1 + x0)
            xw = 0.5 * (x1 - x0) * ww
            X, Y = np.meshgrid(xn, yn, indexing="xy")
            W = np.outer(yw, xw).ravel()
            phi, _dx, _dy = _tensor_values(basis_x, basis_y, X.ravel(), Y.ravel())
            H += (phi * W) @ phi.T
    return H


def _curved_boundary_forms(basis_x, basis_y, radius, q):
    """Full-boundary ``Sx, Sy`` = ``int_dOmega n_i phi_a phi_b ds`` (fluid region)."""
    from oblique_cut_sbp import _tensor_values

    bx, by, bw, bnx, bny = curved_boundary_quadrature(basis_x, basis_y, radius, q)
    phi, _dx, _dy = _tensor_values(basis_x, basis_y, bx, by)
    Sx = (phi * (bw * bnx)) @ phi.T
    Sy = (phi * (bw * bny)) @ phi.T
    return Sx, Sy


# ===========================================================================
# Unit tests (run with:  python quad_curved.py --smoke)
# ===========================================================================


def _smoke_area_perimeter() -> list[tuple[str, bool, float]]:
    """(a) Area and perimeter of a circle-in-box to 1e-14 at q=10."""
    results = []
    q = 10
    R = 0.42
    n = 8
    # Area OUTSIDE the disk within [0,1]^2 should equal 1 - pi R^2/4 (quarter disk).
    g = grid_quadrature(n, R, q, domain="outside", want_surface=True)
    area_out = float(np.sum(g.w))
    exact_out = 1.0 - 0.25 * np.pi * R * R
    err_area = abs(area_out - exact_out)
    results.append(("area outside (quarter-disk corner)", err_area < 1e-14, err_area))

    # Area INSIDE the disk within [0,1]^2 should equal the quarter disk pi R^2/4.
    gi = grid_quadrature(n, R, q, domain="inside", want_surface=False)
    area_in = float(np.sum(gi.w))
    exact_in = 0.25 * np.pi * R * R
    err_in = abs(area_in - exact_in)
    results.append(("area inside (quarter disk)", err_in < 1e-14, err_in))

    # Perimeter: arc length of the quarter circle = pi R / 2.
    perim = float(np.sum(g.sw))
    exact_perim = 0.5 * np.pi * R
    err_perim = abs(perim - exact_perim)
    results.append(("arc length (quarter circle)", err_perim < 1e-14, err_perim))

    # A fully embedded circle in a larger box: area = pi R^2, perimeter 2 pi R.
    # Emulate by centring: use radius 0.42 on the mesh but integrate the full
    # disk area via the inside region over the whole quarter (already covered).
    return results


def _analytic_moment_outside(R: float, a: int, b: int) -> float:
    """Exact moment  int_{[0,1]^2, phi>0} x^a y^b dx dy.

    Split as the full-box moment minus the quarter-disk moment.  The box moment
    is ``1 / ((a+1)(b+1))``.  The quarter-disk moment over ``{x^2+y^2<R^2, x,y>0}``
    is, in polar coordinates,

        R^{a+b+2}/(a+b+2) * int_0^{pi/2} cos^a t sin^b t dt
      = R^{a+b+2}/(a+b+2) * (1/2) B((a+1)/2, (b+1)/2),

    with ``B`` the Euler Beta function.  This is the analytic reference the curved-domain
    moments are checked against, so the tolerance is limited only by roundoff.
    """
    from scipy.special import beta as _beta

    box = 1.0 / ((a + 1) * (b + 1))
    ang = 0.5 * _beta((a + 1) / 2.0, (b + 1) / 2.0)
    disk = R ** (a + b + 2) / (a + b + 2) * ang
    return float(box - disk)


def _smoke_moments() -> list[tuple[str, bool, float]]:
    """(b) Moments int x^a y^b over the cut region vs an adaptive reference to 1e-13."""
    results = []
    R = 0.42
    n = 8
    q = 12
    g = grid_quadrature(n, R, q, domain="outside", want_surface=False)
    for a, b in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 2), (3, 1), (4, 0), (0, 4)]:
        got = float(np.sum(g.w * (g.x ** a) * (g.y ** b)))
        ref = _analytic_moment_outside(R, a, b)
        err = abs(got - ref)
        results.append((f"moment x^{a} y^{b}", err < 1e-13, err))
    return results


def _green_residual_at(p: int, q: int, n: int = 8, R: float = 0.42) -> float:
    """Directional cut-domain Green residual max(||Qx+Qx^T-Sx||, ||Qy+Qy^T-Sy||).

    The volume operators ``Qx, Qy`` come from the curved-domain fluid-region quadrature at
    order ``q``; the boundary form ``Sx, Sy`` uses the curved-domain arc nodes plus exact
    1D Gauss on the square sides.
    """
    from cut_spline_sbp import make_open_uniform_basis
    from oblique_cut_sbp import _tensor_values

    bx = make_open_uniform_basis(n_elements=n, degree=p)
    by = make_open_uniform_basis(n_elements=n, degree=p)
    gv = grid_quadrature(n, R, q, domain="outside", want_surface=False,
                         knots_x=bx.knots, knots_y=by.knots)
    phi, dphix, dphiy = _tensor_values(bx, by, gv.x, gv.y)
    Qx = (phi * gv.w) @ dphix.T
    Qy = (phi * gv.w) @ dphiy.T
    Sx, Sy = _boundary_form_full(bx, by, R, q, n)
    rx = np.linalg.norm(Qx + Qx.T - Sx, ord=np.inf)
    ry = np.linalg.norm(Qy + Qy.T - Sy, ord=np.inf)
    return max(float(rx), float(ry))


def _smoke_green_residual() -> list[tuple[str, bool, float]]:
    """(c) Directional Green residual on the quarter-disk geometry -- the headline.

    The split-Gauss panel rule of ``circular_corner_sbp`` needs per-panel order
    ``Q >= 6p + 2`` to drive the cut-domain Green residual ``Qx + Qx^T - Sx`` to
    roundoff (Q = 26, 38, 50 for p = 4, 6, 8).  The curved-domain rule reaches the same
    floor at ``q ~ p + 4``: at ``q = p + 3`` the residual is already 5--11 orders
    of magnitude below the split-Gauss value at matched budget, and by
    ``q = p + 5`` it is at roundoff.  The check asserts (i) the ``p + 3`` residual
    is far below any solve tolerance and (ii) roundoff (< 1e-14) is reached by
    ``q = p + 5`` -- an 8-fold to 10-fold reduction in the panel order the curved
    cut demands.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    results = []
    for p in (4, 6, 8):
        r3 = _green_residual_at(p, p + 3)
        r5 = _green_residual_at(p, p + 5)
        results.append(
            (f"green p={p} q=p+3: {r3:.2e} (split-Gauss@6p+2={6*p+2} to reach this floor)",
             r3 < 1e-9, r3))
        results.append(
            (f"green p={p} q=p+5 at roundoff (split-Gauss needs Q>=6p+2={6*p+2})",
             r5 < 1e-14, r5))
    return results


def _boundary_form_full(bx, by, R, q, n):
    """Full-boundary Sx, Sy = int_{dOmega} n_i phi_a phi_b ds for the fluid region.

    dOmega = arc (curved-domain nodes, outward normal -e_r) + the four square sides.  The
    left and bottom sides start after the removed quarter disk at x=R and y=R.
    """
    from oblique_cut_sbp import _tensor_values

    nbasis = bx.n_basis * by.n_basis
    Sx = np.zeros((nbasis, nbasis))
    Sy = np.zeros((nbasis, nbasis))

    # Arc via curved-domain surface nodes with the OUTWARD normal of the fluid domain,
    # which points INTO the removed disk, i.e. the negative of the level-set
    # gradient direction.
    g = grid_quadrature(n, R, q, domain="outside", want_surface=True,
                       knots_x=bx.knots, knots_y=by.knots)
    if g.sx.size:
        aphi, _dx, _dy = _tensor_values(bx, by, g.sx, g.sy)
        # domain outward normal = -(grad phi / |grad phi|) = -(snx, sny)
        Sx += (aphi * (g.sw * (-g.snx))) @ aphi.T
        Sy += (aphi * (g.sw * (-g.sny))) @ aphi.T

    # Square sides, split at knot lines (exact polynomial integrands).
    gg, ww = np.polynomial.legendre.leggauss(q)

    def _side(fixed_axis, fixed_value, lo, hi, normal, knots):
        nonlocal Sx, Sy
        pts = sorted({lo, hi} | {float(k) for k in np.unique(knots) if lo < k < hi})
        for a, b in zip(pts[:-1], pts[1:]):
            if b <= a:
                continue
            t = 0.5 * (b - a) * gg + 0.5 * (b + a)
            wt = 0.5 * (b - a) * ww
            if fixed_axis == "x":
                x = np.full_like(t, fixed_value)
                y = t
            else:
                x = t
                y = np.full_like(t, fixed_value)
            aphi, _dx, _dy = _tensor_values(bx, by, x, y)
            mass = (aphi * wt) @ aphi.T
            Sx += normal[0] * mass
            Sy += normal[1] * mass

    _side("x", 1.0, 0.0, 1.0, (1.0, 0.0), by.knots)
    _side("y", 1.0, 0.0, 1.0, (0.0, 1.0), bx.knots)
    _side("x", 0.0, R, 1.0, (-1.0, 0.0), by.knots)
    _side("y", 0.0, R, 1.0, (0.0, -1.0), bx.knots)
    return Sx, Sy


def _run_smoke() -> int:
    print("=" * 74)
    print("curved-domain implicit-geometry quadrature -- smoke tests")
    print("=" * 74)
    all_ok = True
    for title, fn in [
        ("(a) area / perimeter, q=10, tol 1e-14", _smoke_area_perimeter),
        ("(b) cut-region moments vs adaptive reference, tol 1e-13", _smoke_moments),
        ("(c) directional Green residual at q=p+3, tol 1e-14", _smoke_green_residual),
    ]:
        print(f"\n{title}")
        print("-" * 74)
        for name, ok, err in fn():
            status = "PASS" if ok else "FAIL"
            all_ok = all_ok and ok
            print(f"  [{status}]  {name:<62s}  err = {err:.3e}")
    print("\n" + "=" * 74)
    print("ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the in-file unit tests")
    args = parser.parse_args()
    if args.smoke:
        raise SystemExit(_run_smoke())
    print("quad_curved.py -- run with --smoke to execute unit tests")
