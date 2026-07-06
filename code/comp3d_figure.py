"""Section-6 figure for the 3D compressible Navier-Stokes cut-cell result.

Three panels, drawn
on the project palette via :mod:`plot_style`:

* (a) a real solved pressure field on the genuinely non-body-fitted sharp cut
  cube: three orthogonal cutaway planes through an evolved acoustic pulse, with
  the removed corner sliver carved out and shaded, on labelled x, y, z axes;
* (b) the discrete conservation ledger versus time: the per-step mass and
  total-energy budget residuals ride the round-off floor over a full acoustic
  transit, with the momentum-component budgets, the inviscid kinetic-energy
  production rate, and free-stream preservation pinned as round-off markers on the
  same floor;
* (c) the cut-independence relief: the explicit-step spectral radius stays at the
  background scale under mimetic aggregation as the corner-sliver volume fraction
  shrinks across nine decades, while the unaggregated active control blows up as
  1/eta -- a headline relief of ~3.7e3 at the tightest cut.

Run:  cd code && MPLBACKEND=Agg uv run python comp3d_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import plot_style as ps
from comp3d_field_snapshot import load_or_compute

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RESULTS = ROOT / "code" / "results"


# ---------------------------------------------------------------------------
# (a) pressure on the sharp 3D cut box: three cutaway planes + carved corner.
# ---------------------------------------------------------------------------
def _panel_field(ax) -> None:
    snap = load_or_compute()
    coords = snap["coords"]
    p = snap["pressure"]
    h = float(snap["h"])
    fr = np.asarray(snap["cut_fractions"], dtype=float)

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    zs = np.unique(coords[:, 2])
    nx, ny, nz = xs.size, ys.size, zs.size
    # coords are laid out z-outermost, x-fastest (build_grid kron order).
    P = p.reshape(nz, ny, nx)

    cmap = ps.FIELD_DIVERGING
    # Symmetric diverging normalisation about the ambient p0=1, saturated at the
    # field extremes so the small acoustic perturbation reads with full contrast.
    vmax = float(np.abs(p - 1.0).max())
    norm = Normalize(vmin=1.0 - vmax, vmax=1.0 + vmax)
    rc = cmap(norm(P))  # RGBA per node

    # The removed corner sliver: an element-sized cube carved from the origin.
    cut = max(fr.mean(), 0.5) * h
    xcut = ycut = zcut = cut

    # Three interior cutaway planes placed THROUGH the pulse (near the cube
    # centre) so the exposed field is the live wavefront, not the quiet corner.
    def nearest(vals, target):
        return int(np.argmin(np.abs(vals - target)))

    ixc, iyc, izc = nearest(xs, 0.55), nearest(ys, 0.55), nearest(zs, 0.55)
    xpl, ypl, zpl = xs[ixc], ys[iyc], zs[izc]

    def _quad_surface(Xg, Yg, Zg, C):
        ax.plot_surface(
            Xg, Yg, Zg, facecolors=C, rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False, zorder=2,
        )

    # z = zpl plane over (x, y).
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    _quad_surface(Xg, Yg, np.full_like(Xg, zpl), rc[izc, :, :])
    # y = ypl plane over (x, z).
    Xg, Zg = np.meshgrid(xs, zs, indexing="xy")
    _quad_surface(Xg, np.full_like(Xg, ypl), Zg, rc[:, iyc, :])
    # x = xpl plane over (y, z).
    Yg, Zg = np.meshgrid(ys, zs, indexing="xy")
    _quad_surface(np.full_like(Yg, xpl), Yg, Zg, rc[:, :, ixc])

    # Draw the outer cube edges and the carved-notch edges as a light wireframe.
    edge = ps.AXIS
    lw_out, lw_notch = 0.8, 1.3

    def seg(a, b, **kw):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **kw)

    lo, hi = 0.0, 1.0
    cube = [
        ((lo, lo, hi), (hi, lo, hi)), ((hi, lo, hi), (hi, hi, hi)),
        ((hi, hi, hi), (lo, hi, hi)), ((lo, hi, hi), (lo, lo, hi)),
        ((hi, lo, lo), (hi, hi, lo)), ((hi, hi, lo), (lo, hi, lo)),
        ((hi, lo, hi), (hi, lo, lo)), ((hi, hi, hi), (hi, hi, lo)),
        ((lo, hi, hi), (lo, hi, lo)),
    ]
    for a, b in cube:
        seg(a, b, color=edge, lw=lw_out, alpha=0.55, zorder=1)

    # The carved corner: a shaded wedge on the three notch faces meeting the
    # origin, and its exposed inner edges in the small-support accent.
    xc, yc, zc = xcut, ycut, zcut
    notch = [
        ((xc, lo, lo), (xc, yc, lo)), ((xc, yc, lo), (lo, yc, lo)),
        ((xc, yc, lo), (xc, yc, zc)),
        ((lo, lo, zc), (xc, lo, zc)), ((xc, lo, zc), (xc, yc, zc)),
        ((xc, yc, zc), (lo, yc, zc)), ((lo, yc, zc), (lo, lo, zc)),
        ((xc, lo, lo), (xc, lo, zc)), ((lo, yc, lo), (lo, yc, zc)),
    ]
    for a, b in notch:
        seg(a, b, color=ps.SMALL_SUPPORT, lw=lw_notch, alpha=0.9, zorder=6)

    # Faint hatched shading on the removed corner's outer faces (z=0, y=0, x=0).
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = [
        [(lo, lo, lo), (xc, lo, lo), (xc, yc, lo), (lo, yc, lo)],   # z=0
        [(lo, lo, lo), (xc, lo, lo), (xc, lo, zc), (lo, lo, zc)],   # y=0
        [(lo, lo, lo), (lo, yc, lo), (lo, yc, zc), (lo, lo, zc)],   # x=0
    ]
    pc = Poly3DCollection(
        faces, facecolor=ps.SETUP_BG, edgecolor=ps.SMALL_SUPPORT,
        linewidths=0.6, alpha=0.85, zorder=5,
    )
    ax.add_collection3d(pc)

    ax.text(
        xc + 0.06, yc * 0.5, zc + 0.02, "removed\ncorner sliver",
        color=ps.SMALL_SUPPORT, fontsize=7.4, ha="left", va="bottom", zorder=8,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$x$", labelpad=-8)
    ax.set_ylabel("$y$", labelpad=-8)
    ax.set_zlabel("$z$", labelpad=-8)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_zticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6.8, pad=-3)
    ax.view_init(elev=22, azim=-127)
    try:
        ax.set_facecolor("white")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white")
            pane.pane.set_edgecolor(ps.GRID)
            pane.pane.set_alpha(1.0)
        ax.grid(False)
    except Exception:
        pass
    ax.set_title("(a) pressure on the sharp 3D cut", pad=-2)

    # Slim horizontal colorbar tucked under the box, clear of panel (b).
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax.inset_axes([0.12, 0.02, 0.62, 0.030], transform=ax.transAxes)
    cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("pressure $p$", fontsize=8.2, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.6, length=2.0)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([1.0 - vmax, 1.0, 1.0 + vmax])
    cb.ax.set_xticklabels([f"{1.0 - vmax:.3f}", "1.000", f"{1.0 + vmax:.3f}"])


# ---------------------------------------------------------------------------
# (b) conservation ledger versus time, on the round-off floor.
# ---------------------------------------------------------------------------
def _panel_ledger(ax, result: dict) -> None:
    g2 = result["gate2_conservation_ke"]
    hist = g2["history"]
    t = np.array([r["t"] for r in hist])
    mass = np.abs([r["mass_closure"] for r in hist])
    energy = np.abs([r["energy_closure"] for r in hist])
    eps = 1.0e-19

    ax.semilogy(t, np.maximum(mass, eps), color=ps.PULLBACK, lw=1.7,
                label="mass budget", zorder=4)
    ax.semilogy(t, np.maximum(energy, eps), color=ps.SMALL_SUPPORT, lw=1.7,
                ls=(0, (4, 2)), label="total-energy budget", zorder=4)

    # Scalar end-of-run budgets pinned as round-off markers on the floor.
    bc = g2["budget_closure"]
    mom = max(abs(bc["mom_x"]), abs(bc["mom_y"]), abs(bc["mom_z"]))
    ke_rate = abs(g2["ke_rate_relative"])
    free = abs(g2["free_stream_residual"])
    tm = t[-1]
    ax.scatter([tm], [max(mom, eps)], marker="^", s=42, color=ps.LOCAL_GRAPH,
               zorder=6, label="momentum budget")
    ax.scatter([tm], [max(ke_rate, eps)], marker="D", s=34, color=ps.BOUNDARY,
               zorder=6, label="inviscid KE-rate")
    ax.scatter([tm], [max(free, eps)], marker="s", s=34, color="#8a6fbf",
               zorder=6, label="free-stream residual")

    ax.axhline(1.0e-12, color=ps.REFERENCE, ls=(0, (1, 2)), lw=1.0, zorder=1)
    ax.text(t[0], 1.55e-12, "round-off floor", color=ps.REFERENCE,
            fontsize=7.6, ha="left", va="bottom")

    ax.set_xlabel("acoustic transit time $t$")
    ax.set_ylabel(r"relative budget residual "
                  r"$|\Delta\,\mathrm{total}-\!\int\!\mathrm{flux}|/|\mathrm{total}_0|$")
    ax.set_title("(b) conservation ledger at round-off")
    ax.set_xlim(0, t[-1] * 1.02)
    ax.set_ylim(1.0e-19, 3.0e-11)


# ---------------------------------------------------------------------------
# (c) cut-independent time step: aggregation relief vs 1/eta collapse.
# ---------------------------------------------------------------------------
def _panel_relief(ax, result: dict) -> None:
    g4 = result["gate4_cut_independence"]
    rows = g4["rows"]
    vol = np.array([r["volume_fraction"] for r in rows])
    agg = np.array([r["aggregated_radius"] for r in rows])
    naive = np.array([r["naive_radius"] for r in rows])
    relief = g4["max_relief_ratio"]

    order = np.argsort(vol)
    vol, agg, naive = vol[order], agg[order], naive[order]

    # 1/eta guide anchored through the naive points, clipped to the data band so
    # it does not shoot off the top of the panel.
    guide = naive[0] * (vol[0] / vol)
    ax.loglog(vol, guide, color=ps.REFERENCE, ls=(0, (1, 2)), lw=1.1, zorder=1)
    ax.loglog(vol, naive, "o-", color=ps.NAIVE, lw=1.7, ms=6.5,
              label="unaggregated active control", zorder=3)
    ax.loglog(vol, agg, "s-", color=ps.PULLBACK, lw=2.0, ms=6.5,
              label="aggregated mimetic core", zorder=4)

    # Label the guide in the whitespace above the small-eta end of the curve.
    ax.text(vol[0] * 6.0, guide[0] * 0.30, r"$\propto 1/\eta$", color=ps.REFERENCE,
            fontsize=9.4, ha="left", va="center", rotation=30)

    imin = int(np.argmin(vol))
    ax.annotate(
        f"{relief:.0f}$\\times$ relief",
        xy=(vol[imin], naive[imin]),
        xytext=(vol[imin] * 6.0, naive[imin] * 0.02),
        color=ps.SMALL_SUPPORT, fontsize=9.6, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=ps.SMALL_SUPPORT, lw=1.1,
                        connectionstyle="arc3,rad=0.18"),
    )

    ax.set_xlabel(r"corner-sliver volume fraction $\eta$")
    ax.set_ylabel(r"explicit-step radius $\rho(H^{-1}Q_x)\,h$")
    ax.set_title("(c) cut-independent time step")
    ax.invert_xaxis()
    ax.set_ylim(0.1, 3.0e4)


def make_figure(result: dict | None = None) -> None:
    if result is None:
        path = RESULTS / "comp3d.json"
        result = json.loads(path.read_text())

    ps.apply_paper_style()
    fig = plt.figure(figsize=ps.WIDE_SIZE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.0], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _panel_field(ax_a)
    _panel_ledger(ax_b, result)
    _panel_relief(ax_c, result)

    for ax in (ax_b, ax_c):
        ps.style_axes(ax)

    # Shared legends: (b) and (c) each get a compact in-panel legend below the
    # title band, kept off the data with the round-off/collapse curves.
    ax_b.legend(loc="lower right", fontsize=7.7, frameon=False, ncol=1,
                handlelength=1.7, borderaxespad=0.4)
    ax_c.legend(loc="lower left", fontsize=8.2, frameon=False,
                handlelength=1.9, borderaxespad=0.4)

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.14, top=0.90)
    FIG.mkdir(parents=True, exist_ok=True)
    ps.save_pdf_png(fig, FIG / "fig_s6_comp3d.pdf", pdf_dpi=300)
    plt.close(fig)
    print("wrote", FIG / "fig_s6_comp3d.pdf", "and .png")


if __name__ == "__main__":
    make_figure()
