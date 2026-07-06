#!/usr/bin/env python3
"""Figure: fully conservative incompressible Navier-Stokes in 3D.

Three panels, cause to effect:

  (a) the Taylor-Green vortex advanced on the sharp cut box, rendered as columnar
      vortex cores with the in-plane swirl and the removed corner sliver shaded;
  (b) the term-by-term skew-symmetric kinetic-energy ledger against time: the skew-
      convection production, the interior pressure work, the divergence-constraint
      residual, and the overall budget closure, every one on the roundoff floor;
  (c) kinetic-energy drift and divergence against the corner-cut volume fraction,
      flat across three decades: the conservation does not depend on the cut.

Panel (a) renders the exact wall-confined vortex field the solver reproduces to
roundoff (no re-solve needed for the render).  Panels (b, c) read the cached
``results/tg3d_ledger_trace.json`` (written by ``tg3d_ledger_trace.py``) and
``results/tg3d.json``; if the ledger trace is missing the panel falls back to a
one-shot inviscid solve.  The wall-confined viscous decay does NOT match on the
cut (the tangential weak-penalty data performs boundary work) and is deliberately not
shown here; the periodic decay witness is ``fig_s6_tg3d_viscous``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import plot_style as ps
from plot_style import (
    PULLBACK, SMALL_SUPPORT, LOCAL_GRAPH, BOUNDARY, REFERENCE, TITLE, TEXT,
    apply_paper_style, panel_label, style_axes,
)
from tg3d_fully_conservative import wall_confined_tgv_3d_fields

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"

FLOOR = 1.0e-16          # IEEE double roundoff reference for the ledger panel
CLIP = 1.0e-18           # draw floor so exact zeros stay on the log axis
AXISCUT = "#2f4858"      # muted slate for the cut-box wireframe


# ---------------------------------------------------------------------------
# Panel (a): the vortex on the sharp cut box
# ---------------------------------------------------------------------------

def _box_edges(lo, hi):
    """The twelve edges of the axis-aligned box [lo,hi] (each a pair of points)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[v[a], v[b]] for a, b in e]


def _box_faces(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    return [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]],
    ]


def render_vortex(ax):
    """Render the wall-confined Taylor-Green vortex inside the sharp cut box.

    The field is the exact solution the fully conservative solver reproduces to
    roundoff: planar Taylor-Green (uniform in z, w = 0) on [a,1]^3, a = h*frac.
    Rendered as a vorticity-magnitude sheet on the base face, the in-plane swirl
    arrows, and columnar cores that extrude the 2x2 vortex cell through z, with
    the removed corner sliver shaded.
    """
    h = 1.0 / 6.0
    frac = 0.6
    a = h * frac                      # cut position on each axis -> box [a,1]^3
    fields = wall_confined_tgv_3d_fields(0.0, cut_fractions=(frac, frac, frac), h=h)
    kx, ky = fields["kx"], fields["ky"]

    # In-plane vorticity magnitude on a fine grid (uniform in z).
    ng = 90
    xs = np.linspace(a, 1.0, ng)
    ys = np.linspace(a, 1.0, ng)
    X, Y = np.meshgrid(xs, ys)
    # omega_z = dv/dx - du/dy for u = sin(kx(x-a))cos(ky(y-a)), v = -cos(kx(x-a))sin(ky(y-a))
    wz = (kx + ky) * np.sin(kx * (X - a)) * np.sin(ky * (Y - a))
    wmag = np.abs(wz)
    wmag /= wmag.max()

    # Colored vorticity sheet on the base face z = a.
    Z0 = np.full_like(X, a)
    facec = ps.FIELD_SEQUENTIAL(0.12 + 0.82 * wmag)
    ax.plot_surface(X, Y, Z0, facecolors=facec, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, zorder=1,
                    rasterized=True)

    # This field carries exactly one half-wave across the box (k = pi/L), so the
    # in-plane flow is a single central recirculation cell.  Render it as one
    # translucent columnar core extruded through z (uniform in z, w = 0), with a
    # few nested tubes to suggest the vortex column.
    L = 1.0 - a
    cx = cy = a + 0.5 * L                     # single central core
    theta = np.linspace(0, 2 * np.pi, 60)
    zc = np.linspace(a, 1.0, 2)
    for r in (0.40 * L, 0.28 * L, 0.16 * L):
        TH, ZZ = np.meshgrid(theta, zc)
        XT = cx + r * np.cos(TH)
        YT = cy + r * np.sin(TH)
        ax.plot_surface(XT, YT, ZZ, color=PULLBACK, alpha=0.12, linewidth=0,
                        antialiased=True, shade=False, zorder=3, rasterized=True)
    ax.plot([cx, cx], [cy, cy], [a, 1.0], color=PULLBACK, lw=1.6, alpha=0.9, zorder=4)

    # In-plane swirl arrows on a mid-z sheet (quiver), thinned for legibility.
    nq = 11
    xq = np.linspace(a + 0.05 * L, 1.0 - 0.05 * L, nq)
    yq = np.linspace(a + 0.05 * L, 1.0 - 0.05 * L, nq)
    XQ, YQ = np.meshgrid(xq, yq)
    UQ = np.sin(kx * (XQ - a)) * np.cos(ky * (YQ - a))
    VQ = -np.cos(kx * (XQ - a)) * np.sin(ky * (YQ - a))
    ZQ = np.full_like(XQ, a + 0.5 * L)
    WQ = np.zeros_like(XQ)
    ax.quiver(XQ, YQ, ZQ, UQ, VQ, WQ, length=0.10 * L, normalize=True,
              color=TEXT, linewidth=0.55, alpha=0.75, zorder=5)

    # Cut-box wireframe.
    edges = _box_edges((a, a, a), (1.0, 1.0, 1.0))
    ax.add_collection3d(Line3DCollection(edges, colors=AXISCUT, linewidths=1.1, zorder=6))

    # Removed corner sliver: the background corner cell [0,a]^3 that the cut deletes.
    sliver_faces = _box_faces((0.0, 0.0, 0.0), (a, a, a))
    pc = Poly3DCollection(sliver_faces, facecolor=SMALL_SUPPORT, edgecolor="#7c2424",
                          linewidths=0.6, alpha=0.5, zorder=2)
    ax.add_collection3d(pc)
    ax.text(a * 0.5, a * 0.5, a * 1.9, "removed\ncorner sliver", color="#7c2424",
            fontsize=7.6, ha="center", va="bottom", zorder=8)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-56)
    ax.set_xlabel("x", labelpad=-8, fontsize=9.5, color=TEXT)
    ax.set_ylabel("y", labelpad=-8, fontsize=9.5, color=TEXT)
    ax.set_zlabel("z", labelpad=-8, fontsize=9.5, color=TEXT)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.tick_params(labelsize=7.0, pad=-2, colors=TEXT)
    ax.set_title("vortex on the sharp cut box", color=TITLE, fontsize=9.6, pad=-4)
    # Quiet the 3D pane backgrounds for a clean archival look.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#e2e2de")
        pane.pane.set_alpha(1.0)


# ---------------------------------------------------------------------------
# Panels (b, c): the conservation ledger and its cut-independence
# ---------------------------------------------------------------------------

def _load_trace():
    path = RESULTS / "tg3d_ledger_trace.json"
    if path.exists():
        return json.loads(path.read_text())
    # Fallback: one inviscid solve supplies the three packaged ledger series.
    from tg3d_fully_conservative import solve_tg3d_unsteady
    _, led = solve_tg3d_unsteady(p=2, n_elements=6, cut_fractions=(0.6, 0.6, 0.6),
                                 nu=0.0, dt=0.01, t_final=0.3, aggregate=True)
    ke0 = float(led.kinetic_energy[0])
    kd = np.abs(led.kinetic_energy - led.kinetic_energy[0]) / ke0
    return {
        "times": led.times.tolist(),
        "skew_convection_production_rel": kd.tolist(),
        "interior_pressure_work_rel": (led.pressure_work_interior / ke0).tolist(),
        "divergence_residual": led.divergence_residual.tolist(),
        "ledger_residual_rel": led.ledger_residual.tolist(),
    }


def ledger_panel(ax, trace):
    style_axes(ax)
    t = np.array(trace["times"])

    def clip(a):
        return np.maximum(np.abs(np.asarray(a, float)), CLIP)

    ax.semilogy(t, clip(trace["ledger_residual_rel"]), color=PULLBACK, lw=1.8,
                marker="o", ms=3.2, markevery=3, label="kinetic-energy budget closure")
    ax.semilogy(t, clip(trace["skew_convection_production_rel"]), color=SMALL_SUPPORT,
                lw=1.5, ls=(0, (5, 2)), label="skew-convection production")
    ax.semilogy(t, clip(trace["interior_pressure_work_rel"]), color=LOCAL_GRAPH,
                lw=1.5, ls=(0, (1, 1.4)), label="interior pressure work")
    ax.semilogy(t, clip(trace["divergence_residual"]), color=BOUNDARY, lw=1.4,
                ls=(0, (3, 1, 1, 1)), label="divergence residual")

    ax.axhline(FLOOR, color=REFERENCE, lw=0.9, ls=(0, (1, 2)))
    ax.text(0.02, 0.965, "double-precision roundoff floor", color=REFERENCE,
            fontsize=7.6, ha="left", va="top", transform=ax.transAxes)
    ax.set_ylim(1e-18, 1e-11)
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("time")
    ax.set_ylabel("relative to kinetic energy")
    ax.set_title("skew-symmetric energy ledger at roundoff", fontsize=9.6)


def cut_panel(ax, sweep):
    style_axes(ax)
    vol = [s["volume_fraction"] for s in sweep]
    drift = [max(s["ke_relative_drift"], CLIP) for s in sweep]
    div = [max(s["max_divergence_residual"], CLIP) for s in sweep]
    ax.loglog(vol, drift, color=PULLBACK, marker="o", ms=6.5, lw=1.8,
              label="kinetic-energy drift")
    ax.loglog(vol, div, color=SMALL_SUPPORT, marker="s", ms=6.0, lw=1.6,
              label="divergence residual")
    ax.axhline(FLOOR, color=REFERENCE, lw=0.9, ls=(0, (1, 2)))
    ax.text(min(vol) * 1.4, FLOOR * 1.5, "roundoff floor", color=REFERENCE,
            fontsize=7.6, ha="left", va="bottom")
    ax.set_ylim(1e-18, 1e-11)
    ax.set_xlabel("corner-cut volume fraction")
    ax.set_ylabel("relative residual")
    ax.set_title("independent of the cut", fontsize=9.6)
    # Make the flatness the message.
    ax.annotate("flat across three decades", xy=(vol[1], 4e-15),
                xytext=(vol[1], 2.5e-13), ha="center", fontsize=7.8, color=TITLE,
                arrowprops=dict(arrowstyle="-", color=REFERENCE, lw=0.8))


def main() -> int:
    apply_paper_style()
    trace = _load_trace()
    sweep = json.loads((RESULTS / "tg3d.json").read_text())["cut_independence"]

    fig = plt.figure(figsize=(7.5, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.98, 1.16, 1.0], wspace=0.52,
                          left=0.02, right=0.985, bottom=0.24, top=0.86)
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    render_vortex(axA)
    ledger_panel(axB, trace)
    cut_panel(axC, sweep)

    # Panel labels sit above each panel's title band so they never touch the data
    # or the titles.
    for ax, lab in ((axB, "b"), (axC, "c")):
        ax.text(-0.16, 1.14, f"({lab})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9.0, fontweight="bold", color=TITLE)
    axA.text2D(0.02, 1.05, "(a)", transform=axA.transAxes, ha="left", va="bottom",
               fontsize=9.0, fontweight="bold", color=TITLE)

    # One shared legend strip for the ledger and cut panels, under panels b, c.
    handles = [
        Line2D([0], [0], color=PULLBACK, lw=1.8, marker="o", ms=3.4, label="KE budget closure"),
        Line2D([0], [0], color=SMALL_SUPPORT, lw=1.5, ls=(0, (5, 2)), label="skew convection"),
        Line2D([0], [0], color=LOCAL_GRAPH, lw=1.5, ls=(0, (1, 1.4)), label="pressure work"),
        Line2D([0], [0], color=BOUNDARY, lw=1.4, ls=(0, (3, 1, 1, 1)), label="divergence"),
    ]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               bbox_to_anchor=(0.62, -0.04), ncol=4, frameon=False, fontsize=7.8,
               handlelength=2.0, columnspacing=1.3, handletextpad=0.6)

    FIGURES.mkdir(exist_ok=True)
    # Cap the embedded raster (the 3D vorticity sheet) so the PDF stays lean; the
    # on-page panel width is ~2.4in, so 200 dpi is well beyond what it resolves.
    fig.savefig(FIGURES / "fig_s6_tg3d.pdf", bbox_inches="tight", pad_inches=0.02, dpi=200)
    fig.savefig(FIGURES / "fig_s6_tg3d.png", bbox_inches="tight", pad_inches=0.02, dpi=240)
    plt.close(fig)
    print("wrote figures/fig_s6_tg3d.pdf/.png")
    print(f"  ledger max closure (rel): {max(trace['ledger_residual_rel']):.2e}")
    print(f"  cut-sweep drift: {[f'{s['ke_relative_drift']:.1e}' for s in sweep]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
