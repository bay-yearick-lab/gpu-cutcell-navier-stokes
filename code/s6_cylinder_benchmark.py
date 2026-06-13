"""Section 6 driver: vortex shedding past a sharply cut cylinder (2D-2 benchmark).

Runs the 2D-2 cylinder benchmark through the sparse aggregated mixed
solver of ``channel_cylinder_flow`` (Re = 100, parabolic inflow, sharp
cylinder on a non-conforming tensor-product background grid) and renders the
manuscript exhibit: drag and lift histories against the published reference
intervals, plus a vorticity snapshot of the saturated vortex street with the
sharp cut drawn at true scale.

The full validation run advances 21,818 steps on the canonical 270 x 50
background grid (t = 12, dt = 5.5e-4) and is registered in the heavy tier;
``--figure-only`` rebuilds the figure from the stored artifacts.
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

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
FIGURES = ROOT / "figures"

PANEL_BG = "#ffffff"
BLUE = "#22517a"
TEAL = "#1f6f6f"
RED = "#a83232"
GREEN = "#1c7c34"

TAG = "p3_fine"
PRODUCTION_P = 3
PRODUCTION_NX = 270
PRODUCTION_NY = 50
PRODUCTION_DT = 5.5e-4
PRODUCTION_T_FINAL = 12.0


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def write_cylinder_figure(tag: str = TAG) -> None:
    from channel_cylinder_flow import (
        REFERENCE_INTERVALS,
        assemble_channel_operators,
        positive_channel_volume_quadrature,
        sparse_tensor_eval,
    )

    plt = _setup_matplotlib()
    suffix = f"_{tag}" if tag else ""
    series = json.loads((RESULTS / f"cylinder_benchmark{suffix}.json").read_text())
    summary = json.loads((RESULTS / f"cylinder_benchmark{suffix}_summary.json").read_text())
    state = np.load(RESULTS / f"cylinder_state{suffix}.npz")

    p = int(state["p"])
    nx = int(state["nx_elements"])
    ny = int(state["ny_elements"])
    ops = assemble_channel_operators(p=p, nx_elements=nx, ny_elements=ny)
    geo = ops.geo

    fig, (ax_g, ax_w) = plt.subplots(1, 2, figsize=(7.2, 2.95), constrained_layout=True)

    # (a) what the solver faces: background grid, sharp arc, and cut cells
    # tinted by volume fraction.  The inset is the worst sliver cell.
    qx, qy, qw, qcell = positive_channel_volume_quadrature(
        ops.basis_ux, ops.basis_uy, geo, 4
    )
    hx, hy = ops.hx, ops.hy
    cell_area = np.zeros(nx * ny)
    np.add.at(cell_area, qcell, qw)
    fractions = cell_area / (hx * hy)
    cut_ids = np.where((fractions > 1.0e-30) & (fractions < 1.0 - 1.0e-9))[0]
    sliver_id = int(cut_ids[np.argmin(fractions[cut_ids])])
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    norm = Normalize(vmin=-4.0, vmax=0.0)
    cmap = colormaps["Blues_r"]
    sliver_ix = sliver_id % nx
    sliver_iy = sliver_id // nx

    def draw_cut_window(ax, x0: float, x1: float, y0: float, y1: float, *, tick_labels: bool) -> None:
        ax.set_facecolor("white")
        for cid in cut_ids:
            ix, iy = int(cid % nx), int(cid // nx)
            cx0, cy0 = ix * hx, iy * hy
            if cx0 > x1 or cx0 + hx < x0 or cy0 > y1 or cy0 + hy < y0:
                continue
            level = max(np.log10(max(fractions[cid], 1.0e-12)), -4.0)
            ax.add_patch(
                Rectangle(
                    (cx0, cy0),
                    hx,
                    hy,
                    facecolor=cmap(norm(level)),
                    edgecolor="none",
                    zorder=1,
                )
            )
        for k in range(int(np.floor(x0 / hx)), int(np.ceil(x1 / hx)) + 1):
            ax.axvline(k * hx, color="0.78", lw=0.4, zorder=2)
        for k in range(int(np.floor(y0 / hy)), int(np.ceil(y1 / hy)) + 1):
            ax.axhline(k * hy, color="0.78", lw=0.4, zorder=2)
        disk = plt.Circle(
            (geo.cx, geo.cy),
            geo.radius,
            facecolor="white",
            edgecolor=RED,
            lw=1.3,
            zorder=3,
        )
        ax.add_patch(disk)
        # The worst sliver cell, outlined: its fluid part is a hairline strip
        # between the arc and the cell edge at the recorded volume ratio.
        ax.add_patch(
            Rectangle(
                (sliver_ix * hx, sliver_iy * hy),
                hx,
                hy,
                facecolor="none",
                edgecolor=TEAL,
                lw=1.6,
                zorder=4,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.tick_params(labelsize=7)
        if not tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    draw_cut_window(ax_g, 0.115, 0.295, 0.105, 0.295, tick_labels=True)
    ax_zoom = ax_g.inset_axes([0.53, 0.54, 0.42, 0.40])
    draw_cut_window(
        ax_zoom,
        (sliver_ix - 1.10) * hx,
        (sliver_ix + 2.10) * hx,
        (sliver_iy - 1.10) * hy,
        (sliver_iy + 2.10) * hy,
        tick_labels=False,
    )
    ax_zoom.set_xlabel("")
    ax_zoom.set_ylabel("")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_g, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\log_{10}$ cut-cell volume fraction", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # (b) vorticity snapshot from the stored final state.  Benchmark force
    # quantities are reported in the adjacent table against reference bands.
    au = ops.E_u @ state["cu"]
    av = ops.E_u @ state["cv"]
    gx = np.linspace(0.0, geo.length, 661)
    gy = np.linspace(0.0, geo.height, 124)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    _phi, dphix, dphiy = sparse_tensor_eval(ops.basis_ux, ops.basis_uy, X.ravel(), Y.ravel())
    omega = (dphix @ av - dphiy @ au).reshape(X.shape)
    mask = (X - geo.cx) ** 2 + (Y - geo.cy) ** 2 <= geo.radius**2
    omega = np.ma.masked_where(mask, omega)
    span = 60.0
    im = ax_w.pcolormesh(
        X,
        Y,
        np.clip(omega, -span, span),
        cmap="RdBu_r",
        vmin=-span,
        vmax=span,
        shading="auto",
        rasterized=True,
    )
    circle = plt.Circle((geo.cx, geo.cy), geo.radius, facecolor="white", edgecolor="0.2", lw=0.8)
    ax_w.add_patch(circle)
    ax_w.set_aspect("equal")
    ax_w.set_xlim(0.0, geo.length)
    ax_w.set_ylim(0.0, geo.height)
    ax_w.set_xlabel("$x$")
    ax_w.set_ylabel("$y$")
    cbar2 = fig.colorbar(im, ax=ax_w, fraction=0.025, pad=0.015)
    cbar2.set_label(r"vorticity $\omega$", fontsize=8)
    cbar2.ax.tick_params(labelsize=7.5)

    for label, ax in zip(("a", "b"), (ax_g, ax_w)):
        ax.text(
            0.01,
            1.02,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#202020",
            clip_on=False,
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_s6_cylinder_shedding.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(FIGURES / "fig_s6_cylinder_shedding.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[s6_cylinder_benchmark] figure written; summary:")
    summary.setdefault("cl_max_abs", max(abs(summary["cl_max"]), abs(summary["cl_min"])))
    print(json.dumps({k: summary[k] for k in (
        "strouhal", "cd_max", "cl_max_abs", "delta_p_at_clmax",
        "min_cell_volume_fraction", "max_divergence_residual", "n_periods_measured")}, indent=2))


def write_cylinder_table(tag: str = TAG) -> str:
    from channel_cylinder_flow import REFERENCE_INTERVALS

    suffix = f"_{tag}" if tag else ""
    summary = json.loads((RESULTS / f"cylinder_benchmark{suffix}_summary.json").read_text())
    summary.setdefault("cl_max_abs", max(abs(summary["cl_max"]), abs(summary["cl_min"])))
    rows = [
        ("Strouhal number $St$", "strouhal", REFERENCE_INTERVALS["strouhal"]),
        ("maximum drag coefficient $c_{D,\\max}$", "cd_max", REFERENCE_INTERVALS["cd_max"]),
        ("maximum lift amplitude $|c_L|_{\\max}$", "cl_max_abs", REFERENCE_INTERVALS["cl_max"]),
        (
            "pressure difference $\\Delta p(t_{c_{L,\\max}})$",
            "delta_p_at_clmax",
            REFERENCE_INTERVALS["delta_p"],
        ),
    ]
    lines = [
        "% Auto-generated by code/s6_cylinder_benchmark.py (write_cylinder_table).",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"quantity & computed & reference interval & criterion \\",
        r"\midrule",
    ]
    for label, key, (lo, hi) in rows:
        value = float(summary[key])
        if lo <= value <= hi:
            criterion = r"\textsc{inside}"
        else:
            edge = hi if value > hi else lo
            rel = abs(value - edge) / max(abs(edge), 1.0e-300)
            direction = "+" if value > hi else "-"
            criterion = rf"\textsc{{outside}} {direction}{100.0 * rel:.3f}\%"
        lines.append(
            f"{label} & ${value:.4f}$ & $[{lo:.3f},\\,{hi:.3f}]$ & {criterion}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    out = ROOT / "paper" / "generated_cylinder_benchmark_table.tex"
    out.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-only", action="store_true", help="reuse stored artifacts")
    parser.add_argument("--tag", type=str, default=TAG)
    parser.add_argument("--p", type=int, default=PRODUCTION_P)
    parser.add_argument("--nx", type=int, default=PRODUCTION_NX)
    parser.add_argument("--ny", type=int, default=PRODUCTION_NY)
    parser.add_argument("--dt", type=float, default=PRODUCTION_DT)
    parser.add_argument("--t-final", type=float, default=PRODUCTION_T_FINAL)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    args = parser.parse_args()
    if not args.figure_only:
        from channel_cylinder_flow import run_cylinder_benchmark, write_cylinder_artifacts

        suffix = f"_{args.tag}" if args.tag else ""
        RESULTS.mkdir(parents=True, exist_ok=True)
        result = run_cylinder_benchmark(
            p=args.p,
            nx_elements=args.nx,
            ny_elements=args.ny,
            dt=args.dt,
            t_final=args.t_final,
            record_every=args.record_every,
            state_path=RESULTS / f"cylinder_state{suffix}.npz",
            checkpoint_every=args.checkpoint_every,
        )
        write_cylinder_artifacts(result, tag=args.tag)
    write_cylinder_figure(tag=args.tag)
    write_cylinder_table(tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
