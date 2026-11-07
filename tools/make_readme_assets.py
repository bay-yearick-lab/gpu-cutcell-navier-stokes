"""Regenerate the README figures in ``assets/`` from the committed result files.

Usage (from the repository root):

    uv run python tools/make_readme_assets.py

The result cards read ``code/results/``.  The hero banner re-evaluates the
vorticity of the final cylinder state (``code/results/cylinder_state_p3_fine.npz``)
at 1800 x 335 points with the solver's own operators; the evaluated field is
cached in ``$DATA_DIR/vort.npz`` (default ``tools/_cache``).
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "code" / "results"
OUT = Path(os.environ.get("ASSETS_DIR", ROOT / "assets"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "tools" / "_cache"))
plt.style.use(ROOT / "tools" / "lab.mplstyle")
INK, TEAL, AMBER, SLATE, CORAL, GRAY = "#1B2A41", "#0F9D8A", "#F2A541", "#3D5A80", "#E07A5F", "#8D99AE"

def hero_vorticity() -> Path:
    """Evaluate (once) the vorticity of the final cylinder state on a fine grid."""
    path = DATA_DIR / "vort.npz"
    if path.exists():
        return path
    sys.path.insert(0, str(ROOT / "code"))
    from channel_cylinder_flow import assemble_channel_operators, sparse_tensor_eval

    st = np.load(R / "cylinder_state_p3_fine.npz")
    ops = assemble_channel_operators(
        p=int(st["p"]), nx_elements=int(st["nx_elements"]), ny_elements=int(st["ny_elements"])
    )
    geo = ops.geo
    au, av = ops.E_u @ st["cu"], ops.E_u @ st["cv"]
    gx = np.linspace(0, geo.length, 1800)
    gy = np.linspace(0, geo.height, int(1800 * geo.height / geo.length))
    X, Y = np.meshgrid(gx, gy)
    xs, ys = X.ravel(), Y.ravel()
    w = np.empty(X.size)
    for i in range(0, X.size, 100000):
        _, dx, dy = sparse_tensor_eval(ops.basis_ux, ops.basis_uy, xs[i:i + 100000], ys[i:i + 100000])
        w[i:i + 100000] = dx @ av - dy @ au
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, Y=Y, w=w.reshape(X.shape), cx=geo.cx, cy=geo.cy, r=geo.radius,
             L=geo.length, H=geo.height)
    return path


# ---------------------------------------------------------------- GPU card
rows = list(csv.DictReader(open(ROOT / "benchmarks" / "results" / "gpu_solver_benchmark_3d.csv")))
n = np.array([int(r["n_unknowns"]) for r in rows])
nnz = np.array([int(r["nnz"]) for r in rows])
tc = np.array([float(r["cpu_cg_s"]) for r in rows])
tg = np.array([float(r["gpu_cg_s"]) for r in rows])
tf = np.array([float(r["cpu_splu_factor_s"]) for r in rows])
ic = [int(r["cpu_cg_iterations"]) for r in rows]
ig = [int(r["gpu_cg_iterations"]) for r in rows]
sp = np.array([float(r["gpu_speedup_vs_cpu_cg"]) for r in rows])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
a1.loglog(n, tf, "s--", color=GRAY, label="CPU sparse LU (factor only)")
a1.loglog(n, tc, "o-", color=AMBER, lw=2.2, label="CPU Jacobi-PCG (SciPy)")
a1.loglog(n, tg, "o-", color=TEAL, lw=2.6, label="GPU Jacobi-PCG (CuPy CSR)")
a1.set_xlabel("unknowns (3D aggregated cut system, p = 2)")
a1.set_ylabel("solve time [s]")
a1.set_title("Wall-clock per solve", fontsize=12)
a1.legend(loc="upper left", fontsize=9)
a1.set_xticks([1e4, 2e4, 4e4, 8e4], ["10k", "20k", "40k", "80k"]); a1.minorticks_off()
a1.annotate(f"{tf[-1]:.0f} s", (n[-1], tf[-1]), textcoords="offset points", xytext=(-34, -4), fontsize=9, color=GRAY)
x = np.arange(len(n))
b = a2.bar(x, sp, color=TEAL, width=0.7)
for i in range(len(n)):
    a2.text(i, sp[i] + 0.8, f"{sp[i]:.1f}x", ha="center", fontsize=11, fontweight="bold", color=INK)
    a2.text(i, 1.2, f"{ic[i]}\n{ig[i]}", ha="center", fontsize=8.5, color="white", linespacing=1.1)
a2.set_xticks(x, [f"{v/1000:.1f}k" for v in n], fontsize=9.5); a2.set_xlabel("unknowns")
a2.set_ylabel("speedup, GPU vs CPU PCG")
a2.set_ylim(0, 45)
a2.set_title("Speedup (in-bar: CPU / GPU iterations)", fontsize=11.5)
a2.grid(axis="x", visible=False)
fig.suptitle("Same preconditioned CG on the GPU: 9x to 39x faster, matching iterations", fontsize=15, fontweight="bold", color=INK, x=0.02, ha="left")
fig.text(0.02, -0.04, "Screened Poisson A = K + H on an aggregated 3D cut geometry, p = 2, float64, CuPy CSR on the GPU vs SciPy on the CPU.", fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "card_gpu_speedup.png")
plt.close(fig)

# ---------------------------------------------------------------- conditioning card
d = json.load(open(R / "s6_oblique_sweep_summary.json"))
rws = d["rows"]
fig, ax = plt.subplots(figsize=(8, 3.8))
for k, p in enumerate((3, 4, 5)):
    sel = [r for r in rws if r["p"] == p]
    jit = (np.arange(len(sel)) / len(sel) - 0.5) * 0.5
    ax.scatter(k + jit - 0.0, [r["cond_H_active"] for r in sel], s=16, color=AMBER, alpha=0.85, label="unaggregated active space" if k == 0 else None)
    ax.scatter(k + jit, [r["cond_H_phys"] for r in sel], s=16, color=TEAL, label="aggregated (visible coordinates)" if k == 0 else None)
    pp = d["per_p"][str(p)]
    ax.text(k, pp["max_cond_H_phys"] / 12, f"spread {pp['cond_H_phys_ratio_max_over_min']:.2f}x", ha="center", fontsize=9, color=TEAL, fontweight="bold")
    ax.text(k, pp["max_cond_H_active"] * 3, f"max {pp['max_cond_H_active']:.1e}", ha="center", fontsize=9, color="#9A6A1E")
ax.set_yscale("log")
ax.set_ylim(1e2, 1e13)
ax.set_xticks([0, 1, 2], ["p = 3", "p = 4", "p = 5"])
ax.set_xlim(-0.6, 2.6)
ax.set_ylabel("mass-matrix condition number")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Aggregation keeps conditioning within 3.70x per degree over 72 cuts", loc="left", fontsize=13.5)
fig.text(0.01, -0.03, "8 cut angles x 3 offsets x 3 degrees, 6 elements per side.", fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "card_conditioning.png")
plt.close(fig)

# ---------------------------------------------------------------- order ladder card
L = json.load(open(R / "highp_pde_solve_ladder.json"))["degrees"]
fig, ax = plt.subplots(figsize=(8, 4))
cols = {"4": SLATE, "6": TEAL, "8": CORAL}
for p, v in L.items():
    w = v["well"]
    h = 1.0 / np.array(v["meshes"], float)
    e = np.array(w["errors"])
    ax.loglog(h, e, "o-", color=cols[p], lw=2.2, label=f"p = {p}: observed {w['observed_l2_order']:.2f}, formal {v['expected_order']}")
    q = v["expected_order"]
    ref = e[-1] * (h / h[-1]) ** q
    ax.loglog(h, ref, ":", color=cols[p], lw=1.2)
ax.set_xticks([0.2, 0.1, 0.06, 0.04, 0.03], ["0.2", "0.1", "0.06", "0.04", "0.03"]); ax.minorticks_off()
ax.invert_xaxis()
ax.set_xlabel("mesh size h = 1/n")
ax.set_ylabel("relative L2 error")
ax.legend(fontsize=9, loc="lower left")
ax.set_title("Order p + 1 on a sharp curved cut, degrees 4 to 8", loc="left", fontsize=14)
fig.text(0.01, -0.03, "Poisson solve, weak boundary imposition, float64, on a self-similar circular corner cut; dotted lines show slope p + 1.", fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "card_order_ladder.png")
plt.close(fig)

# ---------------------------------------------------------------- stress card
S = list(csv.DictReader(open(R / "fast_solver_stress.csv")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
ne_vals = sorted({int(r["n_elements"]) for r in S})
shades = ["#B7E4DC", "#7CCFC2", "#3FB5A4", "#0F9D8A", "#0A6F62"]
for c, ne in zip(shades, ne_vals):
    sel = [r for r in S if int(r["n_elements"]) == ne]
    vr = [float(r["volume_ratio"]) for r in sel]
    a1.semilogx(vr, [int(r["iterations"]) for r in sel], "o-", color=c, lw=2, label=f"{ne} x {ne}")
    a2.loglog(vr, [float(r["relative_residual"]) for r in sel], "o", color=c, ms=6)
a1.set_xlabel("cut-cell volume ratio")
a1.set_ylabel("PCG iterations")
a1.set_title("Iterations set by mesh, not by sliver", fontsize=11.5)
a1.legend(fontsize=8.5, ncol=3, loc="lower center", title="elements", title_fontsize=8.5)
a1.set_ylim(40, 200)
a2.axhline(1e-8, color=CORAL, ls="--", lw=1.2)
a2.text(12, 1.15e-8, "tolerance 1e-8", color=CORAL, fontsize=9)
a2.set_ylim(3e-9, 2e-8)
a2.set_xlabel("cut-cell volume ratio")
a2.set_ylabel("final relative residual")
a2.set_title("All residuals below tolerance", fontsize=11.5)
fig.suptitle("25/25 stress cases converge, volume ratios 10 to 1e5", fontsize=15, fontweight="bold", color=INK, x=0.02, ha="left")
fig.text(0.02, -0.04, "2D oblique cut, aggregated p = 4 operator, Jacobi-PCG; max residual 9.9e-9, iterations 104 to 188.", fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "card_stress.png")
plt.close(fig)

# ---------------------------------------------------------------- conservation card
tg = json.load(open(R / "tg3d.json"))["inviscid"]
c3 = json.load(open(R / "comp3d.json"))["gate2_conservation_ke"]
cyl = json.load(open(R / "cylinder_benchmark_p3_fine_summary.json"))
bc = c3["budget_closure"]
items = [
    ("compressible 3D: mass budget", bc["mass"], SLATE),
    ("compressible 3D: momentum budget (max)", max(bc["mom_x"], bc["mom_y"], bc["mom_z"]), SLATE),
    ("compressible 3D: total-energy budget", bc["energy"], SLATE),
    ("compressible 3D: inviscid KE production", c3["ke_rate_relative"], SLATE),
    ("incompressible 3D: KE ledger closure", tg["max_ledger_residual_rel"], TEAL),
    ("incompressible 3D: interior pressure work", tg["max_pressure_work_interior"], TEAL),
    ("incompressible 3D: divergence", tg["max_divergence_residual"], TEAL),
    (f"cylinder, {1/cyl['min_cell_volume_fraction']:.0f}:1 sliver: divergence", cyl["max_divergence_residual"], TEAL),
]
fig, ax = plt.subplots(figsize=(8, 3.9))
y = np.arange(len(items))[::-1]
ax.barh(y, [v for _, v, _ in items], color=[c for *_, c in items], height=0.6, left=1e-19)
for yi, (lab, v, _) in zip(y, items):
    ax.text(v * 1.6, yi, f"{v:.1e}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [l for l, *_ in items], fontsize=9.5)
ax.set_xscale("log")
ax.set_xlim(1e-19, 1e-12)
ax.axvline(2.22e-16, color=CORAL, ls="--", lw=1.2)
ax.text(2.6e-16, y[0] + 0.55, "double-precision eps", color=CORAL, fontsize=8.5)
ax.set_xlabel("relative residual")
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
fig.suptitle("Discrete conservation holds to round-off on sharp cuts", fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left")
fig.text(0.01, -0.03, "3D rows: cut box with a corner sliver cell. Cylinder row: Re = 100 shedding run with a 1906:1 cut cell.", fontsize=8, color=GRAY)
fig.tight_layout()
fig.savefig(OUT / "card_conservation.png")
plt.close(fig)

# ---------------------------------------------------------------- hero
V = np.load(hero_vorticity())
X, Y, w = V["X"], V["Y"], V["w"]
cx, cy, r = float(V["cx"]), float(V["cy"]), float(V["r"])
mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
w = np.ma.masked_where(mask, np.clip(w, -60, 60))
cmap = LinearSegmentedColormap.from_list("lab", ["#0B3C5D", "#3D5A80", "#9FB7D0", "#F4F6F8", "#F2C08A", "#E07A5F", "#9E2A2B"])
plt.rcParams["savefig.bbox"] = "standard"
W, H = 16.0, 5.6
fig = plt.figure(figsize=(W, H), dpi=100, facecolor=INK)
strip_h = 2.98 / H
ax = fig.add_axes([0, 0, 1, strip_h])
ax.pcolormesh(X, Y, w, cmap=cmap, vmin=-60, vmax=60, shading="auto", rasterized=True)
ax.add_patch(plt.Circle((cx, cy), r, facecolor="#E9ECF1", edgecolor=INK, lw=1.0))
ax.set_xlim(0, float(V["L"])); ax.set_ylim(0, float(V["H"])); ax.set_aspect("auto"); ax.axis("off")
ax.text(0.995, 0.04, "B-spline cut-cell solver, Re = 100 vortex street, cylinder cut sharply on a 270 x 50 grid (worst cell 1906:1)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
fig.text(0.036, 0.875, "gpu-cutcell-navier-stokes", fontsize=30, color="white", fontweight="bold", family="DejaVu Sans Mono", va="center")
fig.text(0.036, 0.765, "High-order B-spline cut-cell CFD on the GPU", fontsize=17, color="#C9D3E0", va="center")
fig.text(0.036, 0.635, "Yong Yi Bay  &  Kathleen A. Yearick", fontsize=15, color=GRAY, va="center")
fig.text(0.965, 0.865, "0 solver failures on sliver cut cells", fontsize=23, color=TEAL, fontweight="bold", ha="right", va="center")
fig.text(0.965, 0.755, "Structure-preserving aggregation holds conditioning within 3.7x per degree", fontsize=13.5, color="white", ha="right", va="center")
fig.text(0.965, 0.705, "across 72 cut geometries (1.8e11 without it); Jacobi-preconditioned CG", fontsize=13.5, color="white", ha="right", va="center")
fig.text(0.965, 0.655, "converges in 25/25 stress cases, cut cells down to 1e-5 of a full cell", fontsize=13.5, color="white", ha="right", va="center")
fig.savefig(OUT / "hero.png", dpi=100, facecolor=INK, bbox_inches=None)
plt.close(fig)
print("ok")
