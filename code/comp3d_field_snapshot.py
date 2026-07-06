"""Cache one evolved pressure/density snapshot on the sharp 3D cut box.

The Section-6 compressible figure ((a) of ``fig_s6_comp3d``) renders a real
solved field on the genuinely non-body-fitted cut cube: a small acoustic pulse is
launched off-centre in a unit cube whose lower corner sliver has been carved out,
evolved with the split skew-symmetric mimetic core for a fraction of an acoustic
transit time, and the resulting pressure/density stored at every Greville node.

Solving costs ~15 s at the cached resolution, so the snapshot is written once to
``results/comp3d_field.npz`` and reused by the figure driver.  Re-run only to
refresh the field:

    cd code && MPLBACKEND=Agg uv run python comp3d_field_snapshot.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from comp3d_solver import (
    CompressibleCut3D,
    acoustic_pulse_state,
    build_grid,
    set_state,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"

# One modest, visually legible configuration: a corner sliver carved out of the
# unit cube in all three axes, a smooth positive-pressure pulse launched from the
# cavity interior, evolved a short way so the reflected wavefronts read cleanly.
N_ELEMENTS = 12
DEGREE = 3
CUT_FRACTIONS = (0.7, 0.7, 0.7)
EPS = 3.5e-2
CENTER = (0.62, 0.62, 0.62)
SIGMA = 0.15
CFL = 0.22
N_STEPS = 26


def compute_snapshot() -> dict:
    g = build_grid(n_elements=N_ELEMENTS, degree=DEGREE, cut_fractions=CUT_FRACTIONS)
    S = CompressibleCut3D(g, mu=0.0)
    rho, u, v, w, p = acoustic_pulse_state(
        g.coords, 0.0, eps=EPS, center=CENTER, sigma=SIGMA
    )
    set_state(S, rho, u, v, w, p)
    dt = CFL * g.h / S.max_wavespeed()
    for _ in range(N_STEPS):
        S.step(dt)
    _, _, _, pf = S.primitives()
    lefts = tuple(g.h * f for f in g.cut_fractions)
    return {
        "coords": g.coords.astype(np.float64),
        "pressure": pf.astype(np.float64),
        "density": S.rho.astype(np.float64),
        "cut_fractions": np.asarray(g.cut_fractions, dtype=np.float64),
        "lefts": np.asarray(lefts, dtype=np.float64),
        "h": np.float64(g.h),
        "t": np.float64(N_STEPS * dt),
    }


def load_or_compute() -> dict:
    path = RESULTS / "comp3d_field.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    snap = compute_snapshot()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(path, **snap)
    return snap


def main() -> int:
    snap = compute_snapshot()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "comp3d_field.npz"
    np.savez(path, **snap)
    p = snap["pressure"]
    r = snap["density"]
    print(
        f"wrote {path}: n_nodes={snap['coords'].shape[0]} t={float(snap['t']):.3f} "
        f"p in [{p.min():.4f}, {p.max():.4f}] rho in [{r.min():.4f}, {r.max():.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
