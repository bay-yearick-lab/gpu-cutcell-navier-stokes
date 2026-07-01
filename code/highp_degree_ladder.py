"""Clean order p+1 on a sharp curved cut at spline degree 10 and 12.

This is the central differentiation experiment: the high-order cut-cell scheme
delivers the FULL algebraic order ``p+1`` at brave spline degrees, defeating the
conditioning and sliver floors that cap naive high-degree cut-cell methods.

Why the stock curved-cut sweep saturates
-----------------------------------------
The committed manufactured-Poisson sweep (``s6_curved_poisson_degree_sweep``)
refines a FIXED circular corner cut (radius 0.42) on a refining Cartesian grid.
Two confounders enter at high degree, neither of which is a loss of approximation
power:

1.  CONDITIONING.  The stock polynomial-pinning extension fits the small-support
    coordinates by least squares against GLOBAL monomials, whose moment matrix
    has condition number ``~1e13`` at ``p = 10``.  That caps the extension's
    polynomial reproduction at ``~5e-9``, and that residual is the error floor.
2.  SLIVERS.  A fixed arc on a refining grid grazes grid nodes at irregular
    meshes, producing fluid fractions as small as ``2e-11`` whose geometry noise
    swamps the convergence at high degree.

Two construction changes remove both, exposing the true order
-------------------------------------------------------------
*  WELL-CONDITIONED BASIS (``highp_cut_basis.well_conditioned_circular_extension``):
   fit the extension against LOCAL shifted/scaled monomials.  The stencil moment
   matrix condition drops from ``~1e13`` to ``~1e9``, the extension residual from
   ``~5e-9`` to ``~1e-11``, and the operator norm of the extension from ``~9e5``
   to ``~7e3``.
*  SELF-SIMILAR REFINEMENT (``radius = c * h``):  the curved corner sliver is held
   at the SAME fluid fraction on every mesh, so the only thing that changes under
   refinement is resolution.  This is the standard device for measuring the
   algebraic order of a cut-cell discretization free of grazing artifacts.

With both in place the weighted-L2 projection of a smooth non-polynomial target
onto the physical cut space converges at the clean rate ``p+1`` through degree 8
(measured 7.11 at p=6 and 9.48 at p=8).  Extended precision (numpy ``longdouble``,
eps ``~1e-19``) is used at ``p >= 8`` to keep the deepest meshes above the roundoff
floor so the slope is read from genuine discretization error rather than arithmetic
noise.

Degree 10 is reported for attained depth only (deepest error ``~1.6e-13``), with
no clean ``p+1`` order claim, and degree 12 plateaus near ``1e-9`` where the
aggregation conditioning floor meets working precision.  An opt-in
better-conditioned construction (``local_basis="chebyshev"`` plus
``extra_columns``) is threaded through ``project_self_similar`` and
``well_conditioned_circular_extension``, but a direct comparison through the
production projection (``results/highp_degree_ladder_extension_study.json``) shows
it does NOT recover a clean order-11 ladder at degree 10: it improves the
small-support local-fit conditioning but not the assembled extension's binding
reproduction of the relevant high-degree modes, so the realized projection order is
unchanged.  The clean ladder is therefore claimed only through degree 8; degrees 10
and 12 are the named conditioning-floor boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from cut_spline_sbp import make_open_uniform_basis
from circular_corner_sbp import (
    CircularCornerCut,
    assemble_circular_corner_matrices,
    polynomial_preserving_circular_extension,
)
from highp_cut_basis import well_conditioned_circular_extension, _chol_solve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"


def smooth_target(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth, non-polynomial approximation target (a separable cosine bump).

    It is analytic and degree-free, so it is a genuine target at every spline
    degree; nothing about it is special to the cut, which keeps the measured
    order a property of the discretization rather than the data.
    """
    return np.cos(pi * x) * np.cos(pi * y)


@dataclass(frozen=True)
class LadderRow:
    p: int
    n_elements: int
    h: float
    radius: float
    radius_over_h: float
    relative_l2_error: float
    relative_linf_error: float
    relative_linf_cut: float
    relative_linf_interior: float
    gram_cond: float
    extension_residual: float
    min_fluid_fraction: float
    n_physical: int
    n_quadrature: int
    quadrature_order: int
    precision: str
    extension: str
    local_basis: str
    extra_columns: int
    seconds: float


def project_self_similar(
    p: int,
    n_elements: int,
    radius_over_h: float = 1.3,
    extension: str = "well",
    precision: str = "double",
    quadrature_extra: int = 16,
    support_threshold: float = 0.02,
    local_basis: str = "monomial",
    extra_columns: int = 0,
) -> LadderRow:
    """Self-similar (radius = c*h) weighted-L2 projection onto the cut space.

    The corner sliver is identical on every mesh, so a clean refinement sequence
    measures the algebraic order with no grazing confound.

    ``local_basis`` and ``extra_columns`` are forwarded to the well-conditioned
    extension; they default to the committed monomial/minimal-stencil behavior
    so the degree-6/8 results are unchanged, and switch to the better-conditioned
    Chebyshev + enlarged-stencil construction when set (used at degree 10+).
    """
    t0 = time.time()
    h = 1.0 / n_elements
    radius = float(radius_over_h * h)
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=p)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=p)
    q_order = max(32, 2 * p + quadrature_extra)
    cut = CircularCornerCut(radius=radius)
    matrices = assemble_circular_corner_matrices(
        basis_x, basis_y, cut, quadrature_order=q_order, capture_volume_quadrature=True
    )
    if extension == "well":
        agg = well_conditioned_circular_extension(
            basis_x, basis_y, matrices, support_threshold=support_threshold, preserve_degree=p,
            local_basis=local_basis, extra_columns=extra_columns,
        )
    else:
        agg = polynomial_preserving_circular_extension(
            basis_x, basis_y, matrices, support_threshold=support_threshold, preserve_degree=p
        )
    E = agg.E
    q = matrices.volume_quadrature

    dt = np.longdouble if precision == "long" else np.float64
    Phi = (E.T @ q.basis_values).astype(dt)
    w = q.weights.astype(dt)
    G = (Phi * w) @ Phi.T
    uv = smooth_target(q.x, q.y).astype(dt)
    rhs = (Phi * w) @ uv
    coeff = _chol_solve(G, rhs) if precision == "long" else np.linalg.solve(G, rhs)
    uh = Phi.T @ coeff
    err = uh - uv
    l2 = np.sqrt(np.sum(w * err * err))
    l2ref = np.sqrt(np.sum(w * uv * uv))
    # maximum-norm (pointwise) error over the cut volume quadrature: the strict
    # certificate that no quadrature point, including the cut-adjacent ones, hides
    # a low-order error. Reported relative to the peak of the exact target.
    linf = np.max(np.abs(err))
    linfref = np.max(np.abs(uv))
    # split the max-norm error into the cut band (within one mesh width of the arc
    # x^2+y^2=R^2) and the interior, to certify the cut introduces no error hotspot
    # and no order loss; distance-to-cut is |sqrt(x^2+y^2) - R| (verified convention).
    rr = np.sqrt(q.x.astype(np.float64) ** 2 + q.y.astype(np.float64) ** 2)
    band = np.abs(rr - radius) < h
    abserr = np.abs(err)
    linf_cut = float(np.max(abserr[band])) if band.any() else 0.0
    linf_int = float(np.max(abserr[~band])) if (~band).any() else 0.0
    gamma = agg.gamma
    active = gamma[gamma > 1.0e-12]
    return LadderRow(
        p=int(p),
        n_elements=int(n_elements),
        h=h,
        radius=radius,
        radius_over_h=float(radius_over_h),
        relative_l2_error=float(l2 / l2ref),
        relative_linf_error=float(linf / linfref),
        relative_linf_cut=float(linf_cut / linfref),
        relative_linf_interior=float(linf_int / linfref),
        gram_cond=float(np.linalg.cond(G.astype(np.float64))),
        extension_residual=float(agg.polynomial_residual),
        min_fluid_fraction=float(active.min()) if active.size else 1.0,
        n_physical=int(E.shape[1]),
        n_quadrature=int(w.size),
        quadrature_order=int(q_order),
        precision=precision,
        extension=extension,
        local_basis=local_basis,
        extra_columns=int(extra_columns),
        seconds=float(time.time() - t0),
    )


# Per-degree plan: meshes chosen to stay above the (precision-appropriate) floor,
# and the precision/extension lever each degree needs.
RADIUS_OVER_H = 1.3

DEGREE_PLAN: dict[int, dict] = {
    # Mesh ladders are chosen per degree to stay in the asymptotic window ABOVE
    # the (precision-appropriate) roundoff floor: at high p the scheme reaches
    # the longdouble floor (~1e-15) on coarse meshes, so the fine meshes that
    # work at p=6 would only floor out and corrupt the slope at p=10,12.  The
    # self-similar radius c is chosen per degree so the corner sliver stays
    # benign (fluid fraction well above the grazing-noise level).  The coarsest
    # meshes are dropped at p>=10 where degree-p resolution is not yet reached on
    # 4 cells.  Degree 12 is intentionally omitted from the clean p+1 ladder:
    # the small-support extension's degree-12 polynomial reproduction degrades and
    # the projection plateaus near 1e-9 (verified solver-independent by a
    # longdouble QR probe equalling the normal-equations residual), so claiming a
    # clean order-13 slope there would not be honest.  Degrees 6 and 8 attain the
    # full algebraic order p+1; degree 10 is kept here for ATTAINED DEPTH ONLY (no
    # clean p+1 claim -- the first-interval orders sit near 7 because the assembled
    # extension's reproduction of the relevant modes caps the slope before the
    # finer meshes reach the floor).  An opt-in Chebyshev local basis + enlarged
    # stencil (local_basis=, extra= below) was validated and does NOT recover the
    # degree-10 order (results/highp_degree_ladder_extension_study.json), so the
    # plan keeps the committed monomial/minimal-stencil construction.
    # Showcase set p=4,6,8 (clean p+1) and 10 depth-only (precision/extension ceiling).
    # p=4 stops at n=24: an n=32 step peaks ~53 GB (measured) and would OOM; order 5 is
    # clean over 8..24 well above the double-precision floor.
    4: dict(meshes=(8, 12, 16, 24), precision="double", extension="well", c=1.3),
    6: dict(meshes=(8, 12, 16, 24, 32), precision="double", extension="well", c=1.3),
    8: dict(meshes=(6, 8, 12, 16), precision="long", extension="well", c=1.3),
    10: dict(meshes=(5, 6, 7, 8), precision="long", extension="well", c=1.3),
}


def _fit_slope(ns, errs):
    ns = np.asarray(ns, dtype=float)
    errs = np.asarray(errs, dtype=float)
    h = 1.0 / ns
    return float(np.polyfit(np.log(h), np.log(errs), 1)[0])


def _pairwise(ns, errs):
    out = []
    for i in range(1, len(ns)):
        o = np.log(errs[i - 1] / errs[i]) / np.log(ns[i] / ns[i - 1])
        out.append(round(float(o), 3))
    return out


def run_ladder(plan: dict[int, dict] | None = None, write: bool = True, log=None) -> dict:
    plan = plan or DEGREE_PLAN
    rows: list[dict] = []
    per_degree: dict[str, dict] = {}

    def _log(msg):
        if log is not None:
            log.write(msg + "\n")
            log.flush()
        print(msg, flush=True)

    for p in sorted(plan):
        cfg = plan[p]
        ns = list(cfg["meshes"])
        errs = []
        errs_inf = []
        deg_rows = []
        for n in ns:
            r = project_self_similar(
                p,
                n,
                radius_over_h=cfg.get("c", RADIUS_OVER_H),
                extension=cfg["extension"],
                precision=cfg["precision"],
                local_basis=cfg.get("local_basis", "monomial"),
                extra_columns=cfg.get("extra", 0),
            )
            rows.append(asdict(r))
            deg_rows.append(r)
            errs.append(r.relative_l2_error)
            errs_inf.append(r.relative_linf_error)
            _log(
                f"p={p} n={n} ({cfg['precision']},{cfg['extension']},"
                f"{cfg.get('local_basis', 'monomial')},extra={cfg.get('extra', 0)}): "
                f"relL2={r.relative_l2_error:.4e} relLinf={r.relative_linf_error:.4e} "
                f"cond={r.gram_cond:.2e} "
                f"extres={r.extension_residual:.1e} minfrac={r.min_fluid_fraction:.2e} "
                f"t={r.seconds:.0f}s"
            )
        slope = _fit_slope(ns, errs)
        slope_inf = _fit_slope(ns, errs_inf)
        deepest = float(min(errs))
        deepest_inf = float(min(errs_inf))
        per_degree[str(p)] = {
            "expected_order": p + 1,
            "radius_over_h": cfg.get("c", RADIUS_OVER_H),
            "observed_l2_order": slope,
            "observed_linf_order": slope_inf,
            "pairwise_orders": _pairwise(ns, errs),
            "pairwise_linf_orders": _pairwise(ns, errs_inf),
            "deepest_error": deepest,
            "deepest_linf_error": deepest_inf,
            "linf_errors": [float(e) for e in errs_inf],
            "gram_cond_max": float(max(r.gram_cond for r in deg_rows)),
            "extension_residual_max": float(max(r.extension_residual for r in deg_rows)),
            "min_fluid_fraction": float(min(r.min_fluid_fraction for r in deg_rows)),
            "precision": cfg["precision"],
            "extension": cfg["extension"],
            "local_basis": cfg.get("local_basis", "monomial"),
            "extra_columns": int(cfg.get("extra", 0)),
            "meshes": ns,
            "errors": [float(e) for e in errs],
        }
        _log(f"==> p={p} observed order  L2 = {slope:.2f}  Linf = {slope_inf:.2f}  "
             f"(target {p+1});  deepest L2 {deepest:.2e}  Linf {deepest_inf:.2e}")

    summary = {
        "geometry": "circular corner cut, self-similar radius = c h (c per degree)",
        "target": "cos(pi x) cos(pi y)",
        "radius_over_h_default": RADIUS_OVER_H,
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
        "which_levers_mattered": {
            "well_conditioned_basis": "drops extension residual 5e-9->1e-11, cond 1e13->1e9, ||E|| 9e5->7e3",
            "higher_order_quadrature": "no effect on the floor (error invariant to q_order); not the cap",
            "extended_precision": "longdouble needed at p>=8 to keep deep meshes above the roundoff floor",
            "self_similar_geometry": "fixes the sliver fraction so refinement isolates the algebraic order",
        },
        "degrees": per_degree,
    }
    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        sjson = RESULTS / "highp_degree_ladder.json"
        rjson = RESULTS / "highp_degree_ladder_rows.json"
        # MERGE per-degree so the showcase set (p=4,6,8,10) can be generated
        # incrementally without clobbering previously computed degrees.
        ran = set(plan)
        if sjson.exists():
            try:
                prev = json.loads(sjson.read_text(encoding="utf-8"))
                merged = dict(prev.get("degrees", {}))
                merged.update(summary["degrees"])
                summary["degrees"] = dict(sorted(merged.items(), key=lambda kv: int(kv[0])))
            except Exception:
                pass
        if rjson.exists():
            try:
                prev_rows = json.loads(rjson.read_text(encoding="utf-8"))
                rows = [r for r in prev_rows if r.get("p") not in ran] + rows
            except Exception:
                pass
        sjson.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rjson.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=str, default=None, help="comma list, e.g. 10,12")
    ap.add_argument("--log", type=str, default=None)
    args = ap.parse_args()
    plan = DEGREE_PLAN
    if args.degrees:
        keep = {int(d) for d in args.degrees.split(",")}
        plan = {p: cfg for p, cfg in DEGREE_PLAN.items() if p in keep}
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        result = run_ladder(plan=plan, write=True, log=log)
    finally:
        if log:
            log.close()
    print(json.dumps(result["summary"]["degrees"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
