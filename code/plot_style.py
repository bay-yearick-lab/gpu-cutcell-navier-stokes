from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

NAIVE = "#666666"
PULLBACK = "#22517a"
SMALL_SUPPORT = "#a83232"
LOCAL_GRAPH = "#1f6f6f"
BOUNDARY = "#1c7c34"
TEXT = "#202020"
AXIS = "#333333"
GRID = "#d7d7d2"
PANEL_BG = "#ffffff"
DATA_BG = "#ffffff"

# Muted slate accent for panel titles, schematic frames, and axis furniture
# (matches the archival JCP look of the predecessor manuscript).
TITLE = "#2f4858"
# Neutral fill used for removed regions in schematic setup panels.
SETUP_BG = "#f3f5f5"

# Quiet grays for reference guides (slope triangles, asymptotic laws, floors).
REFERENCE = "#9a9a9a"

# Canonical reader-facing labels for Section 6 method/control contrasts.
# The active space is a control/pathology, not a method.  The reduced
# coordinates are the BMG/BMF coordinates used by the paper's two formulations.
LABEL_ACTIVE = "unaggregated active control"
LABEL_PHYSICAL = "BMG/BMF visible coordinates"

# ---------------------------------------------------------------------------
# Uniform Section 6 figure geometry.
#
# Every Section 6 result panel (convergence histories, spectral diagnostics)
# shares one wide, short canvas so that, at a single \includegraphics width,
# all of them render at the same on-page size regardless of panel count.
# Every setup/simulation "hero" shares one square-ish canvas at a single
# smaller width.  Keep these two constants as the only Section 6 figure sizes.
# ---------------------------------------------------------------------------
WIDE_SIZE = (10.2, 3.8)   # result plots: 2- or 3-panel, \includegraphics width 0.98
HERO_SIZE = (5.2, 4.6)    # setup / simulation heroes, \includegraphics width 0.60

# Classy field colormaps on the project palette (used for simulation fields
# instead of saturated rainbow maps).  FIELD_SEQUENTIAL is a single-hue muted
# blue ramp for nonnegative fields (speed, solution magnitude); FIELD_DIVERGING
# is a muted red / neutral-light / muted blue ramp for signed oscillatory fields
# (acoustic pressure).
FIELD_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "bm_sequential",
    ["#fbfcfd", "#e1e9f0", "#bcd0e0", "#8aafca", "#5786ad", "#346089", "#22517a", "#173249"],
)
FIELD_DIVERGING = LinearSegmentedColormap.from_list(
    "bm_diverging",
    ["#7c2424", "#a83232", "#c9817f", "#e3d7d1", "#f7f8f7", "#aebccb", "#5d83a6", "#2f5f86", "#1a3a58"],
)


# Muted degree ramp drawn from the project palette: deep indigo through the
# primal blue, the BMG teal, and a restrained green into a soft sage.  It keeps
# a multi-degree convergence panel reading as one calm color family on the same
# palette as the field, schematic, and structural figures (no viridis rainbow).
DEGREE_CMAP = LinearSegmentedColormap.from_list(
    "bm_degree",
    ["#26305f", "#234e85", "#1f6f6f", "#3a8a4f", "#86a64e"],
)


def degree_color(p: int, p_min: int, p_max: int):
    """Muted, classy per-series color for a degree or order sweep.

    Maps the degree p in [p_min, p_max] onto the project-palette degree ramp
    (deep indigo through primal blue, BMG teal, and muted green to soft sage)
    so a multi-degree convergence panel reads as one calm color family rather
    than a rainbow, consistent with the rest of the manuscript figures.
    """
    t = 0.0 if p_max <= p_min else (p - p_min) / (p_max - p_min)
    return DEGREE_CMAP(0.04 + 0.92 * t)


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            # Embed TrueType (Type 42) rather than Type 3 fonts so the compiled
            # PDF passes journal font checks.  Applies to every figure that
            # imports this module; standalone scripts set the same two keys.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 10.4,
            "axes.titlesize": 10.8,
            "legend.fontsize": 9.3,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": DATA_BG,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.42,
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "axes.prop_cycle": cycler(
                color=[NAIVE, PULLBACK, SMALL_SUPPORT, LOCAL_GRAPH, BOUNDARY, "#8a6fbf"]
            ),
        }
    )


def style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor(DATA_BG)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.45, alpha=0.42)
    else:
        ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(color=AXIS, labelcolor=TEXT, width=0.7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    # Panel titles in the muted slate accent, sitting in their own band above
    # the axes (archival JCP convention).
    ax.title.set_color(TITLE)
    ax.title.set_fontsize(10.6)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = 0.018,
    y: float = 0.975,
    outside: bool = False,
    fontsize: float = 9.0,
) -> None:
    """Place a small consistent panel label without competing with the data."""
    if outside:
        x, y = -0.10, 1.035
        va = "bottom"
        clip = False
    else:
        va = "top"
        clip = True
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=TITLE,
        clip_on=clip,
        zorder=8,
    )


def legend_top(
    fig: plt.Figure,
    handles,
    labels=None,
    *,
    ncol: int = 4,
    y: float = 1.0,
    fontsize: float = 8.8,
) -> None:
    """One shared, deduplicated legend strip above the panels.

    This is the standard for every Section 6 result figure: the legend reads as
    a header band, never merged with the figure caption below.
    """
    if labels is None:
        labels = [h.get_label() for h in handles]
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        if lab and not lab.startswith("_") and lab not in seen:
            seen[lab] = h
    if not seen:
        return
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.6,
        handletextpad=0.7,
        fontsize=fontsize,
    )


def legend_below(
    fig: plt.Figure,
    handles,
    labels,
    *,
    ncol: int = 4,
    y: float = 0.0,
    fontsize: float = 8.4,
) -> None:
    """One shared, deduplicated legend below the panels, off the data region."""
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        if lab and not lab.startswith("_") and lab not in seen:
            seen[lab] = h
    if not seen:
        return
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.5,
        fontsize=fontsize,
    )


def slope_guide(
    ax: plt.Axes,
    h,
    y_anchor: float,
    order: float,
    *,
    label: str | None = None,
    color: str = REFERENCE,
    ls: str = (0, (5, 4)),
    lw: float = 0.9,
    anchor_at: int = 0,
) -> Line2D:
    """Draw a labeled power-law reference line ``y = C h^order`` on log-log axes.

    The guide is anchored to ``y_anchor`` at element ``anchor_at`` of ``h`` and
    returned as a ``Line2D`` so it can be carried into the figure legend; this
    keeps every reference slope explained instead of left as in-data text.
    """
    h = np.asarray(h, dtype=float)
    ref = y_anchor * (h / h[anchor_at]) ** order
    (line,) = ax.loglog(h, ref, ls=ls, lw=lw, color=color, zorder=1.5)
    if label is not None:
        line.set_label(label)
    return line


def reference_handle(label: str, *, color: str = REFERENCE, ls: str = ":", lw: float = 1.2) -> Line2D:
    """A free-standing legend handle for a reference curve (cap, floor, law)."""
    return Line2D([0], [0], color=color, ls=ls, lw=lw, label=label)


def asymptotic_order(elements, errors, *, is_h: bool = False, min_decrease: float = 1.05) -> float:
    """Observed order of accuracy from the finest strictly-converging mesh pair.

    Walks from the two finest meshes toward coarser and returns the local order
    ``log(e_coarse/e_fine)/log(h_coarse/h_fine)`` for the finest pair whose error
    actually decreased by at least ``min_decrease`` under refinement.  This skips
    error-floor upticks and plateaus, so it reports the genuine asymptotic rate
    rather than a global least-squares fit smeared across pre-asymptotic and
    floored points.  Returns NaN if no converging pair exists.
    """
    h = np.asarray(elements, dtype=float)
    h = h if is_h else 1.0 / h
    e = np.asarray(errors, dtype=float)
    order = np.argsort(h)  # ascending h: finest mesh first
    h, e = h[order], e[order]
    for i in range(len(h) - 1):
        if e[i] > 0.0 and e[i + 1] > 0.0 and e[i + 1] / e[i] >= min_decrease:
            return float(np.log(e[i + 1] / e[i]) / np.log(h[i + 1] / h[i]))
    return float("nan")


def order_triangle(
    ax: plt.Axes,
    anchor_xy: tuple[float, float],
    order: float,
    *,
    dec: float = -0.30,
    color: str = "#7c7c76",
    face: str = "#e9e9e3",
    face_alpha: float = 0.7,
    text_color: str | None = None,
    fontsize: float = 9.2,
) -> None:
    """Draw a classy JCP slope triangle on log-log axes.

    The hypotenuse runs from ``anchor_xy`` to a second point ``dec`` decades
    away in ``x``, with data slope ``order`` so it reads parallel to a
    degree-``order`` convergence curve.  The right-angle vertex sits at the
    lower corner: the horizontal leg (labeled ``1``) runs along the base and the
    vertical leg (labeled by ``order``) rises on the anchor side, so the marker
    nestles beneath the curves in the conventional convergence-plot orientation,
    regardless of whether the ``x`` axis is inverted.
    """
    from matplotlib.patches import Polygon

    if text_color is None:
        text_color = TITLE
    xA, yA = float(anchor_xy[0]), float(anchor_xy[1])
    xB = xA * (10.0 ** dec)
    yB = yA * (xB / xA) ** float(order)
    # Right-angle corner at the anchor x and the lower y, so the triangle sits
    # under the hypotenuse with the base along the bottom and the rise on the
    # anchor side.
    yC = min(yA, yB)
    tri = Polygon([(xA, yA), (xB, yB), (xA, yC)], closed=True, facecolor=face,
                  edgecolor="none", alpha=face_alpha, zorder=1.8)
    ax.add_patch(tri)
    ax.plot([xA, xB], [yC, yC], color=color, lw=1.05, zorder=1.9, solid_capstyle="round")
    ax.plot([xA, xA], [yC, yA if yA > yB else yB], color=color, lw=1.05, zorder=1.9,
            solid_capstyle="round")
    ax.plot([xA, xB], [yA, yB], color="#3a3a36", lw=1.25, zorder=2.0, solid_capstyle="round")
    x_mid = (xA * xB) ** 0.5
    y_mid = (yA * yB) ** 0.5
    order_txt = f"{int(order)}" if float(order).is_integer() else f"{order:g}"
    ax.annotate("1", (x_mid, yC), textcoords="offset points", xytext=(0, -8),
                ha="center", va="top", fontsize=fontsize, color=text_color, zorder=2.1)
    ax.annotate(order_txt, (xA, y_mid), textcoords="offset points", xytext=(-6, 0),
                ha="right", va="center", fontsize=fontsize, color=text_color, zorder=2.1)


# ---------------------------------------------------------------------------
# Schematic setup panels (governing-equation header, engineering hatching).
#
# These render an experiment's setup as a clean cartoon: no axis frame, no
# colorbar, a faint field hint, the embedded cut boundary, engineering wall
# hatching, and the governing equation typeset above the domain.  This is the
# archival JCP idiom; it deliberately does not look like a raw matplotlib field
# dump with ticks and a colorbar.
# ---------------------------------------------------------------------------

def setup_panel(ax: plt.Axes, *, xlim, ylim) -> None:
    """Prepare an axis as a schematic setup cartoon: neutral panel, no ticks."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def equation_banner(ax: plt.Axes, lines, *, y0: float = 0.965, dy: float = 0.072,
                    fontsize: float = 12.5, sub_fontsize: float = 9.8) -> None:
    """Typeset a governing-equation header above the domain (figure-fraction y).

    The first line is the governing equation (large); subsequent lines are
    smaller annotations (data, boundary conditions).
    """
    for i, line in enumerate(lines):
        ax.text(0.5, y0 - i * dy, line, transform=ax.transAxes, ha="center", va="top",
                fontsize=fontsize if i == 0 else sub_fontsize,
                color=TITLE if i == 0 else TEXT, zorder=8)


def wall_hatch(ax, p0, p1, *, n: int = 16, length: float = 0.04, side: float = 1.0,
               color: str = "#2b2b2b", lw_wall: float = 2.0, lw_tick: float = 0.8) -> None:
    """Draw a solid wall segment p0->p1 with engineering hatching on one side.

    ``side`` is +1 or -1, selecting which side the inward hatch ticks fall on.
    Used to mark strongly enforced (Dirichlet / no-slip) boundaries.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw_wall, zorder=5,
            solid_capstyle="round")
    d = p1 - p0
    L = float(np.hypot(*d))
    if L <= 0:
        return
    t = d / L
    nrm = np.array([-t[1], t[0]]) * side
    for i in range(n):
        s = (i + 0.5) / n
        base = p0 + d * s
        tip = base + (nrm - t) * length
        ax.plot([base[0], tip[0]], [base[1], tip[1]], color=color, lw=lw_tick, zorder=5,
                solid_capstyle="round")


def hero_label(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    color: str = "white",
    fontsize: float = 9.0,
    ha: str = "left",
    va: str = "center",
    arrow: bool = True,
) -> None:
    """White annotation with a soft dark halo, legible over any field color."""
    import matplotlib.patheffects as pe

    stroke = [pe.withStroke(linewidth=2.1, foreground="#1c1c1c", alpha=0.55)]
    ann = ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        zorder=6,
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.9) if arrow else None,
    )
    ann.set_path_effects(stroke)


def field_colorbar(fig: plt.Figure, im, ax: plt.Axes, label: str):
    """Standardized slim colorbar for simulation/field heroes."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=9.3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8.0)
    return cb


def save_pdf(fig: plt.Figure, path: Path, *, pad_inches: float = 0.03) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)


def save_pdf_png(
    fig: plt.Figure,
    path: Path,
    *,
    png_dpi: int = 180,
    pdf_dpi: int | None = None,
    pad_inches: float = 0.03,
) -> None:
    # ``pdf_dpi`` caps the resolution of any rasterized layer (e.g. an imshow
    # texture) embedded in the PDF; vector elements stay vector regardless.  It
    # is used to keep a raster-heavy hero (the line-integral-convolution flow
    # field) from inflating the PDF well beyond what the on-page size resolves.
    pdf_kw = {"dpi": pdf_dpi} if pdf_dpi is not None else {}
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches, **pdf_kw)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=pad_inches, dpi=png_dpi)


apply_paper_style()
