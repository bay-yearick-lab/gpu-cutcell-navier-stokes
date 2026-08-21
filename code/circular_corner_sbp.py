from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import pi, sqrt

import numpy as np
from scipy.linalg import eigvalsh, svdvals

from cut_spline_sbp import SplineBasis1D, assemble_cut_matrices, make_open_uniform_basis
from oblique_cut_sbp import (
    ObliqueAggregation2D,
    VolumeQuadrature2D,
    _eval_all,
    _tensor_greville,
    _tensor_values,
    tensor_polynomial_coefficients,
)


@dataclass(frozen=True)
class CircularCornerCut:
    """Square with a quarter-disk removed from the lower-left corner.

    The physical domain is

        Omega_R = [0,1]^2 minus {x^2 + y^2 < R^2, x>=0, y>=0}.

    For R close to sqrt(2) h the circular boundary nearly passes through the
    first-cell corner (h,h), creating a curved small cut cell in the first cell.
    """

    radius: float


class CircularStructureMode(str, Enum):
    """Select raw quadrature matrices or certified sharp Green matrices."""

    RAW = "raw"
    CERTIFIED = "certified"


class CircularStructureStatus(str, Enum):
    """State whether the returned sharp matrices carry a Green certificate."""

    NOT_CERTIFIED = "not_certified"
    CERTIFIED = "certified"


SHARP_GREEN_MAX_RELATIVE_CORRECTION = 1.0e-12
SHARP_GREEN_FORMULA_ID = "symS_skewQ_v1"
SHARP_GREEN_CERTIFICATE_SCHEMA = "sharp_green_certificate_v1"


@dataclass(frozen=True)
class DirectionalSharpGreenDiagnostics:
    """Raw defect and candidate correction measurements for one direction."""

    direction: str
    raw_green_inf: float
    delta_q_inf: float
    delta_s_inf: float
    rel_delta_q: float
    rel_delta_s: float
    candidate_closed_green_inf: float
    admissible_for_certification: bool
    relative_correction_limit: float
    formula_id: str


class DirectionalSharpGreenClosureError(ValueError):
    """Raised when a sharp directional correction is not roundoff admissible."""

    def __init__(self, diagnostics: DirectionalSharpGreenDiagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            "sharp Green closure rejected for direction "
            f"{diagnostics.direction}: raw_green_inf={diagnostics.raw_green_inf:.17e}, "
            f"delta_q_inf={diagnostics.delta_q_inf:.17e}, "
            f"delta_s_inf={diagnostics.delta_s_inf:.17e}, "
            f"rel_delta_q={diagnostics.rel_delta_q:.17e}, "
            f"rel_delta_s={diagnostics.rel_delta_s:.17e}"
        )


class SharpGreenCertificationError(ValueError):
    """Report an atomic two-direction sharp Green certification failure."""

    def __init__(
        self,
        diagnostics_x: DirectionalSharpGreenDiagnostics,
        diagnostics_y: DirectionalSharpGreenDiagnostics,
    ):
        self.diagnostics_x = diagnostics_x
        self.diagnostics_y = diagnostics_y
        self.failed_directions = tuple(
            item.direction
            for item in (diagnostics_x, diagnostics_y)
            if not item.admissible_for_certification
        )
        self.relative_correction_limit = SHARP_GREEN_MAX_RELATIVE_CORRECTION
        details = "; ".join(
            f"{item.direction}: raw_green_inf={item.raw_green_inf:.17e}, "
            f"delta_q_inf={item.delta_q_inf:.17e}, delta_s_inf={item.delta_s_inf:.17e}, "
            f"rel_delta_q={item.rel_delta_q:.17e}, rel_delta_s={item.rel_delta_s:.17e}"
            for item in (diagnostics_x, diagnostics_y)
        )
        super().__init__(
            "sharp Green certification rejected for directions "
            f"{self.failed_directions} with limit {self.relative_correction_limit:.17e}; {details}"
        )


class UncertifiedCircularStructureError(ValueError):
    """Reject theorem-facing use of an uncertified circular structure."""


@dataclass(frozen=True)
class SharpGreenCertificate:
    """Immutable provenance for one atomic sharp Green certification."""

    schema_version: str
    formula_id: str
    structure_mode: CircularStructureMode
    structure_status: CircularStructureStatus
    relative_correction_limit: float
    quadrature_backend: str
    quadrature_order: int
    degree: int
    n_elements_x: int
    n_elements_y: int
    radius: float
    diagnostics_x: DirectionalSharpGreenDiagnostics
    diagnostics_y: DirectionalSharpGreenDiagnostics
    closed_green_inf_x: float
    closed_green_inf_y: float
    assembly_configuration_hash: str
    raw_pair_hash: str
    certified_pair_hash: str
    certificate_hash: str


@dataclass(frozen=True)
class CircularStructureLineage:
    """Hash lineage from a sharp assembly through a structure-preserving map."""

    record_kind: str
    parent_operator_hash: str | None
    child_operator_hash: str
    sharp_certificate_hash: str | None
    transformation_kind: str
    transformation_hash: str | None
    measured_green_inf_x: float
    measured_green_inf_y: float


def _canonical_hash_payload(value):
    """Convert metadata to a finite deterministic JSON hash representation."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("hash metadata must be finite")
        return value.hex()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _canonical_hash_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_hash_payload(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported hash payload type: {type(value).__name__}")


def _hash_metadata(value) -> str:
    payload = _canonical_hash_payload(value)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_descriptor(name: str, array: np.ndarray, *, integer: bool = False) -> dict:
    """Describe exact little-endian contiguous bytes for deterministic hashing."""
    dtype = "<i8" if integer else "<f8"
    data = np.ascontiguousarray(np.asarray(array), dtype=np.dtype(dtype))
    if not integer and not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains nonfinite values")
    return {
        "name": name,
        "dtype": dtype,
        "shape": [int(item) for item in data.shape],
        "data_sha256": hashlib.sha256(data.tobytes(order="C")).hexdigest(),
    }


def hash_named_float64_matrices(**matrices: np.ndarray) -> str:
    """Hash a named set of exact float64 matrix values."""
    return _hash_metadata({
        name: _array_descriptor(name, array)
        for name, array in sorted(matrices.items())
    })


def _sharp_operator_hash(H, Qx, Qy, K, H_background, Sx, Sy) -> str:
    return hash_named_float64_matrices(
        H=H, Qx=Qx, Qy=Qy, K=K, H_background=H_background, Sx=Sx, Sy=Sy
    )


def measure_directional_sharp_green_pair(
    Q: np.ndarray,
    S: np.ndarray,
    direction: str,
) -> DirectionalSharpGreenDiagnostics:
    """Measure a directional constrained projection without modifying inputs."""
    q = np.asarray(Q, dtype=float)
    s = np.asarray(S, dtype=float)
    if q.ndim != 2 or s.ndim != 2 or q.shape != s.shape or q.shape[0] != q.shape[1]:
        raise ValueError("Q and S must be square matrices with identical shapes")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(s)):
        raise ValueError("Q and S must contain only finite values")
    if str(direction) not in {"x", "y"}:
        raise ValueError("direction must be 'x' or 'y'")

    s_candidate = 0.5 * (s + s.T)
    q_candidate = 0.5 * (q - q.T) + 0.5 * s_candidate
    raw_green_inf = float(np.linalg.norm(q + q.T - s, ord=np.inf))
    delta_q_inf = float(np.linalg.norm(q_candidate - q, ord=np.inf))
    delta_s_inf = float(np.linalg.norm(s_candidate - s, ord=np.inf))
    tiny = np.finfo(np.float64).tiny
    rel_delta_q = delta_q_inf / max(float(np.linalg.norm(q, ord=np.inf)), tiny)
    rel_delta_s = delta_s_inf / max(float(np.linalg.norm(s, ord=np.inf)), tiny)
    candidate_closed_green_inf = float(
        np.linalg.norm(q_candidate + q_candidate.T - s_candidate, ord=np.inf)
    )
    return DirectionalSharpGreenDiagnostics(
        direction=str(direction),
        raw_green_inf=raw_green_inf,
        delta_q_inf=delta_q_inf,
        delta_s_inf=delta_s_inf,
        rel_delta_q=rel_delta_q,
        rel_delta_s=rel_delta_s,
        candidate_closed_green_inf=candidate_closed_green_inf,
        admissible_for_certification=(
            rel_delta_q <= SHARP_GREEN_MAX_RELATIVE_CORRECTION
            and rel_delta_s <= SHARP_GREEN_MAX_RELATIVE_CORRECTION
        ),
        relative_correction_limit=SHARP_GREEN_MAX_RELATIVE_CORRECTION,
        formula_id=SHARP_GREEN_FORMULA_ID,
    )


def close_directional_sharp_green_pair(
    Q: np.ndarray,
    S: np.ndarray,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, DirectionalSharpGreenClosureDiagnostics]:
    """Project a roundoff-admissible sharp pair onto the Green identity."""
    q = np.asarray(Q, dtype=float)
    s = np.asarray(S, dtype=float)
    diagnostics = measure_directional_sharp_green_pair(q, s, direction)
    s_closed = 0.5 * (s + s.T)
    q_closed = 0.5 * (q - q.T) + 0.5 * s_closed
    if not diagnostics.admissible_for_certification:
        raise DirectionalSharpGreenClosureError(diagnostics)
    return q_closed, s_closed, diagnostics


DirectionalSharpGreenClosureDiagnostics = DirectionalSharpGreenDiagnostics


@dataclass(frozen=True)
class CircularCornerMatrices2D:
    H: np.ndarray
    Qx: np.ndarray
    Qy: np.ndarray
    K: np.ndarray
    H_background: np.ndarray
    Sx: np.ndarray
    Sy: np.ndarray
    nx: int
    ny: int
    cut: CircularCornerCut
    volume_quadrature: VolumeQuadrature2D | None = None
    structure_mode: CircularStructureMode = CircularStructureMode.RAW
    structure_status: CircularStructureStatus = CircularStructureStatus.NOT_CERTIFIED
    green_diagnostics_x: DirectionalSharpGreenDiagnostics = field(default=None)
    green_diagnostics_y: DirectionalSharpGreenDiagnostics = field(default=None)
    green_certificate: SharpGreenCertificate | None = None
    structure_lineage: CircularStructureLineage = field(default=None)

    def __post_init__(self) -> None:
        """Populate omitted structure metadata as an explicit raw record."""
        mode = CircularStructureMode(self.structure_mode)
        status = CircularStructureStatus(self.structure_status)
        diagnostics_x = self.green_diagnostics_x or measure_directional_sharp_green_pair(
            self.Qx, self.Sx, "x"
        )
        diagnostics_y = self.green_diagnostics_y or measure_directional_sharp_green_pair(
            self.Qy, self.Sy, "y"
        )
        lineage = self.structure_lineage or CircularStructureLineage(
            record_kind="sharp",
            parent_operator_hash=None,
            child_operator_hash=_sharp_operator_hash(
                self.H, self.Qx, self.Qy, self.K, self.H_background, self.Sx, self.Sy
            ),
            sharp_certificate_hash=None,
            transformation_kind="assembly",
            transformation_hash=None,
            measured_green_inf_x=diagnostics_x.raw_green_inf,
            measured_green_inf_y=diagnostics_y.raw_green_inf,
        )
        object.__setattr__(self, "structure_mode", mode)
        object.__setattr__(self, "structure_status", status)
        object.__setattr__(self, "green_diagnostics_x", diagnostics_x)
        object.__setattr__(self, "green_diagnostics_y", diagnostics_y)
        object.__setattr__(self, "structure_lineage", lineage)


def _certificate_payload(certificate: SharpGreenCertificate) -> dict:
    payload = asdict(certificate)
    payload.pop("certificate_hash")
    return payload


def _assembly_configuration_hash(
    *,
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: CircularCornerCut,
    quadrature_backend: str,
    quadrature_order: int,
    structure_mode: CircularStructureMode,
    capture_volume_quadrature: bool,
    capture_basis: bool,
) -> str:
    """Hash every sharp-assembly choice that can affect the returned record."""
    return _hash_metadata({
        "quadrature_backend": quadrature_backend,
        "quadrature_order": int(quadrature_order),
        "degree_x": int(basis_x.degree),
        "degree_y": int(basis_y.degree),
        "n_elements_x": int(basis_x.n_elements),
        "n_elements_y": int(basis_y.n_elements),
        "radius": float(cut.radius),
        "basis_type": "open_uniform_bspline",
        "knots_x": _array_descriptor("knots_x", basis_x.knots),
        "knots_y": _array_descriptor("knots_y", basis_y.knots),
        "dtype": "<f8",
        "structure_mode": structure_mode,
        "relative_correction_limit": SHARP_GREEN_MAX_RELATIVE_CORRECTION,
        "formula_id": SHARP_GREEN_FORMULA_ID,
        "capture_volume_quadrature": bool(capture_volume_quadrature),
        "capture_basis": bool(capture_basis),
    })


def _sharp_structure_transaction(
    *,
    H: np.ndarray,
    Qx: np.ndarray,
    Qy: np.ndarray,
    K: np.ndarray,
    H_background: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: CircularCornerCut,
    quadrature_backend: str,
    quadrature_order: int,
    structure_mode: CircularStructureMode,
    capture_volume_quadrature: bool,
    capture_basis: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    DirectionalSharpGreenDiagnostics,
    DirectionalSharpGreenDiagnostics,
    CircularStructureStatus,
    SharpGreenCertificate | None,
    CircularStructureLineage,
]:
    """Apply one atomic raw or certified structure transaction."""
    try:
        mode = CircularStructureMode(structure_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown circular structure mode: {structure_mode!r}") from error
    diagnostics_x = measure_directional_sharp_green_pair(Qx, Sx, "x")
    diagnostics_y = measure_directional_sharp_green_pair(Qy, Sy, "y")
    configuration_hash = _assembly_configuration_hash(
        basis_x=basis_x,
        basis_y=basis_y,
        cut=cut,
        quadrature_backend=quadrature_backend,
        quadrature_order=quadrature_order,
        structure_mode=mode,
        capture_volume_quadrature=capture_volume_quadrature,
        capture_basis=capture_basis,
    )
    if mode is CircularStructureMode.RAW:
        operator_hash = _sharp_operator_hash(H, Qx, Qy, K, H_background, Sx, Sy)
        lineage = CircularStructureLineage(
            record_kind="sharp",
            parent_operator_hash=None,
            child_operator_hash=operator_hash,
            sharp_certificate_hash=None,
            transformation_kind="assembly",
            transformation_hash=None,
            measured_green_inf_x=diagnostics_x.raw_green_inf,
            measured_green_inf_y=diagnostics_y.raw_green_inf,
        )
        return (
            Qx, Qy, Sx, Sy, diagnostics_x, diagnostics_y,
            CircularStructureStatus.NOT_CERTIFIED, None, lineage,
        )

    if not (
        diagnostics_x.admissible_for_certification
        and diagnostics_y.admissible_for_certification
    ):
        raise SharpGreenCertificationError(diagnostics_x, diagnostics_y)

    sx_closed = 0.5 * (Sx + Sx.T)
    sy_closed = 0.5 * (Sy + Sy.T)
    qx_closed = 0.5 * (Qx - Qx.T) + 0.5 * sx_closed
    qy_closed = 0.5 * (Qy - Qy.T) + 0.5 * sy_closed
    closed_x = float(np.linalg.norm(qx_closed + qx_closed.T - sx_closed, ord=np.inf))
    closed_y = float(np.linalg.norm(qy_closed + qy_closed.T - sy_closed, ord=np.inf))
    for direction, residual, q_closed, s_closed in (
        ("x", closed_x, qx_closed, sx_closed),
        ("y", closed_y, qy_closed, sy_closed),
    ):
        rounding_bound = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            float(np.linalg.norm(q_closed + q_closed.T, ord=np.inf)),
            float(np.linalg.norm(s_closed, ord=np.inf)),
        )
        if residual > rounding_bound:
            raise ValueError(
                f"{direction} certified sharp Green residual exceeds final-rounding bound"
            )

    raw_pair_hash = hash_named_float64_matrices(
        Qx_raw=Qx, Qy_raw=Qy, Sx_raw=Sx, Sy_raw=Sy
    )
    certified_pair_hash = hash_named_float64_matrices(
        Qx_certified=qx_closed, Qy_certified=qy_closed,
        Sx_certified=sx_closed, Sy_certified=sy_closed,
    )
    certificate_fields = {
        "schema_version": SHARP_GREEN_CERTIFICATE_SCHEMA,
        "formula_id": SHARP_GREEN_FORMULA_ID,
        "structure_mode": CircularStructureMode.CERTIFIED,
        "structure_status": CircularStructureStatus.CERTIFIED,
        "relative_correction_limit": SHARP_GREEN_MAX_RELATIVE_CORRECTION,
        "quadrature_backend": quadrature_backend,
        "quadrature_order": int(quadrature_order),
        "degree": int(basis_x.degree),
        "n_elements_x": int(basis_x.n_elements),
        "n_elements_y": int(basis_y.n_elements),
        "radius": float(cut.radius),
        "diagnostics_x": diagnostics_x,
        "diagnostics_y": diagnostics_y,
        "closed_green_inf_x": closed_x,
        "closed_green_inf_y": closed_y,
        "assembly_configuration_hash": configuration_hash,
        "raw_pair_hash": raw_pair_hash,
        "certified_pair_hash": certified_pair_hash,
    }
    certificate_hash = _hash_metadata({
        key: asdict(value) if isinstance(value, DirectionalSharpGreenDiagnostics) else value
        for key, value in certificate_fields.items()
    })
    certificate = SharpGreenCertificate(
        **certificate_fields,
        certificate_hash=certificate_hash,
    )
    operator_hash = _sharp_operator_hash(
        H, qx_closed, qy_closed, K, H_background, sx_closed, sy_closed
    )
    lineage = CircularStructureLineage(
        record_kind="sharp",
        parent_operator_hash=None,
        child_operator_hash=operator_hash,
        sharp_certificate_hash=certificate_hash,
        transformation_kind="assembly",
        transformation_hash=None,
        measured_green_inf_x=closed_x,
        measured_green_inf_y=closed_y,
    )
    return (
        qx_closed, qy_closed, sx_closed, sy_closed,
        diagnostics_x, diagnostics_y, CircularStructureStatus.CERTIFIED,
        certificate, lineage,
    )


def require_certified_structure(
    matrices: CircularCornerMatrices2D,
) -> CircularCornerMatrices2D:
    """Validate certificate consistency before theorem-facing structure use."""
    certificate = matrices.green_certificate
    consistent = (
        matrices.structure_mode is CircularStructureMode.CERTIFIED
        and matrices.structure_status is CircularStructureStatus.CERTIFIED
        and certificate is not None
        and certificate.structure_mode is CircularStructureMode.CERTIFIED
        and certificate.structure_status is CircularStructureStatus.CERTIFIED
        and certificate.certificate_hash == matrices.structure_lineage.sharp_certificate_hash
        and certificate.certificate_hash == _hash_metadata(_certificate_payload(certificate))
        and matrices.structure_lineage.child_operator_hash == _sharp_operator_hash(
            matrices.H, matrices.Qx, matrices.Qy, matrices.K,
            matrices.H_background, matrices.Sx, matrices.Sy,
        )
    )
    if not consistent:
        raise UncertifiedCircularStructureError(
            "circular matrices do not carry a consistent certified sharp Green structure"
        )
    return matrices


@dataclass(frozen=True)
class CircularCornerActiveSets2D:
    active: np.ndarray
    interior: np.ndarray
    small_support: np.ndarray
    gamma: np.ndarray


def _node_chunk(n_basis: int, n_nodes: int, budget_bytes: float = 8.5e8) -> int:
    """Node-chunk size for streaming a dense ``(n_basis, chunk)`` basis tensor.

    A chunk holds three such tensors (values and both gradients) inside
    ``_tensor_values``; ``budget_bytes`` targets one tensor.  If the whole node
    set already fits the budget the chunk equals ``n_nodes`` (a single matmul,
    bit-identical to the unchunked path); otherwise the set is split into pieces
    of at least this size.  This keeps coarse and mid grids exact while bounding
    peak memory on the finest grids, where the sole difference is a sub-ULP
    cross-chunk summation order.
    """
    budget = max(20000, int(budget_bytes / (max(n_basis, 1) * 8)))
    return n_nodes if n_nodes <= budget else budget


def _unique_sorted(values: list[float], lo: float, hi: float, tol: float = 1.0e-13) -> np.ndarray:
    clipped = [lo, hi]
    for value in values:
        if lo + tol < value < hi - tol:
            clipped.append(float(value))
    arr = np.asarray(sorted(clipped), dtype=float)
    keep = [arr[0]]
    for value in arr[1:]:
        if abs(value - keep[-1]) > tol:
            keep.append(float(value))
    return np.asarray(keep, dtype=float)


def _radial_breaks(theta: float, radius: float, knots_x: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    values: list[float] = []
    if c > 1.0e-14:
        for xk in np.unique(knots_x):
            r = float(xk / c)
            if 0.0 < r < radius:
                values.append(r)
    if s > 1.0e-14:
        for yk in np.unique(knots_y):
            r = float(yk / s)
            if 0.0 < r < radius:
                values.append(r)
    return _unique_sorted(values, 0.0, radius)


def _angle_breaks(radius: float, knots_x: np.ndarray, knots_y: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for xk in np.unique(knots_x):
        if 0.0 < xk < radius:
            values.append(float(np.arccos(xk / radius)))
    for yk in np.unique(knots_y):
        if 0.0 < yk < radius:
            values.append(float(np.arcsin(yk / radius)))
    return _unique_sorted(values, 0.0, 0.5 * pi)


def _quarter_disk_quadrature(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    radius: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar quadrature over the removed quarter disk.

    The angular panels are split where the circular arc crosses knot lines.
    For each angular point, the radial interval is split at knot crossings along
    that ray.  This keeps the spline factors on single polynomial pieces in r.
    """
    gt, wt = np.polynomial.legendre.leggauss(order)
    gr, wr = np.polynomial.legendre.leggauss(order)
    theta_breaks = _angle_breaks(radius, basis_x.knots, basis_y.knots)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for ta, tb in zip(theta_breaks[:-1], theta_breaks[1:]):
        if tb <= ta:
            continue
        theta_nodes = 0.5 * (tb - ta) * gt + 0.5 * (tb + ta)
        theta_weights = 0.5 * (tb - ta) * wt
        for theta, wtheta in zip(theta_nodes, theta_weights):
            rb = _radial_breaks(float(theta), radius, basis_x.knots, basis_y.knots)
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            for ra, rb_hi in zip(rb[:-1], rb[1:]):
                if rb_hi <= ra:
                    continue
                r = 0.5 * (rb_hi - ra) * gr + 0.5 * (rb_hi + ra)
                w = 0.5 * (rb_hi - ra) * wr
                xs.append(r * c)
                ys.append(r * s)
                ws.append(wtheta * w * r)
    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def _positive_circular_volume_quadrature(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    radius: float,
    order: int,
    compute_basis: bool = True,
) -> VolumeQuadrature2D:
    """Positive quadrature over [0,1]^2 minus the quarter disk.

    The representation is cell-local for bookkeeping: every node is assigned to
    the Cartesian cell that generated it. Cut cells are integrated as vertical
    panels with lower limit y = sqrt(R^2 - x^2).

    With ``compute_basis=False`` the dense ``(n_full, nq)`` basis-value and
    gradient arrays are skipped (returned empty); only the quadrature nodes,
    weights, and cell ids are filled.  This lets callers that evaluate the basis
    in chunks avoid the multi-gigabyte dense tensor on fine high-order grids.
    """
    g, w = np.polynomial.legendre.leggauss(order)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    cell_ids: list[np.ndarray] = []
    h = 1.0 / basis_x.n_elements
    r2 = radius * radius
    knots_x = np.unique(basis_x.knots)
    knots_y = np.unique(basis_y.knots)

    def _append_points(x: np.ndarray, y: np.ndarray, weight: np.ndarray, cell_id: int) -> None:
        if x.size == 0:
            return
        xs.append(x)
        ys.append(y)
        ws.append(weight)
        cell_ids.append(np.full(x.size, cell_id, dtype=int))

    def _append_rect(xa: float, xb: float, ya: float, yb: float, cell_id: int) -> None:
        if xb <= xa or yb <= ya:
            return
        xq = 0.5 * (xb - xa) * g + 0.5 * (xb + xa)
        yq = 0.5 * (yb - ya) * g + 0.5 * (yb + ya)
        wx = 0.5 * (xb - xa) * w
        wy = 0.5 * (yb - ya) * w
        xx, yy = np.meshgrid(xq, yq, indexing="xy")
        ww = wy[:, None] * wx[None, :]
        _append_points(xx.ravel(), yy.ravel(), ww.ravel(), cell_id)

    for iy in range(basis_y.n_elements):
        y0 = iy * h
        y1 = (iy + 1) * h
        for ix in range(basis_x.n_elements):
            x0 = ix * h
            x1 = (ix + 1) * h
            cell_id = ix + basis_x.n_elements * iy
            if x0 * x0 + y0 * y0 >= r2 - 1.0e-15:
                _append_rect(x0, x1, y0, y1, cell_id)
                continue
            if x1 * x1 + y1 * y1 <= r2 + 1.0e-15:
                continue

            xb = [x0, x1]
            for value in knots_x:
                if x0 < value < x1:
                    xb.append(float(value))
            for yk in knots_y:
                if y0 < yk < y1 and yk < radius:
                    x_cross = float(np.sqrt(max(r2 - yk * yk, 0.0)))
                    if x0 < x_cross < x1:
                        xb.append(x_cross)
            if x0 < radius < x1:
                xb.append(radius)
            xbreaks = _unique_sorted(xb, x0, x1)

            for xa, xb_hi in zip(xbreaks[:-1], xbreaks[1:]):
                if xb_hi <= xa:
                    continue
                xq = 0.5 * (xb_hi - xa) * g + 0.5 * (xb_hi + xa)
                wx = 0.5 * (xb_hi - xa) * w
                for xnode, wxnode in zip(xq, wx):
                    if xnode >= radius:
                        ylo = y0
                    else:
                        ylo = max(y0, float(np.sqrt(max(r2 - xnode * xnode, 0.0))))
                    if ylo >= y1:
                        continue
                    ybreaks = _unique_sorted(
                        [float(yk) for yk in knots_y if ylo < yk < y1],
                        ylo,
                        y1,
                    )
                    for ya, yb in zip(ybreaks[:-1], ybreaks[1:]):
                        if yb <= ya:
                            continue
                        yq = 0.5 * (yb - ya) * g + 0.5 * (yb + ya)
                        wy = 0.5 * (yb - ya) * w
                        _append_points(
                            np.full_like(yq, xnode),
                            yq,
                            wxnode * wy,
                            cell_id,
                        )

    if not xs:
        n = basis_x.n_basis * basis_y.n_basis
        return VolumeQuadrature2D(
            x=np.empty(0),
            y=np.empty(0),
            weights=np.empty(0),
            cell_ids=np.empty(0, dtype=int),
            basis_values=np.empty((n, 0)),
            nx=basis_x.n_basis,
            ny=basis_y.n_basis,
            basis_grad_x=np.empty((n, 0)),
            basis_grad_y=np.empty((n, 0)),
        )
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    weights = np.concatenate(ws)
    n = basis_x.n_basis * basis_y.n_basis
    if not compute_basis:
        empty = np.empty((n, 0))
        return VolumeQuadrature2D(
            x=x,
            y=y,
            weights=weights,
            cell_ids=np.concatenate(cell_ids),
            basis_values=empty,
            nx=basis_x.n_basis,
            ny=basis_y.n_basis,
            basis_grad_x=empty,
            basis_grad_y=empty,
        )
    phi, dphix, dphiy = _tensor_values(basis_x, basis_y, x, y)
    return VolumeQuadrature2D(
        x=x,
        y=y,
        weights=weights,
        cell_ids=np.concatenate(cell_ids),
        basis_values=phi,
        nx=basis_x.n_basis,
        ny=basis_y.n_basis,
        basis_grad_x=dphix,
        basis_grad_y=dphiy,
    )


def _line_segment_boundary(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    *,
    fixed_axis: str,
    fixed_value: float,
    lo: float,
    hi: float,
    normal: tuple[float, float],
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = basis_x.n_basis * basis_y.n_basis
    Sx = np.zeros((n, n))
    Sy = np.zeros((n, n))
    if hi <= lo:
        return Sx, Sy
    g, w = np.polynomial.legendre.leggauss(order)
    knots = basis_y.knots if fixed_axis == "x" else basis_x.knots
    panels = _unique_sorted([float(k) for k in np.unique(knots) if lo < k < hi], lo, hi)
    for a, b in zip(panels[:-1], panels[1:]):
        t = 0.5 * (b - a) * g + 0.5 * (b + a)
        weights = 0.5 * (b - a) * w
        if fixed_axis == "x":
            x = np.full_like(t, fixed_value)
            y = t
        else:
            x = t
            y = np.full_like(t, fixed_value)
        phi, _dx, _dy = _tensor_values(basis_x, basis_y, x, y)
        mass = (phi * weights) @ phi.T
        Sx += normal[0] * mass
        Sy += normal[1] * mass
    return Sx, Sy


def _arc_boundary(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    radius: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary form on the circular hole, with domain outward normal -e_r."""
    n = basis_x.n_basis * basis_y.n_basis
    Sx = np.zeros((n, n))
    Sy = np.zeros((n, n))
    g, w = np.polynomial.legendre.leggauss(order)
    panels = _angle_breaks(radius, basis_x.knots, basis_y.knots)
    for a, b in zip(panels[:-1], panels[1:]):
        if b <= a:
            continue
        theta = 0.5 * (b - a) * g + 0.5 * (b + a)
        weights = 0.5 * (b - a) * w * radius
        c = np.cos(theta)
        s = np.sin(theta)
        x = radius * c
        y = radius * s
        phi, _dx, _dy = _tensor_values(basis_x, basis_y, x, y)
        # Normal points out of the computational domain and into the removed disk.
        Sx += (phi * (weights * (-c))) @ phi.T
        Sy += (phi * (weights * (-s))) @ phi.T
    return Sx, Sy


def _outer_boundary_forms(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    radius: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = basis_x.n_basis * basis_y.n_basis
    Sx = np.zeros((n, n))
    Sy = np.zeros((n, n))
    # Right and top full square sides.
    sx, sy = _line_segment_boundary(
        basis_x,
        basis_y,
        fixed_axis="x",
        fixed_value=1.0,
        lo=0.0,
        hi=1.0,
        normal=(1.0, 0.0),
        order=order,
    )
    Sx += sx
    Sy += sy
    sx, sy = _line_segment_boundary(
        basis_x,
        basis_y,
        fixed_axis="y",
        fixed_value=1.0,
        lo=0.0,
        hi=1.0,
        normal=(0.0, 1.0),
        order=order,
    )
    Sx += sx
    Sy += sy
    # Left and bottom sides start after the removed quarter disk.
    if radius < 1.0:
        sx, sy = _line_segment_boundary(
            basis_x,
            basis_y,
            fixed_axis="x",
            fixed_value=0.0,
            lo=radius,
            hi=1.0,
            normal=(-1.0, 0.0),
            order=order,
        )
        Sx += sx
        Sy += sy
        sx, sy = _line_segment_boundary(
            basis_x,
            basis_y,
            fixed_axis="y",
            fixed_value=0.0,
            lo=radius,
            hi=1.0,
            normal=(0.0, -1.0),
            order=order,
        )
        Sx += sx
        Sy += sy
    return Sx, Sy


def assemble_circular_corner_matrices(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    cut: CircularCornerCut,
    quadrature_order: int | None = None,
    *,
    capture_volume_quadrature: bool = False,
    capture_basis: bool = True,
    structure_mode: CircularStructureMode = CircularStructureMode.RAW,
) -> CircularCornerMatrices2D:
    """Assemble the circular-corner cut matrices.

    ``capture_volume_quadrature`` attaches the positive fluid-region quadrature to
    the returned object.  ``capture_basis`` (default ``True`` to preserve every
    existing caller) additionally fills the dense ``(n_basis, n_nodes)`` basis
    value/gradient tensors; passing ``capture_basis=False`` keeps only the nodes,
    weights, and cell ids so callers that evaluate the basis in chunks avoid the
    O(n^4) dense tensor on fine high-order grids.
    """
    if basis_x.n_elements != basis_y.n_elements:
        raise ValueError("The prototype expects the same element count in x and y.")
    if basis_x.degree != basis_y.degree:
        raise ValueError("The prototype expects the same spline degree in x and y.")
    radius = float(cut.radius)
    if not (0.0 < radius < 1.0):
        raise ValueError("radius must satisfy 0 < radius < 1 for this prototype.")
    q = quadrature_order or max(28, 2 * basis_x.degree + 10)
    mx = assemble_cut_matrices(basis_x, left=0.0, quadrature_order=q)
    my = assemble_cut_matrices(basis_y, left=0.0, quadrature_order=q)
    H_square = np.kron(my.H, mx.H)
    Qx_square = np.kron(my.H, mx.Q)
    Qy_square = np.kron(my.Q, mx.H)
    K_square = np.kron(my.H, mx.K) + np.kron(my.K, mx.H)
    H_background = H_square.copy()

    xh, yh, wh = _quarter_disk_quadrature(basis_x, basis_y, radius, q)
    # Accumulate the removed-hole Gram matrices in node chunks so the dense
    # (n_basis, n_hole_nodes) basis tensor never materialises: at q ~ 6p+2 on a
    # fine grid that single tensor exceeds several GB (e.g. 5 GB at p=6, n=32).
    n_basis = basis_x.n_basis * basis_y.n_basis
    H_hole = np.zeros((n_basis, n_basis))
    Qx_hole = np.zeros((n_basis, n_basis))
    Qy_hole = np.zeros((n_basis, n_basis))
    K_hole = np.zeros((n_basis, n_basis))
    # Node-chunk sized to a ~1 GB-per-tensor budget (three dense (n_basis, chunk)
    # arrays live at once inside _tensor_values).  Coarse and mid grids
    # (n <= 16 at p <= 8) fall in a single chunk and are bit-identical to the
    # unchunked assembly; only the finest grids split, where the sole difference
    # is a sub-ULP cross-chunk summation order.
    _chunk = _node_chunk(n_basis, xh.size)
    for _s in range(0, xh.size, _chunk):
        _e = min(_s + _chunk, xh.size)
        phi, dphix, dphiy = _tensor_values(basis_x, basis_y, xh[_s:_e], yh[_s:_e])
        whc = wh[_s:_e]
        H_hole += (phi * whc) @ phi.T
        Qx_hole += (phi * whc) @ dphix.T
        Qy_hole += (phi * whc) @ dphiy.T
        K_hole += (dphix * whc) @ dphix.T + (dphiy * whc) @ dphiy.T

    H = H_square - H_hole
    Qx = Qx_square - Qx_hole
    Qy = Qy_square - Qy_hole
    K = K_square - K_hole

    Sx, Sy = _outer_boundary_forms(basis_x, basis_y, radius, q)
    sx_arc, sy_arc = _arc_boundary(basis_x, basis_y, radius, q)
    Sx += sx_arc
    Sy += sy_arc
    volume_quadrature = None
    if capture_volume_quadrature:
        volume_quadrature = _positive_circular_volume_quadrature(
            basis_x, basis_y, radius, q, compute_basis=capture_basis
        )

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
        quadrature_backend="split_gauss",
        quadrature_order=q,
        structure_mode=structure_mode,
        capture_volume_quadrature=capture_volume_quadrature,
        capture_basis=capture_basis,
    )
    mode = CircularStructureMode(structure_mode)

    return CircularCornerMatrices2D(
        H=H,
        Qx=Qx,
        Qy=Qy,
        K=K,
        H_background=H_background,
        Sx=Sx,
        Sy=Sy,
        nx=basis_x.n_basis,
        ny=basis_y.n_basis,
        cut=cut,
        structure_mode=mode,
        structure_status=structure_status,
        green_diagnostics_x=diagnostics_x,
        green_diagnostics_y=diagnostics_y,
        green_certificate=certificate,
        structure_lineage=lineage,
        volume_quadrature=volume_quadrature,
    )


def circular_corner_active_sets(
    matrices: CircularCornerMatrices2D,
    support_threshold: float = 0.05,
    active_tol: float = 1.0e-12,
) -> CircularCornerActiveSets2D:
    background = np.diag(matrices.H_background)
    physical = np.diag(matrices.H)
    gamma = np.zeros_like(physical)
    positive = background > 0.0
    gamma[positive] = physical[positive] / background[positive]
    active = np.where(gamma > active_tol)[0]
    interior = np.where(gamma >= support_threshold)[0]
    small_support = np.array([i for i in active if gamma[i] < support_threshold], dtype=int)
    return CircularCornerActiveSets2D(active=active, interior=interior, small_support=small_support, gamma=gamma)


def polynomial_preserving_circular_extension(
    basis_x: SplineBasis1D,
    basis_y: SplineBasis1D,
    matrices: CircularCornerMatrices2D,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    active_tol: float = 1.0e-12,
) -> ObliqueAggregation2D:
    sets = circular_corner_active_sets(
        matrices,
        support_threshold=support_threshold,
        active_tol=active_tol,
    )
    if len(sets.interior) == 0:
        raise ValueError("No interior active tensor-product basis functions were found.")
    requested_degree = basis_x.degree if preserve_degree is None else preserve_degree
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

    active_mask = sets.gamma > active_tol
    residuals = [np.linalg.norm(E @ row[sets.interior] - row * active_mask, ord=np.inf) for row in poly]
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


def restrict_circular_corner_2d(
    m: CircularCornerMatrices2D,
    active: np.ndarray,
) -> CircularCornerMatrices2D:
    indices = np.ascontiguousarray(np.asarray(active), dtype=np.dtype("<i8"))
    idx = np.ix_(indices, indices)
    H = m.H[idx]
    Qx = m.Qx[idx]
    Qy = m.Qy[idx]
    K = m.K[idx]
    H_background = m.H_background[idx]
    Sx = m.Sx[idx]
    Sy = m.Sy[idx]
    residual_x = float(np.linalg.norm(Qx + Qx.T - Sx, ord=np.inf))
    residual_y = float(np.linalg.norm(Qy + Qy.T - Sy, ord=np.inf))
    lineage = CircularStructureLineage(
        record_kind="derived",
        parent_operator_hash=m.structure_lineage.child_operator_hash,
        child_operator_hash=_sharp_operator_hash(H, Qx, Qy, K, H_background, Sx, Sy),
        sharp_certificate_hash=m.structure_lineage.sharp_certificate_hash,
        transformation_kind="active_index_restriction",
        transformation_hash=_hash_metadata(_array_descriptor("active", indices, integer=True)),
        measured_green_inf_x=residual_x,
        measured_green_inf_y=residual_y,
    )
    return CircularCornerMatrices2D(
        H=H,
        Qx=Qx,
        Qy=Qy,
        K=K,
        H_background=H_background,
        Sx=Sx,
        Sy=Sy,
        nx=m.nx,
        ny=m.ny,
        cut=m.cut,
        structure_mode=m.structure_mode,
        structure_status=m.structure_status,
        green_diagnostics_x=m.green_diagnostics_x,
        green_diagnostics_y=m.green_diagnostics_y,
        green_certificate=m.green_certificate,
        structure_lineage=lineage,
        volume_quadrature=None,
    )


def pullback_circular_corner_2d(
    m: CircularCornerMatrices2D,
    E: np.ndarray,
) -> CircularCornerMatrices2D:
    transform = np.asarray(E, dtype=float)
    H = transform.T @ m.H @ transform
    Qx = transform.T @ m.Qx @ transform
    Qy = transform.T @ m.Qy @ transform
    K = transform.T @ m.K @ transform
    H_background = transform.T @ m.H_background @ transform
    Sx = transform.T @ m.Sx @ transform
    Sy = transform.T @ m.Sy @ transform
    residual_x = float(np.linalg.norm(Qx + Qx.T - Sx, ord=np.inf))
    residual_y = float(np.linalg.norm(Qy + Qy.T - Sy, ord=np.inf))
    lineage = CircularStructureLineage(
        record_kind="derived",
        parent_operator_hash=m.structure_lineage.child_operator_hash,
        child_operator_hash=_sharp_operator_hash(H, Qx, Qy, K, H_background, Sx, Sy),
        sharp_certificate_hash=m.structure_lineage.sharp_certificate_hash,
        transformation_kind="congruence_pullback",
        transformation_hash=hash_named_float64_matrices(E=transform),
        measured_green_inf_x=residual_x,
        measured_green_inf_y=residual_y,
    )
    return CircularCornerMatrices2D(
        H=H,
        Qx=Qx,
        Qy=Qy,
        K=K,
        H_background=H_background,
        Sx=Sx,
        Sy=Sy,
        nx=m.nx,
        ny=m.ny,
        cut=m.cut,
        structure_mode=m.structure_mode,
        structure_status=m.structure_status,
        green_diagnostics_x=m.green_diagnostics_x,
        green_diagnostics_y=m.green_diagnostics_y,
        green_certificate=m.green_certificate,
        structure_lineage=lineage,
        volume_quadrature=None,
    )


def directional_sbp_residuals_circular(m: CircularCornerMatrices2D) -> tuple[float, float]:
    rx = np.linalg.norm(m.Qx + m.Qx.T - m.Sx, ord=np.inf)
    ry = np.linalg.norm(m.Qy + m.Qy.T - m.Sy, ord=np.inf)
    return float(rx), float(ry)


def mass_condition_number(H: np.ndarray) -> float:
    s = svdvals(0.5 * (H + H.T))
    return float(s[0] / s[-1])


def heat_lambda_max(K: np.ndarray, H: np.ndarray) -> float:
    vals = eigvalsh(0.5 * (K + K.T), 0.5 * (H + H.T))
    return float(vals[-1])


def make_circular_corner_pair(
    n_elements: int = 8,
    degree: int = 3,
    epsilon: float = 0.05,
    support_threshold: float = 0.05,
    preserve_degree: int | None = None,
    quadrature_order: int | None = None,
    structure_mode: CircularStructureMode = CircularStructureMode.RAW,
) -> tuple[
    CircularCornerMatrices2D,
    CircularCornerMatrices2D,
    ObliqueAggregation2D,
    SplineBasis1D,
    SplineBasis1D,
]:
    basis_x = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    basis_y = make_open_uniform_basis(n_elements=n_elements, degree=degree)
    h = 1.0 / n_elements
    radius = sqrt(2.0) * h * (1.0 - epsilon)
    cut = CircularCornerCut(radius=radius)
    m_all = assemble_circular_corner_matrices(
        basis_x,
        basis_y,
        cut,
        quadrature_order=quadrature_order,
        structure_mode=structure_mode,
    )
    if CircularStructureMode(structure_mode) is CircularStructureMode.CERTIFIED:
        require_certified_structure(m_all)
    agg = polynomial_preserving_circular_extension(
        basis_x,
        basis_y,
        m_all,
        support_threshold=support_threshold,
        preserve_degree=preserve_degree,
    )
    full = restrict_circular_corner_2d(m_all, agg.active)
    red = pullback_circular_corner_2d(m_all, agg.E)
    return full, red, agg, basis_x, basis_y
