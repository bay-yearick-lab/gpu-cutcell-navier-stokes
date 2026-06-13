"""Vectorized B-spline evaluation and local-block cut-cell assembly.

The reference assembly paths in this repository evaluate every basis function
through a per-function Python closure and accumulate full-matrix outer
products per cell, which costs O(n^2 m) per cell for n total basis functions
and m quadrature points even though only (p+1)^d basis functions are nonzero
on any cell.  This module supplies the two ingredients that remove the
bottleneck without changing a single quadrature point:

* ``basis_value_table`` evaluates all basis values or first derivatives on a
  batch of points in one vectorized ``scipy.interpolate.BSpline.design_matrix``
  call (the derivative uses the standard degree-lowering recurrence on the
  same knot vector);
* ``element_support`` returns the contiguous local index range of the
  open-uniform clamped basis on one element, so per-cell assembly touches
  only (p+1)^d rows and columns.

Everything here is exact reorganization: the resulting matrices agree with
the closure-based reference assembly to roundoff.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

from cut_spline_sbp import SplineBasis1D


def basis_value_table(basis: SplineBasis1D, x: np.ndarray, nu: int = 0) -> np.ndarray:
    """Values (nu=0) or first derivatives (nu=1) of all basis functions.

    Returns a dense array of shape (n_basis, len(x)).  Points are clipped to
    the closed parametric interval, matching the nan-to-zero convention of
    the closure-based evaluator.
    """
    t = basis.knots
    k = basis.degree
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros((basis.n_basis, 0))
    xc = np.clip(x, t[k], t[-k - 1])
    if nu == 0:
        dm = BSpline.design_matrix(xc, t, k, extrapolate=False)
        return np.asarray(dm.todense()).T
    if nu != 1:
        raise ValueError("only nu in {0, 1} is supported")
    if k == 0:
        return np.zeros((basis.n_basis, x.size))
    # At the exact right endpoint the degree-lowered design matrix lands on
    # the degenerate end-knot interval; the derivative is a polynomial on the
    # last element, so the one-ulp left nudge is exact to roundoff.
    right = t[-k - 1]
    xc = np.where(xc >= right, np.nextafter(right, t[k]), xc)
    # B'_{i,k} = k * ( B_{i,k-1}/(t_{i+k}-t_i) - B_{i+1,k-1}/(t_{i+k+1}-t_{i+1}) )
    # with the degree-(k-1) family taken on the same knot vector.
    low = np.asarray(BSpline.design_matrix(xc, t, k - 1, extrapolate=False).todense())
    n = basis.n_basis
    d = np.zeros((x.size, n))
    denom_a = t[k : k + n] - t[:n]
    denom_b = t[k + 1 : k + 1 + n] - t[1 : 1 + n]
    with np.errstate(divide="ignore", invalid="ignore"):
        coef_a = np.where(denom_a > 0.0, k / np.where(denom_a > 0.0, denom_a, 1.0), 0.0)
        coef_b = np.where(denom_b > 0.0, k / np.where(denom_b > 0.0, denom_b, 1.0), 0.0)
    d += low[:, :n] * coef_a[None, :]
    d -= low[:, 1 : 1 + n] * coef_b[None, :]
    return d.T


def element_support(basis: SplineBasis1D, element: int) -> tuple[int, int]:
    """Local basis index range [lo, hi) active on one element.

    For the open-uniform clamped basis the nonzero functions on element e are
    exactly e, ..., e+p.
    """
    p = basis.degree
    return element, element + p + 1


def local_tables(
    basis: SplineBasis1D, element: int, x: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """Values and derivatives of the local basis functions on one element.

    Returns (lo, values, derivatives) with arrays of shape (p+1, len(x)).
    """
    lo, hi = element_support(basis, element)
    vals = basis_value_table(basis, x, nu=0)[lo:hi]
    ders = basis_value_table(basis, x, nu=1)[lo:hi]
    return lo, vals, ders


class TripletAccumulator:
    """COO triplet accumulator for symmetric local-block assembly."""

    def __init__(self, n: int):
        self.n = int(n)
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []

    def add_block(self, rows: np.ndarray, cols: np.ndarray, block: np.ndarray) -> None:
        r = np.repeat(rows, len(cols))
        c = np.tile(cols, len(rows))
        self._rows.append(r)
        self._cols.append(c)
        self._vals.append(np.asarray(block, dtype=float).reshape(-1))

    def tocsr(self) -> sp.csr_matrix:
        if not self._rows:
            return sp.csr_matrix((self.n, self.n))
        rows = np.concatenate(self._rows)
        cols = np.concatenate(self._cols)
        vals = np.concatenate(self._vals)
        return sp.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()

    def todense(self) -> np.ndarray:
        return np.asarray(self.tocsr().todense())
