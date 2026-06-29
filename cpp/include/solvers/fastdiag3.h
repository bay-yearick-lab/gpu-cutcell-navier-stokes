// fastdiag3.h
//
// Fast-diagonalization DIRECT solver for the tensor-product operators of the 3D
// wall-NS loop. Both the pressure Schur complement and the implicit-viscous
// Helmholtz have the form
//
//   ( a M (x) M (x) M  +  b [ A1 (x) M (x) M + M (x) A2 (x) M + M (x) M (x) A3 ] ) phi = rhs,
//
// where A_a is the 1D Schur factor K M^{-1} K^T (pressure) or the stiffness S
// (viscous). Solving the generalized eigenproblem A_a Phi = M_a Phi Lambda once
// per axis (M-orthonormal Phi) diagonalizes the whole 3D operator:
//
//   phi = (Phi1(x)Phi2(x)Phi3) D^{-1} (Phi1^T(x)Phi2^T(x)Phi3^T) rhs,
//   D_ijk = a + b (lambda1_i + lambda2_j + lambda3_k).
//
// One O(N^4) transform each way -- a DIRECT solve, no iteration (vs the
// unpreconditioned CG). This is the Lynch-Rice-Thomas fast diagonalization, the
// standard accelerator for separable tensor-product elliptic solves.

#ifndef FASTDIAG3_H
#define FASTDIAG3_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "linalg.h"

// lower Cholesky M = L L^T (dense SPD)
inline DenseMatrix fd_chol(const DenseMatrix& M) {
    const int n = M.n; DenseMatrix L(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = M(i, j);
            for (int k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            L(i, j) = (i == j) ? std::sqrt(s) : s / L(j, j);
        }
    return L;
}
// symmetric eig of A (dense) by cyclic Jacobi: eigenvalues ev (ascending) and
// eigenvectors as columns of V (A = V diag(ev) V^T).
inline void fd_jacobi(DenseMatrix A, std::vector<double>& ev, DenseMatrix& V) {
    const int n = A.n; V = DenseMatrix(n);
    for (int i = 0; i < n; ++i) V(i, i) = 1.0;
    for (int sweep = 0; sweep < 120; ++sweep) {
        double off = 0;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) off += A(p, q) * A(p, q);
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) {
            if (std::fabs(A(p, q)) < 1e-300) continue;
            const double th = 0.5 * (A(q, q) - A(p, p)) / A(p, q);
            const double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < n; ++k) { const double a1 = A(k, p), a2 = A(k, q); A(k, p) = c * a1 - s * a2; A(k, q) = s * a1 + c * a2; }
            for (int k = 0; k < n; ++k) { const double a1 = A(p, k), a2 = A(q, k); A(p, k) = c * a1 - s * a2; A(q, k) = s * a1 + c * a2; }
            for (int k = 0; k < n; ++k) { const double v1 = V(k, p), v2 = V(k, q); V(k, p) = c * v1 - s * v2; V(k, q) = s * v1 + c * v2; }
        }
    }
    ev.assign(n, 0);
    for (int i = 0; i < n; ++i) ev[i] = A(i, i);
}

// One axis' generalized eigendata: Phi (M-orthonormal eigenvectors), Lambda.
struct AxisEig {
    int n = 0;
    DenseMatrix Phi, PhiT;     // Phi columns are eigenvectors; PhiT = Phi^T
    std::vector<double> lam;
    // generalized eig  A Phi = M Phi Lambda,  Phi^T M Phi = I
    void build(const DenseMatrix& A, const DenseMatrix& M) {
        n = A.n;
        DenseMatrix L = fd_chol(M);
        // C = L^{-1} A L^{-T}
        DenseMatrix Y(n);                          // Y = L^{-1} A
        for (int col = 0; col < n; ++col) for (int i = 0; i < n; ++i) {
            double s = A(i, col); for (int k = 0; k < i; ++k) s -= L(i, k) * Y(k, col); Y(i, col) = s / L(i, i);
        }
        DenseMatrix C(n);                          // C = Y L^{-T}
        for (int row = 0; row < n; ++row) for (int j = 0; j < n; ++j) {
            double s = Y(row, j); for (int k = 0; k < j; ++k) s -= L(j, k) * C(row, k); C(row, j) = s / L(j, j);
        }
        DenseMatrix V; fd_jacobi(C, lam, V);
        // Phi = L^{-T} V   (solve L^T Phi = V)
        Phi = DenseMatrix(n);
        for (int col = 0; col < n; ++col)
            for (int i = n - 1; i >= 0; --i) {
                double s = V(i, col); for (int k = i + 1; k < n; ++k) s -= L(k, i) * Phi(k, col); Phi(i, col) = s / L(i, i);
            }
        PhiT = DenseMatrix(n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) PhiT(i, j) = Phi(j, i);
    }
};

#endif  // FASTDIAG3_H
