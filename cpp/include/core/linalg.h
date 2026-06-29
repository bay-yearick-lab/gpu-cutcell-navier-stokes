// linalg.h
//
// Small, self-contained linear algebra for the B-spline operators.
//
// Everything here is "C-style C++": plain structs, std::vector, no exceptions,
// no templates, no clever metaprogramming. If you know C and Fortran this will
// read naturally.
//
// Three things live here:
//
//   1. DenseMatrix   -- a square n x n matrix stored row-major. Used to build
//                       and inspect the operators B, B', B'' and to apply them
//                       (the matrix-vector product u' = B' * alpha).
//
//   2. BandSolver    -- a banded LU factorization (no pivoting) plus solve.
//                       This is the fast, memory-light way to solve B alpha = u
//                       for a clamped (boundary) operator, which is banded with
//                       half-bandwidth <= p. The factor is computed ONCE and
//                       reused for every right-hand side (every grid line, every
//                       time step). That reuse is the whole speed story.
//
//   3. CyclicSolver  -- a banded solve for the PERIODIC case, where the operator
//                       is banded everywhere except two small corner blocks that
//                       "wrap around". We handle the wrap with the standard
//                       Sherman-Morrison-Woodbury correction on top of BandSolver.
//
// Why banded and not a general sparse library (CSR, etc.)? Because these
// operators are banded with a small, fixed bandwidth. Band storage is
// contiguous and cache-friendly, the factorization is O(n * bandwidth^2), and
// there is no index indirection to slow the inner loops. A general sparse
// solver would be both more complex and slower for this regular structure.
//
// Why no pivoting in the band LU? B-spline collocation matrices at the Greville
// points are totally positive (de Boor, "A Practical Guide to Splines"), so
// Gaussian elimination without row swaps is stable and keeps the bandwidth
// fixed. We still report the residual ||A x - b|| so a bad case can never pass
// silently.

#ifndef LINALG_H
#define LINALG_H

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>

// ---------------------------------------------------------------------------
// DenseMatrix: an n x n matrix, row-major. a[i*n + j] is the (i, j) entry.
// ---------------------------------------------------------------------------
struct DenseMatrix {
    int n = 0;
    std::vector<double> a;

    DenseMatrix() {}
    explicit DenseMatrix(int n_) : n(n_), a(static_cast<size_t>(n_) * n_, 0.0) {}

    double&       operator()(int i, int j)       { return a[static_cast<size_t>(i) * n + j]; }
    double        operator()(int i, int j) const { return a[static_cast<size_t>(i) * n + j]; }

    // Matrix-vector product y = A x. This is how we APPLY an operator, e.g.
    // u' = B' alpha. (For a banded operator most entries are zero; a banded
    // mat-vec would skip them. Dense is used here for clarity; switch to a
    // banded apply when n gets large.)
    std::vector<double> matvec(const std::vector<double>& x) const {
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            const double* row = &a[static_cast<size_t>(i) * n];
            for (int j = 0; j < n; ++j) sum += row[j] * x[j];
            y[i] = sum;
        }
        return y;
    }
};

// ---------------------------------------------------------------------------
// Dense LU with partial pivoting. Small and robust. We use it for two jobs:
//   - solving the tiny correction system inside CyclicSolver, and
//   - cross-checking the fast banded/cyclic solvers in the tests.
// It overwrites a local copy, so the caller's matrix is untouched.
// Returns false if the matrix is singular.
// ---------------------------------------------------------------------------
inline bool dense_lu_solve(const DenseMatrix& A_in, const std::vector<double>& b_in,
                           std::vector<double>& x)
{
    const int n = A_in.n;
    DenseMatrix A = A_in;            // working copy
    std::vector<double> b = b_in;
    std::vector<int> piv(n);
    for (int i = 0; i < n; ++i) piv[i] = i;

    for (int k = 0; k < n; ++k) {
        // Find the largest pivot in column k (partial pivoting).
        int p = k;
        double big = std::fabs(A(k, k));
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(A(i, k));
            if (v > big) { big = v; p = i; }
        }
        if (big == 0.0) return false;       // singular
        // Swap rows k and p (in A and in b).
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(A(k, j), A(p, j));
            std::swap(b[k], b[p]);
        }
        // Eliminate below the pivot.
        const double inv = 1.0 / A(k, k);
        for (int i = k + 1; i < n; ++i) {
            const double f = A(i, k) * inv;
            A(i, k) = 0.0;
            for (int j = k + 1; j < n; ++j) A(i, j) -= f * A(k, j);
            b[i] -= f * b[k];
        }
    }
    // Back substitution.
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) sum -= A(i, j) * x[j];
        x[i] = sum / A(i, i);
    }
    return true;
}

// ---------------------------------------------------------------------------
// BandSolver: banded LU (no pivoting) for a matrix with kl sub-diagonals and
// ku super-diagonals.
//
// Storage: each row i keeps only the (kl + ku + 1) entries A(i, i-kl .. i+ku).
// The entry A(i, j) lives at band[i*w + (j - i + kl)] with w = kl + ku + 1.
// Out-of-range columns are simply not stored.
//
// factor() does in-place LU. Because we do not pivot, L keeps kl sub-diagonals
// and U keeps ku super-diagonals: no fill outside the band. solve() then does a
// forward sweep (unit-lower L) and a back substitution (upper U).
// ---------------------------------------------------------------------------
struct BandSolver {
    int n = 0, kl = 0, ku = 0, w = 0;
    std::vector<double> band;       // n * w, row-major in band coordinates
    bool ok = false;

    BandSolver() {}

    // Build the band layout from a dense matrix, keeping diagonals -kl..+ku.
    void set_from_dense(const DenseMatrix& A, int kl_, int ku_) {
        n = A.n; kl = kl_; ku = ku_; w = kl + ku + 1;
        band.assign(static_cast<size_t>(n) * w, 0.0);
        for (int i = 0; i < n; ++i) {
            const int jlo = (i - kl > 0) ? i - kl : 0;
            const int jhi = (i + ku < n - 1) ? i + ku : n - 1;
            for (int j = jlo; j <= jhi; ++j) at(i, j) = A(i, j);
        }
    }

    double& at(int i, int j) { return band[static_cast<size_t>(i) * w + (j - i + kl)]; }
    double  at(int i, int j) const { return band[static_cast<size_t>(i) * w + (j - i + kl)]; }

    // In-place banded LU without pivoting.
    void factor() {
        ok = true;
        for (int i = 0; i < n; ++i) {
            const double piv = at(i, i);
            if (piv == 0.0) { ok = false; return; }
            const double inv = 1.0 / piv;
            const int rmax = (i + kl < n - 1) ? i + kl : n - 1;   // last row touched
            const int cmax = (i + ku < n - 1) ? i + ku : n - 1;   // last col touched
            for (int r = i + 1; r <= rmax; ++r) {
                const double f = at(r, i) * inv;
                at(r, i) = f;                                     // store multiplier (L)
                for (int c = i + 1; c <= cmax; ++c) at(r, c) -= f * at(i, c);
            }
        }
    }

    // Solve A x = b in place (x holds b on entry, the solution on exit). No
    // allocation: this is the hot path reused across every line and time step.
    void solve_inplace(std::vector<double>& x) const {
        // Forward sweep: L y = b. L has unit diagonal; multipliers are at(i, i-kl..i-1).
        for (int i = 0; i < n; ++i) {
            const int jlo = (i - kl > 0) ? i - kl : 0;
            double sum = x[i];
            for (int j = jlo; j < i; ++j) sum -= at(i, j) * x[j];
            x[i] = sum;
        }
        // Back substitution: U x = y. U has entries at(i, i..i+ku).
        for (int i = n - 1; i >= 0; --i) {
            const int jhi = (i + ku < n - 1) ? i + ku : n - 1;
            double sum = x[i];
            for (int j = i + 1; j <= jhi; ++j) sum -= at(i, j) * x[j];
            x[i] = sum / at(i, i);
        }
    }

    // Solve A x = b using the stored factors. Returns x.
    std::vector<double> solve(const std::vector<double>& b) const {
        std::vector<double> x(b);
        solve_inplace(x);
        return x;
    }

    // Solve W systems at once. X is n x W, row-major (X[i*W + l] is point i of
    // system l). The inner loop over the W lanes is unit-stride and vectorizes
    // (AVX2 = 4 doubles), and W adjacent lines gathered from a 3D field share
    // cache lines, so this is both the SIMD and the cache-efficiency win.
    void solve_block(double* X, int W) const {
        for (int i = 0; i < n; ++i) {
            const int jlo = (i - kl > 0) ? i - kl : 0;
            for (int j = jlo; j < i; ++j) {
                const double f = at(i, j);
                const double* xj = X + static_cast<size_t>(j) * W;
                double* xi = X + static_cast<size_t>(i) * W;
                for (int l = 0; l < W; ++l) xi[l] -= f * xj[l];
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            const int jhi = (i + ku < n - 1) ? i + ku : n - 1;
            double* xi = X + static_cast<size_t>(i) * W;
            for (int j = i + 1; j <= jhi; ++j) {
                const double f = at(i, j);
                const double* xj = X + static_cast<size_t>(j) * W;
                for (int l = 0; l < W; ++l) xi[l] -= f * xj[l];
            }
            const double d = 1.0 / at(i, i);
            for (int l = 0; l < W; ++l) xi[l] *= d;
        }
    }
};

// ---------------------------------------------------------------------------
// CyclicSolver: solve A x = b when A is "banded with wrap-around", i.e. the
// periodic case. A is banded with half-bandwidth b everywhere except two b x b
// corner blocks (top-right and bottom-left) that couple the two ends of the
// domain.
//
// Idea (Sherman-Morrison-Woodbury): write A = C + correction, where C is the
// plain banded matrix you get by deleting the corner (wrap) entries, and the
// correction lives only on the boundary index set
//     S = {0,1,...,b-1} U {n-b,...,n-1}   (size m = 2b).
// Both i and j of every wrap entry lie in S, so the correction is exactly the
// m x m block  Dss = A[S,S] - C[S,S]. Then
//
//     A = C + E_S Dss E_S^T,    E_S = columns of identity at indices S,
//
// and Woodbury gives, with y = C^{-1} b,
//
//     x = y - Z * ( M^{-1} ( Dss * y[S] ) ),
//
// where Z = C^{-1} E_S (one banded solve per boundary index) and
// M = I + Dss * Z[S,:] (a tiny m x m matrix). C, Z, and M are all built ONCE in
// the constructor; each solve() is one banded solve plus O(m^2) work.
// ---------------------------------------------------------------------------
struct CyclicSolver {
    int n = 0, b = 0, m = 0;
    BandSolver core;                 // C: banded part, factored
    std::vector<int> S;              // boundary indices, size m
    std::vector<double> Z;           // n x m, column-major: Z[c*n + i] = (C^{-1} e_{S[c]})_i
    DenseMatrix M;                   // m x m correction matrix (stored dense)
    std::vector<double> Dss;         // m x m defect block, row-major

    void build_index_set() {                 // S = {0..b-1} U {n-b..n-1}
        m = 2 * b;
        S.clear();
        for (int i = 0; i < b; ++i)      S.push_back(i);
        for (int i = n - b; i < n; ++i)  S.push_back(i);
    }

    // Given a factored core, the index set S, and the defect Dss, build the
    // Woodbury pieces Z = C^{-1} E_S and M = I + Dss Z[S,:]. Shared by both the
    // dense and the stencil constructors.
    void finalize() {
        Z.assign(static_cast<size_t>(m) * n, 0.0);
        std::vector<double> e(n, 0.0);
        for (int c = 0; c < m; ++c) {
            std::fill(e.begin(), e.end(), 0.0);
            e[S[c]] = 1.0;
            std::vector<double> z = core.solve(e);
            for (int i = 0; i < n; ++i) Z[static_cast<size_t>(c) * n + i] = z[i];
        }
        M = DenseMatrix(m);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c) {
                double sum = (r == c) ? 1.0 : 0.0;
                for (int k = 0; k < m; ++k)
                    sum += Dss[r * m + k] * Z[static_cast<size_t>(c) * n + S[k]];
                M(r, c) = sum;
            }
    }

    // Build from a dense periodic matrix A with half-bandwidth b_.
    void set_from_dense(const DenseMatrix& A, int b_) {
        n = A.n; b = b_; build_index_set();

        // Core C = A with the wrap (corner) entries removed.
        DenseMatrix C = A;
        for (int i = 0; i < b; ++i)
            for (int j = n - b; j < n; ++j) C(i, j) = 0.0;       // top-right block
        for (int i = n - b; i < n; ++i)
            for (int j = 0; j < b; ++j)     C(i, j) = 0.0;       // bottom-left block

        Dss.assign(static_cast<size_t>(m) * m, 0.0);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c)
                Dss[r * m + c] = A(S[r], S[c]) - C(S[r], S[c]);

        core.set_from_dense(C, b, b);
        core.factor();
        finalize();
    }

    // Build directly from a centered circulant stencil st[o + b], o in [-b, b],
    // so the entry A(i, j) = st[(j-i) + b] when |j-i| <= b (modulo n). No dense
    // n x n matrix is ever formed: storage is O(n*b). This is the production
    // path that lets a single line grow to millions of points.
    void set_from_stencil(int n_, int b_, const std::vector<double>& st) {
        n = n_; b = b_; build_index_set();

        // Toeplitz band core: the same in-range band that set_from_dense keeps,
        // with the wrap entries dropped. Fill the band storage directly.
        core.n = n; core.kl = b; core.ku = b; core.w = 2 * b + 1;
        core.band.assign(static_cast<size_t>(n) * core.w, 0.0);
        for (int i = 0; i < n; ++i)
            for (int o = -b; o <= b; ++o) {
                const int j = i + o;
                if (j >= 0 && j < n) core.at(i, j) = st[o + b];
            }
        core.factor();

        // Dss = wrap entries only: circulant value minus the in-range band value.
        auto circ = [&](int i, int j) {
            int off = ((j - i) % n + n) % n;
            if (off > n - off) off -= n;                 // minimal signed offset
            return (off >= -b && off <= b) ? st[off + b] : 0.0;
        };
        auto band = [&](int i, int j) {
            const int d = j - i;
            return (d >= -b && d <= b) ? st[d + b] : 0.0;
        };
        Dss.assign(static_cast<size_t>(m) * m, 0.0);
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c)
                Dss[r * m + c] = circ(S[r], S[c]) - band(S[r], S[c]);
        finalize();
    }

    // Solve A x = b in place. Only the tiny m-vectors (m = 2b) allocate; the
    // O(n) work writes x directly. This is the periodic hot path.
    void solve_inplace(std::vector<double>& x) const {
        core.solve_inplace(x);                       // x = C^{-1} b
        std::vector<double> ys(m), rhs(m), wv;
        for (int r = 0; r < m; ++r) ys[r] = x[S[r]];
        for (int r = 0; r < m; ++r) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) sum += Dss[r * m + k] * ys[k];
            rhs[r] = sum;
        }
        dense_lu_solve(M, rhs, wv);                  // w = M^{-1} rhs   (m x m, tiny)
        for (int c = 0; c < m; ++c) {                // x -= Z w
            const double wc = wv[c];
            const double* zc = &Z[static_cast<size_t>(c) * n];
            for (int i = 0; i < n; ++i) x[i] -= zc[i] * wc;
        }
    }

    std::vector<double> solve(const std::vector<double>& bvec) const {
        std::vector<double> x = bvec;
        solve_inplace(x);
        return x;
    }

    // Solve W systems at once (X is n x W, row-major). Banded core solve and the
    // Woodbury correction both vectorize over the W lanes.
    void solve_block(double* X, int W) const {
        core.solve_block(X, W);                          // x = C^{-1} b for each lane
        std::vector<double> ys(static_cast<size_t>(m) * W, 0.0);
        std::vector<double> rhs(static_cast<size_t>(m) * W, 0.0);
        std::vector<double> wv(static_cast<size_t>(m) * W, 0.0);
        for (int r = 0; r < m; ++r)
            for (int l = 0; l < W; ++l) ys[r * W + l] = X[static_cast<size_t>(S[r]) * W + l];
        for (int r = 0; r < m; ++r)
            for (int k = 0; k < m; ++k) {
                const double d = Dss[r * m + k];
                for (int l = 0; l < W; ++l) rhs[r * W + l] += d * ys[k * W + l];
            }
        std::vector<double> rl(m), wl;
        for (int l = 0; l < W; ++l) {                    // tiny m x m solve per lane
            for (int r = 0; r < m; ++r) rl[r] = rhs[r * W + l];
            dense_lu_solve(M, rl, wl);
            for (int r = 0; r < m; ++r) wv[r * W + l] = wl[r];
        }
        for (int c = 0; c < m; ++c) {                    // X -= Z w
            const double* zc = &Z[static_cast<size_t>(c) * n];
            const double* wc = &wv[static_cast<size_t>(c) * W];
            for (int i = 0; i < n; ++i) {
                const double zci = zc[i];
                double* xi = X + static_cast<size_t>(i) * W;
                for (int l = 0; l < W; ++l) xi[l] -= zci * wc[l];
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Mat: a general (possibly rectangular) r x c dense matrix, row-major.
//
// The mimetic operators are built by composing the spline building blocks
// B, M, K, the derivative-coefficient map, and so on. Some of those pieces are
// rectangular (the BMF staggered gradient maps R^N -> R^{N-1}), so the square
// DenseMatrix above is not enough. Mat carries the explicit (r, c) shape and a
// small algebra: multiply, transpose, inverse, matvec.
//
// These are DENSE O(n^3) routines used to ASSEMBLE and VERIFY the operators on
// the modest grids used in the tests and to materialize an operator for
// inspection. The production solver never forms these matrices: it applies the
// reduce-solve-expand chain through the banded solvers above.
// ---------------------------------------------------------------------------
struct Mat {
    int r = 0, c = 0;
    std::vector<double> a;            // r * c, row-major

    Mat() {}
    Mat(int r_, int c_) : r(r_), c(c_), a(static_cast<size_t>(r_) * c_, 0.0) {}

    double&       operator()(int i, int j)       { return a[static_cast<size_t>(i) * c + j]; }
    double        operator()(int i, int j) const { return a[static_cast<size_t>(i) * c + j]; }
};

// Copy a square DenseMatrix into a Mat (so the existing B, B', B'' compose with
// the new algebra).
inline Mat to_mat(const DenseMatrix& D) {
    Mat A(D.n, D.n);
    for (int i = 0; i < D.n; ++i)
        for (int j = 0; j < D.n; ++j) A(i, j) = D(i, j);
    return A;
}

inline Mat identity_mat(int n) {
    Mat I(n, n);
    for (int i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

// Copy a square Mat back into a DenseMatrix (so the band/cyclic solvers, which
// take a DenseMatrix, can be built from an assembled Mat such as the mass form).
inline DenseMatrix to_dense(const Mat& A) {
    DenseMatrix D(A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j) D(i, j) = A(i, j);
    return D;
}

// C = A * B, with A (r x k) and B (k x c). The loop order (i, l, j) keeps the
// inner stride-1 over j for both B and C, which is the cache-friendly order.
inline Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.r, B.c);
    for (int i = 0; i < A.r; ++i) {
        for (int l = 0; l < A.c; ++l) {
            const double ail = A(i, l);
            if (ail == 0.0) continue;
            const double* brow = &B.a[static_cast<size_t>(l) * B.c];
            double* crow = &C.a[static_cast<size_t>(i) * B.c];
            for (int j = 0; j < B.c; ++j) crow[j] += ail * brow[j];
        }
    }
    return C;
}

inline Mat transpose(const Mat& A) {
    Mat T(A.c, A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j) T(j, i) = A(i, j);
    return T;
}

inline std::vector<double> matvec(const Mat& A, const std::vector<double>& x) {
    std::vector<double> y(A.r, 0.0);
    for (int i = 0; i < A.r; ++i) {
        const double* row = &A.a[static_cast<size_t>(i) * A.c];
        double s = 0.0;
        for (int j = 0; j < A.c; ++j) s += row[j] * x[j];
        y[i] = s;
    }
    return y;
}

// Dense inverse of a square matrix by Gauss-Jordan with partial pivoting.
// Returns the n x n inverse; sets *ok=false (if provided) on a singular matrix.
inline Mat inverse(const Mat& A_in, bool* ok = nullptr) {
    const int n = A_in.r;
    Mat A = A_in;                      // working copy, augmented mentally with I
    Mat X = identity_mat(n);
    if (ok) *ok = true;
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double big = std::fabs(A(k, k));
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(A(i, k));
            if (v > big) { big = v; piv = i; }
        }
        if (big == 0.0) { if (ok) *ok = false; return X; }
        if (piv != k) {
            for (int j = 0; j < n; ++j) { std::swap(A(k, j), A(piv, j)); std::swap(X(k, j), X(piv, j)); }
        }
        const double inv = 1.0 / A(k, k);
        for (int j = 0; j < n; ++j) { A(k, j) *= inv; X(k, j) *= inv; }
        for (int i = 0; i < n; ++i) {
            if (i == k) continue;
            const double f = A(i, k);
            if (f == 0.0) continue;
            for (int j = 0; j < n; ++j) { A(i, j) -= f * A(k, j); X(i, j) -= f * X(k, j); }
        }
    }
    return X;
}

// Largest absolute entry of A - B (same shape). Used to verify matrix identities.
inline double max_abs_diff(const Mat& A, const Mat& B) {
    double e = 0.0;
    for (size_t i = 0; i < A.a.size(); ++i) e = std::fmax(e, std::fabs(A.a[i] - B.a[i]));
    return e;
}

inline double max_abs(const Mat& A) {
    double e = 0.0;
    for (double v : A.a) e = std::fmax(e, std::fabs(v));
    return e;
}

// ---------------------------------------------------------------------------
// Small helper: the largest absolute difference between two vectors. Used all
// over the tests to report errors.
// ---------------------------------------------------------------------------
inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i) e = std::fmax(e, std::fabs(a[i] - b[i]));
    return e;
}

#endif // LINALG_H
