// bspline.h
//
// Build the three B-spline operator matrices B, B', B'' of arbitrary polynomial
// degree p, in two flavors:
//
//   PERIODIC : a uniform grid x_j = j*h on a periodic domain. The matrices are
//              circulant (every row is a shift of the same stencil) because the
//              basis is the centered cardinal B-spline, which is translation
//              invariant.
//
//   CLAMPED  : a bounded domain [0, L] with a clamped (open-uniform) knot vector
//              and the collocation nodes placed at the true Greville points.
//              The matrices are banded (half-bandwidth <= p) with denser rows
//              near the two walls.
//
// In both cases the meaning is the same. If alpha is the vector of spline
// coefficients and the nodes are x_0..x_{n-1}, then
//
//     u(x_k)        = sum_i B(k,i)   alpha_i        ->   u   = B   alpha
//     u'(x_k)       = sum_i B'(k,i)  alpha_i        ->   u'  = B'  alpha
//     u''(x_k)      = sum_i B''(k,i) alpha_i        ->   u'' = B'' alpha
//
// So to differentiate a sampled field u you first solve B alpha = u for the
// coefficients (one banded solve), then apply B' or B''. That is exactly the
// manufactured-solution check in mms_check.cpp.
//
// The only basis-evaluation engine used here is the standard de Boor / Cox
// recurrence. Nothing exotic.

#ifndef BSPLINE_H
#define BSPLINE_H

#include <cmath>
#include <vector>
#include "linalg.h"

// ===========================================================================
// Part 1. Cardinal B-spline on integer knots (used to build the PERIODIC case)
// ===========================================================================
//
// N_{i,k}(u) is the degree-k B-spline on the integer knot vector t_j = j,
// evaluated by the Cox-de Boor recurrence. On integer knots the two
// denominators are both equal to k, which keeps the recurrence clean.
//
// This is the same Cox-de Boor recurrence as the textbook recursive definition,
// but filled iteratively on the small triangular dependency table. That matters
// once we push p toward the spectral-like regime: the naive recursion repeats
// the same subproblems exponentially many times.
inline double cardinal_N(int i, int k, double u) {
    std::vector<double> N(k + 1, 0.0);
    for (int j = 0; j <= k; ++j) {
        const double left = static_cast<double>(i + j);
        N[j] = (u >= left && u < left + 1.0) ? 1.0 : 0.0;
    }
    for (int d = 1; d <= k; ++d) {
        for (int j = 0; j <= k - d; ++j) {
            const double jj = static_cast<double>(i + j);
            const double left  = (u - jj) / d;
            const double right = (jj + d + 1.0 - u) / d;
            N[j] = left * N[j] + right * N[j + 1];
        }
    }
    return N[0];
}

// The centered cardinal B-spline beta_p and its derivatives, as a function of a
// real offset x. "Centered" means the peak sits at x = 0 and the support is
// [-(p+1)/2, (p+1)/2]. We get it by shifting N_{0,p} so its center lands at 0.
//
// The derivative formulas come from the standard B-spline derivative rule
//     d/du N_{i,p} = N_{i,p-1} - N_{i+1,p-1}     (on integer knots),
// applied once for beta' and twice for beta''.
inline double beta_value(int p, double x) {
    return cardinal_N(0, p, x + 0.5 * (p + 1));
}
inline double beta_deriv1(int p, double x) {
    const double u = x + 0.5 * (p + 1);
    return cardinal_N(0, p - 1, u) - cardinal_N(1, p - 1, u);
}
inline double beta_deriv2(int p, double x) {
    const double u = x + 0.5 * (p + 1);
    return cardinal_N(0, p - 2, u) - 2.0 * cardinal_N(1, p - 2, u) + cardinal_N(2, p - 2, u);
}

// ===========================================================================
// Part 2. General B-spline basis with derivatives (used to build the CLAMPED case)
// ===========================================================================
//
// For an arbitrary (here: clamped open-uniform) knot vector we need the basis
// functions and their first two derivatives at a query point. These are the
// textbook FindSpan and DersBasisFuns routines (Piegl & Tiller, "The NURBS
// Book", algorithms A2.1 and A2.3). They are written plainly below.

// Find the knot span index s such that knots[s] <= u < knots[s+1], with the
// usual special handling of the right endpoint. n_last = (number of basis
// functions) - 1.
inline int find_span(int n_last, int p, double u, const std::vector<double>& knots) {
    if (u >= knots[n_last + 1]) return n_last;     // right endpoint
    if (u <= knots[p])          return p;          // left endpoint
    int low = p, high = n_last + 1, mid = (low + high) / 2;
    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) high = mid; else low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

// Compute the (p+1) nonzero basis functions and their derivatives up to order
// nders at u, given the span. ders[d][j] = d-th derivative of the local basis
// function with global index (span - p + j), for d = 0..nders, j = 0..p.
inline void ders_basis_funs(int span, double u, int p, int nders,
                            const std::vector<double>& knots,
                            std::vector<std::vector<double>>& ders) {
    std::vector<std::vector<double>> ndu(p + 1, std::vector<double>(p + 1, 0.0));
    std::vector<double> left(p + 1, 0.0), right(p + 1, 0.0);
    std::vector<std::vector<double>> a(2, std::vector<double>(p + 1, 0.0));

    ndu[0][0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j]  = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];        // lower triangle: denominators
            const double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;       // upper triangle: basis values
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    ders.assign(nders + 1, std::vector<double>(p + 1, 0.0));
    for (int j = 0; j <= p; ++j) ders[0][j] = ndu[j][p];   // the basis values themselves

    // Compute the derivatives via the recurrence over the triangular table.
    for (int r = 0; r <= p; ++r) {
        int s1 = 0, s2 = 1;
        a[0][0] = 1.0;
        for (int k = 1; k <= nders; ++k) {
            double d = 0.0;
            const int rk = r - k, pk = p - k;
            if (r >= k) {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (r - 1 <= pk) ? k - 1 : p - r;
            for (int j = j1; j <= j2; ++j) {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d += a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk) {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            std::swap(s1, s2);
        }
    }
    // Multiply by the falling factorial p, p(p-1), ... that the recurrence omits.
    int fac = p;
    for (int k = 1; k <= nders; ++k) {
        for (int j = 0; j <= p; ++j) ders[k][j] *= fac;
        fac *= (p - k);
    }
}

// Clamped open-uniform knot vector on [0, L]: the value 0 repeated p+1 times,
// then (npts - p - 1) evenly spaced interior knots, then L repeated p+1 times.
// Length = npts + p + 1, giving exactly npts basis functions.
inline std::vector<double> open_uniform_knots(int npts, int p, double L) {
    std::vector<double> knots;
    knots.reserve(npts + p + 1);
    for (int i = 0; i <= p; ++i) knots.push_back(0.0);
    const int interior = npts - p - 1;
    for (int i = 1; i <= interior; ++i)
        knots.push_back(L * static_cast<double>(i) / (interior + 1));
    for (int i = 0; i <= p; ++i) knots.push_back(L);
    return knots;
}

// True Greville points: xi_i = average of knots[i+1 .. i+p]. These are the
// natural collocation nodes for a clamped B-spline space; the endpoints are
// pinned exactly to the domain ends.
inline std::vector<double> greville_points(const std::vector<double>& knots, int p, int npts) {
    std::vector<double> xi(npts, 0.0);
    for (int i = 0; i < npts; ++i) {
        if (p == 0) { xi[i] = knots[i]; continue; }
        double sum = 0.0;
        for (int k = i + 1; k <= i + p; ++k) sum += knots[k];
        xi[i] = sum / p;
    }
    xi[0]        = knots.front();
    xi[npts - 1] = knots.back();
    return xi;
}

// ===========================================================================
// Part 3. Operator assembly
// ===========================================================================
//
// A small bundle holding the node coordinates and the three operators. "half_bw"
// is the half-bandwidth used by the banded/cyclic solver; "periodic" records
// which solve path to take.
struct SplineOperators {
    int n = 0;
    int degree = 0;
    bool periodic = false;
    int half_bw = 0;                 // half-bandwidth of B (for the solver)
    double h = 0.0;                  // grid spacing (periodic) or reference spacing
    std::vector<double> nodes;       // x_0 .. x_{n-1}
    DenseMatrix B, Bp, Bpp;          // value, first-derivative, second-derivative
};

// Build the periodic operators on a uniform grid of n points spanning [0, L).
// The grid spacing is h = L/n and x_j = j*h. Each matrix is circulant: row i is
// the centered cardinal stencil placed at column i, wrapping around modulo n.
inline SplineOperators build_periodic(int n, int p, double L) {
    SplineOperators op;
    op.n = n; op.degree = p; op.periodic = true;
    op.h = L / n;
    op.nodes.resize(n);
    for (int i = 0; i < n; ++i) op.nodes[i] = i * op.h;

    op.B   = DenseMatrix(n);
    op.Bp  = DenseMatrix(n);
    op.Bpp = DenseMatrix(n);

    // The centered cardinal B-spline has support radius (p+1)/2 grid cells.
    const int radius = (p + 1) / 2;
    op.half_bw = radius;
    const double inv_h  = 1.0 / op.h;
    const double inv_h2 = inv_h * inv_h;

    std::vector<double> b0(2 * radius + 1), b1(2 * radius + 1), b2(2 * radius + 1);
    for (int off = -radius; off <= radius; ++off) {
        // The kernel argument is (x_i - x_j)/h = i - j = -off. This sign only
        // matters for the first derivative, whose kernel beta' is an ODD
        // function; the value and second-derivative kernels are even.
        const double t = static_cast<double>(-off);
        const int k = off + radius;
        b0[k] = beta_value(p, t);
        b1[k] = (p >= 1) ? beta_deriv1(p, t) * inv_h  : 0.0;
        b2[k] = (p >= 2) ? beta_deriv2(p, t) * inv_h2 : 0.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int off = -radius; off <= radius; ++off) {
            const int j = ((i + off) % n + n) % n;        // column j = i + off (mod n)
            const int k = off + radius;
            op.B(i, j)   += b0[k];
            op.Bp(i, j)  += b1[k];
            op.Bpp(i, j) += b2[k];
        }
    }
    return op;
}

// Build the clamped operators on [0, L] with Greville collocation nodes.
// B(k,i) = N_i(xi_k), Bp(k,i) = N_i'(xi_k), Bpp(k,i) = N_i''(xi_k). Each node
// only sees the p+1 basis functions nonzero on its span, so the matrices come
// out banded directly.
inline SplineOperators build_clamped(int npts, int p, double L) {
    SplineOperators op;
    op.n = npts; op.degree = p; op.periodic = false;
    op.half_bw = p;

    std::vector<double> knots = open_uniform_knots(npts, p, L);
    op.nodes = greville_points(knots, p, npts);
    op.h = (npts > 1) ? (op.nodes[1] - op.nodes[0]) : L;   // reference spacing

    op.B   = DenseMatrix(npts);
    op.Bp  = DenseMatrix(npts);
    op.Bpp = DenseMatrix(npts);

    const int nders = (p >= 2) ? 2 : p;     // ask for as many derivatives as exist
    std::vector<std::vector<double>> ders;
    for (int k = 0; k < npts; ++k) {
        const double xk = op.nodes[k];
        const int span = find_span(npts - 1, p, xk, knots);
        ders_basis_funs(span, xk, p, nders, knots, ders);
        for (int j = 0; j <= p; ++j) {
            const int col = span - p + j;
            op.B(k, col)  = ders[0][j];
            if (p >= 1) op.Bp(k, col)  = ders[1][j];
            if (p >= 2) op.Bpp(k, col) = ders[2][j];
        }
    }
    return op;
}

// ===========================================================================
// Part 4. Coefficient bilinear forms M, K, S (Galerkin building blocks)
// ===========================================================================
//
// The mimetic Galerkin route needs the exact continuous forms in coefficient
// space (identical convention in both papers):
//
//     M_{ij} = \int N_i N_j  dx   (mass)
//     K_{ij} = \int N_i N_j' dx   (derivative form)
//     S_{ij} = \int N_i' N_j' dx  (stiffness;  \int N_i N_j'' = -S in the interior)
//
// They are assembled exactly by Gauss-Legendre quadrature on each knot span:
// the integrand is a product of two degree-p (or degree-(p-1)) polynomials, so
// an (p+1)-point rule is exact.

// Gauss-Legendre nodes x and weights w on [-1, 1], computed by Newton iteration
// on the Legendre polynomial. Works for arbitrary order, so the quadrature is
// exact for arbitrary spline degree.
inline void gauss_legendre(int npts, std::vector<double>& x, std::vector<double>& w) {
    x.assign(npts, 0.0);
    w.assign(npts, 0.0);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < npts; ++i) {
        // Initial guess: the asymptotic node location.
        double xi = std::cos(pi * (i + 0.75) / (npts + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            // Evaluate Legendre P_npts and its derivative by recurrence.
            double p0 = 1.0, p1 = xi;
            for (int k = 2; k <= npts; ++k) {
                const double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
                p0 = p1; p1 = p2;
            }
            dp = npts * (xi * p1 - p0) / (xi * xi - 1.0);
            const double dx = -p1 / dp;
            xi += dx;
            if (std::fabs(dx) < 1e-15) break;
        }
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
}

// Bundle of the three coefficient forms.
struct CoeffForms { Mat M, K, S; };

// Assemble M, K, S for a clamped (open-uniform) degree-p basis on its knot
// vector. Integration runs span by span over the distinct interior breakpoints
// knots[p .. npts]; on each span only the p+1 local basis functions are active.
inline CoeffForms assemble_forms_clamped(int npts, int p, const std::vector<double>& knots) {
    CoeffForms cf{ Mat(npts, npts), Mat(npts, npts), Mat(npts, npts) };
    std::vector<double> gx, gw;
    gauss_legendre(p + 1, gx, gw);
    std::vector<std::vector<double>> ders;
    const int nders = (p >= 1) ? 1 : 0;
    for (int s = p; s < npts; ++s) {            // span [knots[s], knots[s+1]]
        const double a = knots[s], b = knots[s + 1];
        if (b <= a) continue;                   // skip repeated knots
        const double half = 0.5 * (b - a), mid = 0.5 * (a + b);
        for (int q = 0; q < p + 1; ++q) {
            const double xq = mid + half * gx[q];
            const double wq = gw[q] * half;
            ders_basis_funs(s, xq, p, nders, knots, ders);
            for (int aa = 0; aa <= p; ++aa) {
                const int ca = s - p + aa;
                const double Na  = ders[0][aa];
                const double dNa = (p >= 1) ? ders[1][aa] : 0.0;
                for (int bb = 0; bb <= p; ++bb) {
                    const int cb = s - p + bb;
                    const double Nb  = ders[0][bb];
                    const double dNb = (p >= 1) ? ders[1][bb] : 0.0;
                    cf.M(ca, cb) += wq * Na  * Nb;
                    cf.K(ca, cb) += wq * Na  * dNb;
                    cf.S(ca, cb) += wq * dNa * dNb;
                }
            }
        }
    }
    return cf;
}

// Overlap integral \int beta_p^{(di)}(u) beta_p^{(dj)}(u - d) du for integer
// shift d and derivative orders di, dj in {0, 1}. beta_p is supported on
// [-(p+1)/2, (p+1)/2] and is piecewise polynomial with unit-spaced breakpoints,
// so a (p+1)-point Gauss rule per unit interval is exact.
inline double beta_overlap(int p, int d, int di, int dj) {
    auto bval = [&](int order, double t) {
        return (order == 0) ? beta_value(p, t) : beta_deriv1(p, t);
    };
    std::vector<double> gx, gw;
    gauss_legendre(p + 1, gx, gw);
    const double R = 0.5 * (p + 1);             // support radius of beta_p
    double total = 0.0;
    for (int k = 0; k <= p; ++k) {              // p+1 unit intervals tiling [-R, R]
        const double left = -R + k, right = left + 1.0;
        const double half = 0.5, mid = 0.5 * (left + right);
        for (int q = 0; q < p + 1; ++q) {
            const double u = mid + half * gx[q];
            const double wq = gw[q] * half;
            total += wq * bval(di, u) * bval(dj, u - d);
        }
    }
    return total;
}

// Assemble M, K, S for the periodic degree-p cardinal basis on a uniform grid
// of n points with spacing h. Every entry depends only on the index shift, so
// the matrices are circulant; M carries an h, K is h-free, S carries a 1/h
// (matching \int N_i N_j dx, \int N_i N_j' dx, \int N_i' N_j' dx).
inline CoeffForms assemble_forms_periodic(int n, int p, double h) {
    CoeffForms cf{ Mat(n, n), Mat(n, n), Mat(n, n) };
    const int rad = p + 1;                      // overlap radius in index units
    for (int off = -rad; off <= rad; ++off) {   // off = j - i (column minus row)
        const double m  = beta_overlap(p, off, 0, 0);
        const double k  = (p >= 1) ? beta_overlap(p, off, 0, 1) : 0.0;
        const double s2 = (p >= 1) ? beta_overlap(p, off, 1, 1) : 0.0;
        for (int i = 0; i < n; ++i) {
            const int j = ((i + off) % n + n) % n;
            cf.M(i, j) += h * m;
            cf.K(i, j) += k;
            cf.S(i, j) += s2 / h;
        }
    }
    return cf;
}

// ===========================================================================
// Part 5. The induced degree-(p-1) derivative spline space (BMF flux grid)
// ===========================================================================
//
// The derivative of a clamped degree-p spline lies in the degree-(p-1) spline
// space on the SHORTENED knot vector (drop the first and last knot). That space
// has its own Greville grid eta, its own sample matrix, and its own mass form.
// The BMF staggered gradient maps primal Greville values to this grid.
struct DerivativeSpace {
    int p1 = 0;                      // degree p-1
    int m  = 0;                      // number of basis functions = N - 1
    std::vector<double> knots;       // shortened knot vector
    std::vector<double> eta;         // Greville grid of the derivative space
    Mat Bhat;                        // sample matrix Bhat(i,j) = Nhat_j(eta_i)
    Mat Delta;                       // (N-1) x N derivative-coefficient map
    CoeffForms forms;                // M-hat, K-hat, S-hat of the derivative space
};

inline DerivativeSpace build_derivative_space(int npts, int p,
                                              const std::vector<double>& knots) {
    DerivativeSpace ds;
    ds.p1 = p - 1;
    ds.m  = npts - 1;
    // Shortened knot vector: drop the first and last entry.
    ds.knots.assign(knots.begin() + 1, knots.end() - 1);
    ds.eta = greville_points(ds.knots, ds.p1, ds.m);

    // Sample matrix of the degree-(p-1) basis at its own Greville grid.
    ds.Bhat = Mat(ds.m, ds.m);
    const int nders = 0;
    std::vector<std::vector<double>> dbuf;
    for (int i = 0; i < ds.m; ++i) {
        const double x = ds.eta[i];
        const int span = find_span(ds.m - 1, ds.p1, x, ds.knots);
        ders_basis_funs(span, x, ds.p1, nders, ds.knots, dbuf);
        for (int j = 0; j <= ds.p1; ++j) ds.Bhat(i, span - ds.p1 + j) = dbuf[0][j];
    }

    // Standard B-spline derivative formula in coefficient space:
    //   (Delta alpha)_i = p / (t_{i+p+1} - t_{i+1}) * (alpha_{i+1} - alpha_i).
    ds.Delta = Mat(ds.m, npts);
    for (int i = 0; i < ds.m; ++i) {
        const double denom = knots[i + p + 1] - knots[i + 1];
        const double w = (denom > 0.0) ? p / denom : 0.0;
        ds.Delta(i, i)     = -w;
        ds.Delta(i, i + 1) =  w;
    }

    ds.forms = assemble_forms_clamped(ds.m, ds.p1, ds.knots);
    return ds;
}

// Solve B alpha = u for the spline coefficients, choosing the right fast path:
// a cyclic banded solve for periodic operators, a plain banded solve otherwise.
// Both factorize B once internally; for repeated solves with the same B you
// would build the solver once and reuse it (see mms_check.cpp).
inline std::vector<double> coefficients_from_values(const SplineOperators& op,
                                                    const std::vector<double>& u) {
    if (op.periodic) {
        CyclicSolver cs;
        cs.set_from_dense(op.B, op.half_bw);
        return cs.solve(u);
    } else {
        BandSolver bs;
        bs.set_from_dense(op.B, op.half_bw, op.half_bw);
        bs.factor();
        return bs.solve(u);
    }
}

#endif // BSPLINE_H
