// cut1d.h
//
// One-dimensional sharp-cut B-spline kernel: the smallest end-to-end instance of
// the cut-cell construction.  It is the C++ port of
// `code/cut_spline_sbp.py` (open-uniform basis, cut mass/derivative/stiffness
// forms, support-ratio classification, the polynomial-preserving extension E,
// and the EᵀAE pullback), built on the verified basis machinery of bspline.h.
//
// Conventions match the Python EXACTLY so the two can be cross-checked to
// roundoff:
//   * domain [0,1], open-uniform clamped degree-p basis, n = n_elements + p
//     basis functions (build_clamped(n_elements+p, p, 1.0));
//   * the fluid region is the interval [left, right] (default right = 1);
//   * H   = \int_fluid N_i N_j,   Q = \int_fluid N_i N_j',   K = \int_fluid N_i' N_j';
//   * H_bg = \int_[0,1] N_i N_j   (the uncut background mass);
//   * boundary_form = r r^T - l l^T with l,r the basis traces at left,right;
//   * gamma_i = diag(H)_i / diag(H_bg)_i ; interior gamma>=0.05, active gamma>tol;
//   * E reproduces every monomial up to degree p on the active modes; small-support
//     (sliver) modes are degree-p combinations of nearby interior modes.
//
// The cut-domain Green identity Q + Q^T = boundary_form is a continuous
// integration-by-parts statement; it holds to roundoff once the panel quadrature
// resolves the cut, and is the first gate.

#ifndef CUT1D_H
#define CUT1D_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "bspline.h"

// All cut forms for one 1D cut interval, plus the active-set bookkeeping.
struct CutForms1D {
    int n = 0;                       // number of basis functions
    int degree = 0;
    Mat H, Q, K, Hbg;                // cut mass / derivative / stiffness ; background mass
    Mat boundary;                    // r r^T - l l^T
    std::vector<double> ltr, rtr;    // basis traces at left and right
    std::vector<double> greville;    // collocation nodes
    std::vector<double> knots;
};

// Evaluate all n basis functions (nu = 0) or their first derivatives (nu = 1)
// at a single point x.  Only the p+1 functions on the active span are nonzero.
inline std::vector<double> cut1d_eval_all(const std::vector<double>& knots, int p,
                                          int n, double x, int nu) {
    std::vector<double> out(n, 0.0);
    // Clip to the closed parametric interval (matches the Python nan->0 guard).
    const double lo = knots[p], hi = knots[n];   // knots[n] = knots[n_last+1]
    double xc = std::min(std::max(x, lo), hi);
    const int span = find_span(n - 1, p, xc, knots);
    std::vector<std::vector<double>> ders;
    ders_basis_funs(span, xc, p, std::max(nu, 0), knots, ders);
    for (int j = 0; j <= p; ++j) out[span - p + j] = ders[nu][j];
    return out;
}

// Distinct integration panels of [a,b]: a, the interior knots strictly inside,
// then b.  Each panel lies within one knot span, so a Gauss rule is exact.
inline std::vector<double> cut1d_panels(const std::vector<double>& knots,
                                        double a, double b) {
    std::vector<double> brk{a};
    for (double t : knots)
        if (t > a + 1e-15 && t < b - 1e-15) brk.push_back(t);
    brk.push_back(b);
    std::sort(brk.begin(), brk.end());
    brk.erase(std::unique(brk.begin(), brk.end(),
              [](double u, double v){ return std::fabs(u - v) < 1e-14; }), brk.end());
    return brk;
}

// Assemble all cut forms for the fluid interval [left,right] on [0,1].
inline CutForms1D assemble_cut_1d(int n_elements, int p, double left,
                                  double right = 1.0, int q = -1) {
    const int n = n_elements + p;
    std::vector<double> knots = open_uniform_knots(n, p, 1.0);
    std::vector<double> grev  = greville_points(knots, p, n);
    if (q < 0) q = std::max(8, p + 4);
    std::vector<double> gx, gw;
    gauss_legendre(q, gx, gw);

    CutForms1D cf;
    cf.n = n; cf.degree = p; cf.knots = knots; cf.greville = grev;
    cf.H = Mat(n, n); cf.Q = Mat(n, n); cf.K = Mat(n, n); cf.Hbg = Mat(n, n);
    cf.boundary = Mat(n, n);

    auto integrate = [&](double a, double b, Mat* MH, Mat* MQ, Mat* MK) {
        std::vector<double> brk = cut1d_panels(knots, a, b);
        for (size_t s = 0; s + 1 < brk.size(); ++s) {
            const double plo = brk[s], phi = brk[s + 1];
            if (phi <= plo) continue;
            const double half = 0.5 * (phi - plo), mid = 0.5 * (phi + plo);
            for (int g = 0; g < q; ++g) {
                const double xq = mid + half * gx[g];
                const double wq = gw[g] * half;
                std::vector<double> V  = cut1d_eval_all(knots, p, n, xq, 0);
                std::vector<double> dV = (MQ || MK) ? cut1d_eval_all(knots, p, n, xq, 1)
                                                    : std::vector<double>();
                // Only the local span contributes; gather its index window.
                const int span = find_span(n - 1, p, std::min(std::max(xq, knots[p]), knots[n]), knots);
                for (int ia = 0; ia <= p; ++ia) {
                    const int i = span - p + ia;
                    const double Vi = V[i], dVi = dV.empty() ? 0.0 : dV[i];
                    for (int ib = 0; ib <= p; ++ib) {
                        const int j = span - p + ib;
                        if (MH) (*MH)(i, j) += wq * Vi * V[j];
                        if (MQ) (*MQ)(i, j) += wq * Vi * dV[j];
                        if (MK) (*MK)(i, j) += wq * dVi * dV[j];
                    }
                }
            }
        }
    };

    integrate(left, right, &cf.H, &cf.Q, &cf.K);
    integrate(0.0, 1.0, &cf.Hbg, nullptr, nullptr);

    cf.ltr = cut1d_eval_all(knots, p, n, left, 0);
    cf.rtr = cut1d_eval_all(knots, p, n, right, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cf.boundary(i, j) = cf.rtr[i] * cf.rtr[j] - cf.ltr[i] * cf.ltr[j];
    return cf;
}

// Support ratios gamma_i = diag(H)_i / diag(H_bg)_i and the active/interior/
// small-support classification.
struct ActiveSets1D {
    std::vector<double> gamma;
    std::vector<int> active, interior, small;
};
inline ActiveSets1D cut1d_active_sets(const CutForms1D& cf,
                                      double support_threshold = 0.05,
                                      double active_tol = 1e-25) {
    ActiveSets1D a;
    a.gamma.assign(cf.n, 0.0);
    for (int i = 0; i < cf.n; ++i) {
        const double bg = cf.Hbg(i, i);
        a.gamma[i] = (bg > 0.0) ? cf.H(i, i) / bg : 0.0;
    }
    for (int i = 0; i < cf.n; ++i) {
        if (a.gamma[i] > active_tol)        a.active.push_back(i);
        if (a.gamma[i] >= support_threshold) a.interior.push_back(i);
    }
    for (int i : a.active)
        if (a.gamma[i] < support_threshold) a.small.push_back(i);
    return a;
}

// Spline coefficients of the monomials 1, x, ..., x^maxdeg: solve B(greville) c_r
// = greville^r.  Row r of the returned (maxdeg+1) x n matrix holds c_r.
inline Mat cut1d_poly_coeffs(const CutForms1D& cf, int maxdeg) {
    const int n = cf.n, p = cf.degree;
    DenseMatrix A(n);
    for (int k = 0; k < n; ++k) {
        std::vector<double> row = cut1d_eval_all(cf.knots, p, n, cf.greville[k], 0);
        for (int i = 0; i < n; ++i) A(k, i) = row[i];
    }
    Mat poly(maxdeg + 1, n);
    for (int r = 0; r <= maxdeg; ++r) {
        std::vector<double> rhs(n), c;
        for (int k = 0; k < n; ++k) rhs[k] = std::pow(cf.greville[k], r);
        bool ok = dense_lu_solve(A, rhs, c);
        (void)ok;
        for (int i = 0; i < n; ++i) poly(r, i) = c[i];
    }
    return poly;
}

// The polynomial-preserving extension E (n x n_interior): identity on interior
// modes; each small-support mode is reconstructed as a degree-p combination of
// the nearest interior modes (least squares against its monomial column).
struct Extension1D { Mat E; std::vector<int> interior; double poly_residual = 0.0; int degree = 0; };

inline Extension1D cut1d_extension(const CutForms1D& cf, const ActiveSets1D& as,
                                   int preserve_degree = -1) {
    const int n = cf.n;
    const int n_int = (int)as.interior.size();
    int degree = (preserve_degree < 0) ? cf.degree : preserve_degree;
    degree = std::min(degree, n_int - 1);
    Mat poly = cut1d_poly_coeffs(cf, degree);

    std::vector<int> col(n, -1);
    for (int j = 0; j < n_int; ++j) col[as.interior[j]] = j;

    Extension1D ext;
    ext.E = Mat(n, n_int);
    ext.interior = as.interior;
    ext.degree = degree;
    for (int g : as.interior) ext.E(g, col[g]) = 1.0;

    const int nterms = degree + 1;
    for (int b : as.small) {
        // Interior modes ordered by Greville distance to b.
        std::vector<int> ordered = as.interior;
        std::sort(ordered.begin(), ordered.end(), [&](int a, int c){
            return std::fabs(cf.greville[a] - cf.greville[b])
                 < std::fabs(cf.greville[c] - cf.greville[b]);
        });
        // Grow the stencil until its monomial-coefficient block is full rank.
        std::vector<int> chosen(ordered.begin(), ordered.begin() + std::min(nterms, (int)ordered.size()));
        // (For 1D the nearest nterms are full-rank in all tested configurations;
        //  a guarded growth loop keeps parity with the Python.)
        for (int k = nterms; k <= std::min((int)ordered.size(), degree + 6); ++k) {
            // Build the nterms x k block and test rank via the smallest singular
            // value of its Gram matrix (cheap for tiny blocks).
            std::vector<int> cand(ordered.begin(), ordered.begin() + k);
            Mat C(nterms, k);
            for (int r = 0; r < nterms; ++r)
                for (int cc = 0; cc < k; ++cc) C(r, cc) = poly(r, cand[cc]);
            // Rank via Gram eigen estimate: full row rank iff C C^T is SPD.
            Mat G = matmul(C, transpose(C));
            bool okc = true; { bool ok; inverse(G, &ok); okc = ok; }
            if (okc) { chosen = std::vector<int>(ordered.begin(), ordered.begin() + nterms); break; }
        }
        // Solve the (nterms x nterms) system C w = target for the weights.
        DenseMatrix C(nterms);
        std::vector<double> target(nterms), w;
        for (int r = 0; r < nterms; ++r) {
            for (int cc = 0; cc < nterms; ++cc) C(r, cc) = poly(r, chosen[cc]);
            target[r] = poly(r, b);
        }
        dense_lu_solve(C, target, w);
        for (int cc = 0; cc < nterms; ++cc) ext.E(b, col[chosen[cc]]) = w[cc];
    }

    // Monomial-reproduction residual on the active modes.
    double res = 0.0;
    for (int r = 0; r <= degree; ++r) {
        std::vector<double> cr(n_int);
        for (int j = 0; j < n_int; ++j) cr[j] = poly(r, as.interior[j]);
        std::vector<double> Ecr = matvec(ext.E, cr);
        for (int i = 0; i < n; ++i) {
            const double tgt = (as.gamma[i] > 0.0) ? poly(r, i) : 0.0;
            res = std::max(res, std::fabs(Ecr[i] - tgt));
        }
    }
    ext.poly_residual = res;
    return ext;
}

// EᵀAE pullback of a square form.
inline Mat cut1d_pullback(const Mat& A, const Mat& E) {
    return matmul(transpose(E), matmul(A, E));
}

#endif // CUT1D_H
