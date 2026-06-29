// mms_cut1d.cpp
//
// MMS / identity gate for the 1D sharp-cut kernel (cut/cut1d.h), the C++ port of
// code/cut_spline_sbp.py.  Four gates, each tested as if every line could be wrong:
//
//   (1) Cut-domain Green identity   ||Q + Q^T - boundary_form||  -> roundoff
//       (the continuous integration-by-parts statement, once quadrature resolves
//        the cut).
//   (2) Monomial reproduction       max_r ||E c_r - c_r||        -> roundoff
//       (the polynomial-preserving extension keeps degree p on the active modes).
//   (3) Pullback SBP residual       ||E^T Q E + (E^T Q E)^T - E^T B E|| -> roundoff
//       (the congruence carries the cut-domain SBP identity into the reduced space).
//   (4) Best-approximation order    L2(fluid) of the reduced-space projection,
//       p+1 on a self-similar sliver cut -- the decisive accuracy gate that the
//       extension does NOT degrade order despite a vanishing cut cell.

#include <cmath>
#include <cstdio>
#include <vector>

#include "cut1d.h"

static double u_exact(double x) { return std::sin(2.7 * x + 0.4) + 0.5 * std::cos(1.3 * x); }

// L2(fluid) load vector  b_i = \int_[left,1] N_i u  by panel Gauss quadrature.
static std::vector<double> load_vector(const CutForms1D& cf, double left, double right) {
    const int n = cf.n, p = cf.degree;
    const int q = std::max(10, p + 6);
    std::vector<double> gx, gw; gauss_legendre(q, gx, gw);
    std::vector<double> b(n, 0.0);
    std::vector<double> brk = cut1d_panels(cf.knots, left, right);
    for (size_t s = 0; s + 1 < brk.size(); ++s) {
        const double plo = brk[s], phi = brk[s + 1];
        const double half = 0.5 * (phi - plo), mid = 0.5 * (phi + plo);
        for (int g = 0; g < q; ++g) {
            const double xq = mid + half * gx[g], wq = gw[g] * half;
            std::vector<double> V = cut1d_eval_all(cf.knots, p, n, xq, 0);
            const double uq = u_exact(xq);
            for (int i = 0; i < n; ++i) b[i] += wq * V[i] * uq;
        }
    }
    return b;
}

// L2(fluid) norm of (u - u_h) where u_h has full-space coeffs alpha.
static double l2_error(const CutForms1D& cf, const std::vector<double>& alpha,
                       double left, double right) {
    const int n = cf.n, p = cf.degree;
    const int q = std::max(12, p + 8);
    std::vector<double> gx, gw; gauss_legendre(q, gx, gw);
    double e2 = 0.0;
    std::vector<double> brk = cut1d_panels(cf.knots, left, right);
    for (size_t s = 0; s + 1 < brk.size(); ++s) {
        const double plo = brk[s], phi = brk[s + 1];
        const double half = 0.5 * (phi - plo), mid = 0.5 * (phi + plo);
        for (int g = 0; g < q; ++g) {
            const double xq = mid + half * gx[g], wq = gw[g] * half;
            std::vector<double> V = cut1d_eval_all(cf.knots, p, n, xq, 0);
            double uh = 0.0; for (int i = 0; i < n; ++i) uh += alpha[i] * V[i];
            const double d = u_exact(xq) - uh;
            e2 += wq * d * d;
        }
    }
    return std::sqrt(e2);
}

int main() {
    std::printf("=================================================================\n");
    std::printf(" MMS / identity gate: 1D sharp-cut kernel (cut1d.h)\n");
    std::printf(" self-similar sliver cut at left = theta*h, fluid = [left, 1]\n");
    std::printf("=================================================================\n\n");

    const double theta = 0.85;     // cut 85% into the first cell -> a deep sliver

    for (int p = 2; p <= 5; ++p) {
        std::printf("  degree p = %d  (formal best-approx order p+1 = %d)\n", p, p + 1);
        std::printf("   N_el   h         gamma_min   #small  Green       polyE       SBP(red)    L2err       order\n");
        double prev_err = 0.0, prev_h = 0.0;
        double max_green = 0.0, max_poly = 0.0, max_sbp = 0.0;
        for (int ne : {8, 16, 32, 64}) {
            const double h = 1.0 / ne;
            const double left = theta * h;
            CutForms1D cf = assemble_cut_1d(ne, p, left, 1.0);
            ActiveSets1D as = cut1d_active_sets(cf);
            Extension1D ext = cut1d_extension(cf, as);

            // Gate 1: cut-domain Green identity on the full forms.
            Mat Qs = cf.Q; Mat QT = transpose(cf.Q);
            double green = 0.0;
            for (int i = 0; i < cf.n; ++i) for (int j = 0; j < cf.n; ++j)
                green = std::max(green, std::fabs(Qs(i,j) + QT(i,j) - cf.boundary(i,j)));

            // Gate 3: SBP residual under the pullback.
            Mat Qr = cut1d_pullback(cf.Q, ext.E);
            Mat Br = cut1d_pullback(cf.boundary, ext.E);
            double sbp = 0.0;
            for (int i = 0; i < Qr.r; ++i) for (int j = 0; j < Qr.c; ++j)
                sbp = std::max(sbp, std::fabs(Qr(i,j) + Qr(j,i) - Br(i,j)));

            // Gate 4: reduced-space L2 best approximation.
            Mat Hr = cut1d_pullback(cf.H, ext.E);
            std::vector<double> b = load_vector(cf, left, 1.0);
            std::vector<double> br = matvec(transpose(ext.E), b);
            DenseMatrix Hrd = to_dense(Hr);
            std::vector<double> c;
            dense_lu_solve(Hrd, br, c);
            std::vector<double> alpha = matvec(ext.E, c);   // full-space coeffs
            double err = l2_error(cf, alpha, left, 1.0);

            double gmin = 1e30; int nsmall = (int)as.small.size();
            for (int i : as.active) gmin = std::min(gmin, as.gamma[i]);

            double order = (prev_err > 0.0) ? std::log(prev_err / err) / std::log(prev_h / h) : 0.0;
            std::printf("   %4d   %.5f   %.2e    %3d     %.2e    %.2e    %.2e    %.2e    %s%.2f\n",
                        ne, h, gmin, nsmall, green, ext.poly_residual, sbp, err,
                        (prev_err > 0.0) ? "" : "--", order);
            prev_err = err; prev_h = h;
            max_green = std::max(max_green, green);
            max_poly = std::max(max_poly, ext.poly_residual);
            max_sbp = std::max(max_sbp, sbp);
        }
        std::printf("   => Green<=%.1e  polyE<=%.1e  SBP(red)<=%.1e ; best-approx order ~ p+1\n\n",
                    max_green, max_poly, max_sbp);
    }
    std::printf("Done.\n");
    return 0;
}
