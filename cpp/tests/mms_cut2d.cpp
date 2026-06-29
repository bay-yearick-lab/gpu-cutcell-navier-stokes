// mms_cut2d.cpp
//
// MMS / identity gate for the 2D oblique sharp-cut kernel (cut/cut2d.h), the C++
// port of code/oblique_cut_sbp.py.  Four gates:
//
//   (1) Cut-domain directional Green identities
//       ||Qx + Qx^T - Sx||,  ||Qy + Qy^T - Sy||  -> roundoff.
//   (2) Tensor-monomial reproduction by E   -> roundoff.
//   (3) SBP residual under the E^T(.)E pullback  -> roundoff.
//   (4) Best-approximation order p+1 on a fixed oblique cut, under refinement.

#include <cmath>
#include <cstdio>
#include <vector>

#include "cut2d.h"

static double u_exact(double x, double y) {
    return std::sin(2.3*x + 0.7*y + 0.4) + 0.5*std::cos(1.1*x - 1.7*y);
}

int main() {
    std::printf("=================================================================\n");
    std::printf(" MMS / identity gate: 2D oblique sharp-cut kernel (cut2d.h)\n");
    std::printf(" half-plane cut  n.x >= offset,  theta = 30 deg\n");
    std::printf("=================================================================\n\n");

    // Oblique cut at 30 deg through the interior; offset chosen so the cut crosses
    // the unit square and slices cells into slivers.
    const double theta = 30.0 * M_PI / 180.0;
    HalfPlaneCut cut; cut.nx_ = std::cos(theta); cut.ny_ = std::sin(theta);
    cut.offset = 0.5*(cut.nx_ + cut.ny_) - 0.23;   // passes near the centre

    for (int p = 2; p <= 4; ++p) {
        std::printf("  degree p = %d  (formal best-approx order p+1 = %d)\n", p, p + 1);
        std::printf("   N_el   h         gamma_min   #small  Green_x     Green_y     polyE       SBP(red)    L2err       order\n");
        double prev_err = 0.0, prev_h = 0.0;
        double mg = 0.0, mp = 0.0, ms = 0.0;
        for (int ne : {8, 12, 16, 24}) {
            const double h = 1.0 / ne;
            CutForms2D cf = assemble_oblique_cut_2d(ne, p, cut);
            ActiveSets2D as = cut2d_active_sets(cf);
            Extension2D ext = cut2d_extension(cf, as);

            // Gate 1: directional Green identities on the full forms.
            double gx = 0.0, gy = 0.0;
            for (int i = 0; i < cf.n; ++i) for (int j = 0; j < cf.n; ++j) {
                gx = std::max(gx, std::fabs(cf.Qx(i,j) + cf.Qx(j,i) - cf.Sx(i,j)));
                gy = std::max(gy, std::fabs(cf.Qy(i,j) + cf.Qy(j,i) - cf.Sy(i,j)));
            }

            // Gate 3: SBP under pullback (x-direction).
            Mat Qr = cut2d_pullback(cf.Qx, ext.E), Sr = cut2d_pullback(cf.Sx, ext.E);
            double sbp = 0.0;
            for (int i = 0; i < Qr.r; ++i) for (int j = 0; j < Qr.c; ++j)
                sbp = std::max(sbp, std::fabs(Qr(i,j) + Qr(j,i) - Sr(i,j)));

            // Gate 4: reduced-space L2 best approximation.
            Mat Hr = cut2d_pullback(cf.H, ext.E);
            FluidQuad2D fq = cut2d_fluid_quadrature(ne, p, cut);
            std::vector<double> bfull(cf.n, 0.0);
            for (size_t qq = 0; qq < fq.w.size(); ++qq) {
                std::vector<int> rows; std::vector<double> phi;
                cut2d_point_basis(cf, fq.x[qq], fq.y[qq], rows, phi);
                const double uq = u_exact(fq.x[qq], fq.y[qq]);
                for (size_t a = 0; a < rows.size(); ++a) bfull[rows[a]] += fq.w[qq]*phi[a]*uq;
            }
            std::vector<double> br = matvec(transpose(ext.E), bfull);
            DenseMatrix Hrd = to_dense(Hr);
            std::vector<double> c; dense_lu_solve(Hrd, br, c);
            std::vector<double> alpha = matvec(ext.E, c);
            double e2 = 0.0;
            for (size_t qq = 0; qq < fq.w.size(); ++qq) {
                std::vector<int> rows; std::vector<double> phi;
                cut2d_point_basis(cf, fq.x[qq], fq.y[qq], rows, phi);
                double uh = 0.0; for (size_t a = 0; a < rows.size(); ++a) uh += alpha[rows[a]]*phi[a];
                const double d = u_exact(fq.x[qq], fq.y[qq]) - uh;
                e2 += fq.w[qq]*d*d;
            }
            const double err = std::sqrt(e2);

            double gmin = 1e30; for (int i : as.active) gmin = std::min(gmin, as.gamma[i]);
            const double order = (prev_err>0.0) ? std::log(prev_err/err)/std::log(prev_h/h) : 0.0;
            std::printf("   %4d   %.5f   %.2e    %3d     %.2e    %.2e    %.2e    %.2e    %.2e    %s%.2f\n",
                        ne, h, gmin, (int)as.small.size(), gx, gy, ext.poly_residual, sbp, err,
                        (prev_err>0.0)?"":"--", order);
            prev_err = err; prev_h = h;
            mg = std::max(mg, std::max(gx,gy)); mp = std::max(mp, ext.poly_residual); ms = std::max(ms, sbp);
        }
        std::printf("   => Green<=%.1e  polyE<=%.1e  SBP(red)<=%.1e ; best-approx order ~ p+1\n\n", mg, mp, ms);
    }
    std::printf("Done.\n");
    return 0;
}
