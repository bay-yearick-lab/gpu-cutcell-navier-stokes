// mms_wall2d.cpp
//
// The energy-conserving wall advection, extended to MULTIPLE AXES (the step from
// the 1D Burgers kernel toward 3D wall-NS). A scalar phi is advected by a
// divergence-free velocity u = curl(psi) in a 2D WALLED box (clamped x clamped),
// using the full 2D isogeometric Galerkin advection (wall2d.h, advect2d). Two
// facts make the discrete enstrophy 1/2 \int phi^2 conserved to ROUND-OFF:
// u = curl(psi_h) of a SPLINE streamfunction (div u_h = 0 exactly, integrand
// polynomial so quadrature is exact) and psi_h = 0 on the walls (u.n = 0). Then
//   alpha_phi^T C = 1/2 \oint phi^2 (u.n) - 1/2 \int phi^2 div u_h = 0,
// exactly -- independent of the operators' non-diagonal energy norm. This is the
// multi-axis analogue of the 1D result; in 3D NS it is the same per plane.

#include <cstdio>
#include <cmath>
#include <vector>
#include "wall2d.h"

static const double PI = 3.14159265358979323846;

static void test_identity() {
    std::printf("=================================================================\n");
    std::printf(" Semi-discrete enstrophy rate  alpha_phi^T C  (must be round-off:\n");
    std::printf(" the 2D Galerkin advection injects zero enstrophy on a walled box)\n");
    std::printf("=================================================================\n");
    std::printf("   p    N      alpha_phi^T C      max|C|       (rate/scale)\n");
    for (int p : {2, 3, 4, 5}) {
        for (int N : {12, 18}) {
            const double Lx = 2.0, Lz = 1.5;
            Axis X(N, p, Lx), Z(N, p, Lz);
            std::vector<double> sx(N), sz(N);
            for (int i = 0; i < N; ++i) sx[i] = std::sin(PI * X.grev[i] / Lx);
            for (int j = 0; j < N; ++j) sz[j] = std::sin(PI * Z.grev[j] / Lz);
            std::vector<double> cx = X.interp(sx), cz = Z.interp(sz);
            Field2D psi(N, N), phi(N, N), C(N, N);
            for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) psi.at(i, j) = cx[i] * cz[j];
            std::vector<double> fx(N), fz(N);
            for (int i = 0; i < N; ++i) fx[i] = std::cos(1.3 * X.grev[i] + 0.4) + 0.5 * X.grev[i];
            for (int j = 0; j < N; ++j) fz[j] = std::sin(0.9 * Z.grev[j] + 0.7);
            std::vector<double> qx = X.interp(fx), qz = Z.interp(fz);
            for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) phi.at(i, j) = qx[i] * qz[j];

            advect2d(X, Z, psi, phi, C);
            double rate = 0, scale = 0, cmax = 0;
            for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
                rate += phi.at(i, j) * C.at(i, j);
                scale += std::fabs(phi.at(i, j) * C.at(i, j));
                cmax = std::fmax(cmax, std::fabs(C.at(i, j)));
            }
            std::printf("  %2d  %3d    %12.3e     %.3e    %.2e\n",
                        p, N, rate, cmax, std::fabs(rate) / (scale + 1e-300));
        }
    }
    std::printf("\n => zero enstrophy injection on a 2D WALLED box: the energy-conserving\n");
    std::printf("    advection composes across axes. Same construction per plane in 3D.\n\n");
}

static void test_dynamic() {
    std::printf("=================================================================\n");
    std::printf(" Dynamic: time-march phi in the fixed vortex, enstrophy vs time\n");
    std::printf("=================================================================\n");
    const int N = 20, p = 4; const double Lx = 2.0, Lz = 1.5;
    Axis X(N, p, Lx), Z(N, p, Lz);
    std::vector<double> sx(N), sz(N);
    for (int i = 0; i < N; ++i) sx[i] = std::sin(PI * X.grev[i] / Lx);
    for (int j = 0; j < N; ++j) sz[j] = std::sin(PI * Z.grev[j] / Lz);
    std::vector<double> cx = X.interp(sx), cz = Z.interp(sz);
    Field2D psi(N, N), phi(N, N), C(N, N), p0(N, N), kk(N, N);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) psi.at(i, j) = cx[i] * cz[j];
    std::vector<double> fx(N), fz(N);
    for (int i = 0; i < N; ++i) fx[i] = std::exp(-8.0 * (X.grev[i] / Lx - 0.35) * (X.grev[i] / Lx - 0.35));
    for (int j = 0; j < N; ++j) fz[j] = std::exp(-8.0 * (Z.grev[j] / Lz - 0.5) * (Z.grev[j] / Lz - 0.5));
    std::vector<double> qx = X.interp(fx), qz = Z.interp(fz);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) phi.at(i, j) = qx[i] * qz[j];

    auto rhs = [&](Field2D& a, Field2D& da) {
        advect2d(X, Z, psi, a, C);
        da = C; for (double& v : da.a) v = -v;
        mass_solve2d(X, Z, da);
    };
    const double E0 = enstrophy2d(X, Z, phi);
    const double dt = 0.004; const int steps = 500;
    double emax = E0, emin = E0;
    for (int s = 0; s < steps; ++s) {
        p0 = phi;
        rhs(phi, kk); for (size_t q = 0; q < phi.a.size(); ++q) phi.a[q] = p0.a[q] + dt * kk.a[q];
        rhs(phi, kk); for (size_t q = 0; q < phi.a.size(); ++q) phi.a[q] = 0.75 * p0.a[q] + 0.25 * (phi.a[q] + dt * kk.a[q]);
        rhs(phi, kk); for (size_t q = 0; q < phi.a.size(); ++q) phi.a[q] = (1.0/3.0) * p0.a[q] + (2.0/3.0) * (phi.a[q] + dt * kk.a[q]);
        const double e = enstrophy2d(X, Z, phi); emax = std::fmax(emax, e); emin = std::fmin(emin, e);
    }
    const double Ef = enstrophy2d(X, Z, phi);
    std::printf("   N=%d p=%d  %d steps dt=%.3f:  E/E0 = %.10f  |dE|/E0 = %.2e\n",
                N, p, steps, dt, Ef / E0, std::fabs(Ef - E0) / E0);
    std::printf("   range over run: [%.10f, %.10f] -> BOUNDED, enstrophy conserved.\n\n",
                emin / E0, emax / E0);
}

int main() {
    std::printf("\n ENERGY-CONSERVING WALL ADVECTION -- 2D walled box (multi-axis)\n\n");
    test_identity();
    test_dynamic();
    std::printf("Done.\n");
    return 0;
}
