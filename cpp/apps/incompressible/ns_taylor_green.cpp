// ns_taylor_green.cpp
//
// Validation of the incompressible Navier-Stokes solver (ns.h) on the 2D
// Taylor-Green vortex, which has an exact closed-form solution on [0,2pi]^2:
//
//   u(x,y,t) = -cos x sin y e^{-2 nu t}
//   v(x,y,t) =  sin x cos y e^{-2 nu t}
//   p(x,y,t) = -1/4 (cos 2x + cos 2y) e^{-4 nu t}
//
// This exercises the whole machinery at once: skew-symmetric advection, the
// viscous Laplacian, and the discrete pressure projection (advection and pressure
// gradient must cancel so the velocity simply decays). We report the max-norm
// error vs the exact field, the kinetic-energy decay vs the exact e^{-4 nu t},
// the residual divergence (should be round-off), and the temporal order.

#include <cstdio>
#include <cmath>
#include "ns.h"

static const double PI = 3.14159265358979323846;

static double tg_u(double x, double y, double t, double nu) { return -std::cos(x) * std::sin(y) * std::exp(-2 * nu * t); }
static double tg_v(double x, double y, double t, double nu) { return  std::sin(x) * std::cos(y) * std::exp(-2 * nu * t); }

static void init_tg(IncompressibleNS& S, double nu) {
    const double hx = 2 * PI / S.nx, hy = 2 * PI / S.ny;
    for (int j = 0; j < S.ny; ++j)
        for (int i = 0; i < S.nx; ++i) {
            const size_t id = S.u.idx(i, j, 0);
            S.u.a[id] = tg_u(i * hx, j * hy, 0.0, nu);
            S.v.a[id] = tg_v(i * hx, j * hy, 0.0, nu);
        }
    S.project();                                   // clean discrete divergence at t=0
}

// Run to time T with step dt; return max-norm velocity error vs exact.
static double run_to(int N, int p, double nu, double T, double dt,
                     double& div_out, double& ke_ratio_out, bool imex = false,
                     Route route = Route::BMC) {
    SimConfig c; c.p = p; c.route = route; c.nu = nu;
    c.ax[0] = {N, 2 * PI, BC::Periodic, true};
    c.ax[1] = {N, 2 * PI, BC::Periodic, true};
    c.ax[2] = {1, 1.0, BC::Periodic, false};
    IncompressibleNS S(c);
    if (!S.ready) { std::printf("   setup failed: %s\n", S.setup_error); return -1; }

    init_tg(S, nu);
    const double ke0 = S.kinetic_energy();
    const int nsteps = (int)std::lround(T / dt);
    for (int n = 0; n < nsteps; ++n) { if (imex) S.step_imex(dt); else S.step(dt); }
    const double tfin = nsteps * dt;

    const double hx = 2 * PI / N, hy = 2 * PI / N;
    double err = 0.0; bool finite = true;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const size_t id = S.u.idx(i, j, 0);
            const double du = S.u.a[id] - tg_u(i * hx, j * hy, tfin, nu);
            const double dv = S.v.a[id] - tg_v(i * hx, j * hy, tfin, nu);
            if (!std::isfinite(du) || !std::isfinite(dv)) { finite = false; continue; }
            err = std::fmax(err, std::fmax(std::fabs(du), std::fabs(dv)));
        }
    div_out = S.max_divergence();
    ke_ratio_out = S.kinetic_energy() / (ke0 * std::exp(-4 * nu * tfin));
    return finite ? err : INFINITY;            // INFINITY flags a blow-up to the caller
}

int main() {
    std::printf("==================================================================\n");
    std::printf(" Incompressible Navier-Stokes: 2D Taylor-Green vortex validation\n");
    std::printf(" exact: u=-cos x sin y e^{-2nu t},  v=sin x cos y e^{-2nu t}\n");
    std::printf("==================================================================\n\n");

    const int p = 6;
    const double nu = 0.05, T = 1.0, dt = 0.0025;

    // Spatial convergence: refine N at a small fixed dt (so the temporal RK3 error
    // sits below the spatial error). The smooth single-vortex flow is resolved
    // spectrally, so the error drops fast then hits the round-off floor.
    std::printf(" p=%d nu=%.3g  integrate to T=%.1f at dt=%.4f, refine N\n", p, nu, T, dt);
    std::printf("    N      max err(u,v)     order    max|div u|    KE / KE_exact\n");
    double prev_e = 0; int prev_N = 0;
    for (int N : {16, 32, 64, 128}) {              // power-of-two (FFT pressure solve)
        double div, ke;
        double e = run_to(N, p, nu, T, dt, div, ke);
        double ord = (prev_e > 1e-13 && e > 1e-13) ? std::log(prev_e / e) / std::log((double)N / prev_N) : 0.0;
        std::printf("   %3d     %.4e      %s%4.1f     %.2e      %.6f\n",
                    N, e, prev_e > 0 ? "" : "  ", ord, div, ke);
        prev_e = e; prev_N = N;
    }

    std::printf("\n max|div u| ~ round-off  => discrete projection is exactly divergence-free.\n");
    std::printf(" KE / KE_exact ~ 1       => correct viscous energy decay.\n");
    std::printf(" error falls fast in N   => spectral-like spatial accuracy in a full NS solve.\n");

    // Route switch: the same solve with the BMG (Galerkin) first-derivative route.
    {
        double div, ke;
        double e = run_to(64, p, nu, T, dt, div, ke, false, Route::BMG);
        std::printf("\n route=BMG (N=64):  max err %.4e   max|div u| %.2e   KE/KE_exact %.6f\n",
                    e, div, ke);
    }

    // IMEX vs explicit: the implicit viscous treatment lifts the diffusive dt
    // limit. At a strongly viscous setting the explicit RK3 goes unstable while
    // the IMEX RK3/CN stays accurate at the same (large) dt.
    std::printf("\n --- IMEX (implicit viscous) vs explicit, strong viscosity nu=0.5 ---\n");
    std::printf("   dt        explicit RK3        IMEX RK3/CN (err, |div|, KE/exact)\n");
    const int Ni = 64; const double nu2 = 0.5;
    for (double dt : {0.004, 0.016}) {
        double de, ke, di, ki;
        double e_ex = run_to(Ni, p, nu2, 0.5, dt, de, ke, false);
        double e_im = run_to(Ni, p, nu2, 0.5, dt, di, ki, true);
        const bool blew = !std::isfinite(e_ex) || e_ex > 1e2;
        if (blew) std::printf("   %.4f    UNSTABLE            %.3e  %.1e  %.4f\n", dt, e_im, di, ki);
        else      std::printf("   %.4f    err %.2e        %.3e  %.1e  %.4f\n", dt, e_ex, e_im, di, ki);
    }
    std::printf("\nDone.\n");
    return 0;
}
