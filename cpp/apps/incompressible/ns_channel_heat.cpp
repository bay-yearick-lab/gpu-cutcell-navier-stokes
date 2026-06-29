// ns_channel_heat.cpp
//
// First COEFFICIENT-SPACE time loop on a wall (the foundation of the wall-bounded
// Navier-Stokes solver). The unknown is the spline coefficient field alpha, with
// nodal values u = B alpha; nothing forms B^{-1} in the time loop. The implicit
// diffusion step uses the well-conditioned Galerkin channel solver:
//
//   du/dt = nu lap u   ->   (M + nu dt S) alpha^{n+1} = M alpha^n   (implicit Euler)
//
// solved by GalerkinChannelPoisson (FFT x,y + banded SPD z). Validated on a
// channel mode cos(kx x) cos(ky y) cos(m pi z/Lz) (Neumann walls), which decays as
// exp(-nu (kx^2 + ky^2 + (m pi/Lz)^2) t); we report the max error vs that exact
// decay and confirm the discrete energy decays monotonically.

#include <cstdio>
#include <cmath>
#include <vector>
#include "ns_channel.h"

static const double PI = 3.14159265358979323846;

int main() {
    const int N = 32, p = 4;
    const double Lp = 2.0 * PI, Lz = 2.0, nu = 0.02;
    const int kx = 2, ky = 1, mz = 1;
    SimConfig c; c.p = p; c.nu = nu;
    c.ax[0] = {N, Lp, BC::Periodic, true};
    c.ax[1] = {N, Lp, BC::Periodic, true};
    c.ax[2] = {N, Lz, BC::Clamped, true};

    std::printf("==================================================================\n");
    std::printf(" Coefficient-space channel heat equation (u = B alpha)\n");
    std::printf(" implicit diffusion via the Galerkin wall solver; mode decay test\n");
    std::printf("==================================================================\n\n");
    c.print();

    GalerkinChannelPoisson ps; const char* why = nullptr;
    if (!ps.build(c, why)) { std::printf("solver setup failed: %s\n", why); return 1; }

    // Per-axis pieces: FastDeriv (for B and B^{-1}) and mass matrix (for the rhs).
    FastDeriv fx = FastDeriv::build(true,  N, p, Lp);
    FastDeriv fy = FastDeriv::build(true,  N, p, Lp);
    FastDeriv fz = FastDeriv::build(false, N, p, Lz);
    Mat Mx = assemble_forms_periodic(N, p, Lp / N).M;
    Mat My = Mx;
    std::vector<double> knots = open_uniform_knots(N, p, Lz);
    Mat Mz = assemble_forms_clamped(N, p, knots).M;

    // Node coordinates (periodic uniform; clamped Greville) and the initial mode.
    std::vector<double> xz = greville_points(knots, p, N);
    Field3D u(N, N, N), alpha(N, N, N), rhs(N, N, N), tmp(N, N, N);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                u.a[u.idx(i, j, k)] = std::cos(kx * i * Lp / N) * std::cos(ky * j * Lp / N)
                                    * std::cos(mz * PI * xz[k] / Lz);

    // alpha = B^{-1} u  (reduce to coefficients, once, per axis).
    alpha = u;
    nc_axis_fastderiv(alpha, alpha, 0, [&](std::vector<double>& l, std::vector<double>& r){ r = l; fx.reduceB(r); });
    nc_axis_fastderiv(alpha, alpha, 1, [&](std::vector<double>& l, std::vector<double>& r){ r = l; fy.reduceB(r); });
    nc_axis_fastderiv(alpha, alpha, 2, [&](std::vector<double>& l, std::vector<double>& r){ r = l; fz.reduceB(r); });

    auto energy = [&]() {                              // nodal L2 energy of u = B alpha
        Field3D un(N, N, N), t(N, N, N);
        nc_axis_fastderiv(alpha, un, 0, [&](std::vector<double>& l, std::vector<double>& r){ fx.expandB(l, r); });
        nc_axis_fastderiv(un, un, 1, [&](std::vector<double>& l, std::vector<double>& r){ fy.expandB(l, r); });
        nc_axis_fastderiv(un, un, 2, [&](std::vector<double>& l, std::vector<double>& r){ fz.expandB(l, r); });
        double e = 0.0; for (double v : un.a) e += v * v; return e;
    };

    const double lambda = nu * (kx * kx + ky * ky + (mz * PI / Lz) * (mz * PI / Lz));
    const double dt = 2e-3, T = 1.0;
    const int nsteps = (int)std::lround(T / dt);
    const double e0 = energy();
    std::printf("\n decay rate lambda = %.4f   dt = %.4f   steps = %d\n", lambda, dt, nsteps);
    std::printf("   t        energy/E0     exact e^{-2 lambda t}\n");
    for (int n = 0; n <= nsteps; ++n) {
        if (n % (nsteps / 5) == 0) {
            const double t = n * dt;
            std::printf("  %.2f     %.6f      %.6f\n", t, energy() / e0, std::exp(-2 * lambda * t));
        }
        if (n < nsteps) {
            // rhs = M alpha (tensor mass), then (M + nu dt S) alpha = rhs.
            nc_axis_mat(alpha, rhs, 0, Mx);
            nc_axis_mat(rhs, rhs, 1, My);
            nc_axis_mat(rhs, rhs, 2, Mz);
            ps.solve(rhs, alpha, 1.0, nu * dt);
        }
    }

    // Compare nodal u = B alpha to the exact decayed mode.
    Field3D un(N, N, N);
    nc_axis_fastderiv(alpha, un, 0, [&](std::vector<double>& l, std::vector<double>& r){ fx.expandB(l, r); });
    nc_axis_fastderiv(un, un, 1, [&](std::vector<double>& l, std::vector<double>& r){ fy.expandB(l, r); });
    nc_axis_fastderiv(un, un, 2, [&](std::vector<double>& l, std::vector<double>& r){ fz.expandB(l, r); });
    const double decay = std::exp(-lambda * nsteps * dt);
    double err = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double ex = std::cos(kx * i * Lp / N) * std::cos(ky * j * Lp / N)
                                * std::cos(mz * PI * xz[k] / Lz) * decay;
                err = std::fmax(err, std::fabs(un.a[un.idx(i, j, k)] - ex));
            }
    std::printf("\n max|u - exact decayed mode| = %.3e  (init B^-1 + implicit-Euler O(dt) + spatial)\n", err);
    std::printf("\nDone.\n");
    return 0;
}
