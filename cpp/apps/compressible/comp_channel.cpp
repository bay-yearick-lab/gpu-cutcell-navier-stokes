// comp_channel.cpp
//
// Compressible no-slip ISOTHERMAL channel (x,y periodic, z clamped walls) on the
// split compressible solver with strong wall BC (compressible.h apply_wall_bc).
// A small shear mode u_x = eps sin(pi z / Lz) (zero at both walls) decays
// viscously. Stability + physical decay are the test: at low Mach the kinetic
// energy should fall like the Stokes rate exp(-2 nu (pi/Lz)^2 t) while total
// energy is conserved (KE -> internal) and density stays healthy.
//
//   bin/comp_channel [Nz=32 Re=100 T=4]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "compressible.h"
#include "bspline.h"

static const double PI = 3.14159265358979323846;

int main(int argc, char** argv) {
    const int Nz   = (argc > 1) ? std::atoi(argv[1]) : 32;
    const double Re = (argc > 2) ? std::atof(argv[2]) : 100.0;
    const double T  = (argc > 3) ? std::atof(argv[3]) : 4.0;

    const int Nx = 16, Ny = 16;
    const double Lx = 2 * PI, Lz = 2.0, gamma = 1.4, R = 1.0, Tw = 1.0;
    const double rho0 = 1.0, eps = 0.05;
    const double mu = rho0 * 1.0 * Lz / Re;             // U=1 ref
    const double nu = mu / rho0;
    const double lam = nu * (PI / Lz) * (PI / Lz);      // Stokes decay rate of the mode

    SimConfig c; c.p = 4; c.route = Route::BMC;
    c.ax[0] = {Nx, Lx, BC::Periodic, true};
    c.ax[1] = {Ny, Lx, BC::Periodic, true};
    c.ax[2] = {Nz, Lz, BC::Clamped,  true};
    CompressibleNS S(c, gamma, R, mu, 0.71);
    S.Twall = Tw;
    if (!S.ready) { std::printf("setup failed: %s\n", S.setup_error); return 1; }

    // node coords (z = Greville for the clamped wall axis)
    std::vector<double> Z = greville_points(open_uniform_knots(Nz, c.p, Lz), c.p, Nz);
    const double p0 = rho0 * R * Tw;
    for (int k = 0; k < Nz; ++k) for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        const double u = eps * std::sin(PI * Z[k] / Lz);     // zero at z=0,Lz
        const size_t id = S.rho.idx(i, j, k);
        S.rho.a[id] = rho0; S.mx.a[id] = rho0 * u; S.my.a[id] = 0; S.mz.a[id] = 0;
        S.Egy.a[id] = p0 / (gamma - 1.0) + 0.5 * rho0 * u * u;
    }
    S.apply_wall_bc();

    // dt: min of acoustic CFL and explicit-viscous limit on the smallest dz.
    const double c0 = std::sqrt(gamma * R * Tw);
    double dzmin = Lz;
    for (int k = 1; k < Nz; ++k) dzmin = std::fmin(dzmin, Z[k] - Z[k - 1]);
    const double dt_ac = 0.4 * dzmin / (c0 + eps);
    const double dt_vis = 0.2 * dzmin * dzmin / nu;
    const double dt = std::fmin(dt_ac, dt_vis);
    const int nsteps = (int)std::lround(T / dt);

    std::printf("=================================================================\n");
    std::printf(" Compressible no-slip isothermal channel  Nz=%d  Re=%.0f\n", Nz, Re);
    std::printf("   mu=%.4g  nu=%.4g  Stokes rate lam=%.4g  c0=%.3f\n", mu, nu, lam, c0);
    std::printf("   dzmin=%.4f  dt=%.5f (ac %.5f, vis %.5f)  nsteps=%d\n",
                dzmin, dt, dt_ac, dt_vis, nsteps);
    std::printf("=================================================================\n");

    auto ke = [&]() { double s = 0; const size_t M = S.N();
        S.primitives();
        for (size_t q = 0; q < M; ++q)
            s += 0.5 * S.rho.a[q] * (S.u.a[q]*S.u.a[q] + S.v.a[q]*S.v.a[q] + S.w.a[q]*S.w.a[q]);
        return s / M; };
    const double ke0 = ke(), e0 = S.total(S.Egy);
    std::printf("    t     KE/KE0    exp(-2lam t)   (KE+IE drift)   max|div u|  min rho\n");
    for (int n = 0; n <= nsteps; ++n) {
        if (n % std::max(1, nsteps / 10) == 0) {
            const double t = n * dt;
            // max|div u|
            S.rhs(); double md = 0; for (double th : S.theta.a) md = std::fmax(md, std::fabs(th));
            std::printf("  %5.2f   %7.4f   %9.4f      %.2e      %.2e   %.4f\n",
                        t, ke() / ke0, std::exp(-2 * lam * t),
                        std::fabs(S.total(S.Egy) - e0), md, S.min_density());
            std::fflush(stdout);
            if (!(S.min_density() > 0) || ke() / ke0 > 1e3) { std::printf("  ** UNSTABLE **\n"); return 2; }
        }
        if (n < nsteps) S.step(dt);
    }
    std::printf("\n stable run. KE decay vs Stokes rate above (low-Mach => ~match).\n");
    return 0;
}
