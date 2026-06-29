// comp_tgv.cpp
//
// Compressible Taylor-Green vortex on the split skew-symmetric compressible
// solver (compressible.h). The canonical compressible-DNS transition: the same
// vortex roll-up and turbulent breakdown as the incompressible TGV, but now with
// genuine ACOUSTIC / dilatational structure (theta = div u radiating sound).
// Reports the full energy budget (KE + internal = total, conserved inviscidly to
// round-off) and dumps a movie of vorticity |omega| and dilatation theta.
//
//   bin/comp_tgv [N=96] [Mach=0.4] [Re=1600] [T=10] [nframes=80]
//
// Output: data/ctgv/ctgv_XXXX.vti + ctgv.pvd  ->  make video PVD=data/ctgv/ctgv.pvd

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include "compressible.h"
#include "vtk.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;
static double s_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const int    N    = (argc > 1) ? std::atoi(argv[1]) : 96;
    const double Mach = (argc > 2) ? std::atof(argv[2]) : 0.4;
    const double Re   = (argc > 3) ? std::atof(argv[3]) : 1600.0;
    const double T    = (argc > 4) ? std::atof(argv[4]) : 10.0;
    const int nframes = (argc > 5) ? std::atoi(argv[5]) : 80;

    const double gamma = 1.4, Rgas = 1.0, V0 = 1.0, rho0 = 1.0;
    const double c0 = V0 / Mach, p0 = rho0 * c0 * c0 / gamma, T0 = p0 / (rho0 * Rgas);
    const double mu = rho0 * V0 / Re;          // L=1 reference

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("=================================================================\n");
    std::printf(" Compressible Taylor-Green  N=%d^3  M=%.2f  Re=%.0f  (%d threads)\n",
                N, Mach, Re, threads);
    std::printf("=================================================================\n");

    SimConfig c; c.p = 4; c.route = Route::BMC; c.nu = 0.0;
    for (int a = 0; a < 3; ++a) c.ax[a] = {N, 2 * PI, BC::Periodic, true};
    CompressibleNS S(c, gamma, Rgas, mu, /*Pr=*/0.71);
    if (!S.ready) { std::printf("setup failed: %s\n", S.setup_error); return 1; }

    const double h = 2 * PI / N;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h, z = k * h;
        const double u =  V0 * std::sin(x) * std::cos(y) * std::cos(z);
        const double v = -V0 * std::cos(x) * std::sin(y) * std::cos(z);
        const double p = p0 + (rho0 * V0 * V0 / 16.0)
                         * (std::cos(2 * x) + std::cos(2 * y)) * (std::cos(2 * z) + 2.0);
        const double rho = p / (Rgas * T0);                 // isothermal init
        const size_t id = S.rho.idx(i, j, k);
        S.rho.a[id] = rho; S.mx.a[id] = rho * u; S.my.a[id] = rho * v; S.mz.a[id] = 0.0;
        S.Egy.a[id] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }

    const double dt = 0.5 * h / S.max_wavespeed();          // acoustic CFL
    const int nsteps = (int)std::lround(T / dt);
    const int frame_every = std::max(1, nsteps / nframes);

    std::vector<double> omag((size_t)N * N * N), dil((size_t)N * N * N);
    vtk::PvdSeries series("data/ctgv", "ctgv");
    auto diag = [&](double& ke, double& ie, double& ens, double& maxdil) {
        S.rhs();   // refresh primitives + gradients (g, theta)
        const size_t M = S.N();
        ke = ie = ens = 0.0; maxdil = 0.0;
        for (size_t q = 0; q < M; ++q) {
            const double r = S.rho.a[q];
            ke += 0.5 * r * (S.u.a[q] * S.u.a[q] + S.v.a[q] * S.v.a[q] + S.w.a[q] * S.w.a[q]);
            ie += S.pr.a[q] / (gamma - 1.0);
            const double ox = S.g[2][1].a[q] - S.g[1][2].a[q];
            const double oy = S.g[0][2].a[q] - S.g[2][0].a[q];
            const double oz = S.g[1][0].a[q] - S.g[0][1].a[q];
            omag[q] = std::sqrt(ox * ox + oy * oy + oz * oz);
            dil[q] = S.theta.a[q];
            ens += omag[q] * omag[q];
            if (std::fabs(dil[q]) > maxdil) maxdil = std::fabs(dil[q]);
        }
        ke /= M; ie /= M; ens = 0.5 * ens / M;
    };

    double ke0, ie0, e0d, md;
    diag(ke0, ie0, e0d, md);
    const double tot0 = ke0 + ie0;
    std::printf("\n dt=%.4f  nsteps=%d  c0=%.2f  ~%d frames\n", dt, nsteps, c0, nsteps / frame_every + 1);
    std::printf("    t      KE/KE0    (KE+IE)/tot0   enstrophy  max|div u|  min rho\n");

    auto t0 = Clock::now();
    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % frame_every == 0) {
            double ke, ie, ens, maxd; diag(ke, ie, ens, maxd);
            series.add_frame(n * dt, N, N, N, h, h, h,
                             {{"vorticity", &omag}, {"dilatation", &dil}});
            std::printf("  %5.2f   %7.4f   %10.7f   %8.3f   %.2e   %.4f\n",
                        n * dt, ke / ke0, (ke + ie) / tot0, ens, maxd, S.min_density());
            std::fflush(stdout); frame++;
        }
        if (n < nsteps) S.step(dt);
    }
    series.write_pvd();
    std::printf("\n wrote %d frames to data/ctgv/ctgv.pvd in %.1f s (%.3f s/step)\n",
                frame, s_since(t0), s_since(t0) / nsteps);
    std::printf(" render:  uv run ... python viz/render.py --pvd data/ctgv/ctgv.pvd "
                "--field dilatation --out figures/comp_tgv.mp4\n");
    return 0;
}
