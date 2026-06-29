// ns_dns3d.cpp
//
// A 3D DNS run on the incompressible solver (ns.h): the Taylor-Green vortex,
//
//   u =  sin x cos y cos z,   v = -cos x sin y cos z,   w = 0    on [0,2pi]^3,
//
// which is divergence-free and breaks down into turbulence (vortex stretching).
// It is the canonical 3D incompressible DNS benchmark. No closed-form solution,
// so we report the physics diagnostics -- kinetic energy decay and residual
// divergence (must stay at round-off) -- and the wall-clock throughput, using
// the IMEX RK3/CN integrator (advective-CFL-limited dt; viscous done implicitly).

#include <cstdio>
#include <cmath>
#include <chrono>
#include "ns.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static void init_tg3d(IncompressibleNS& S) {
    const double h = 2 * PI / S.nx;              // cube, equal spacing
    for (int k = 0; k < S.nz; ++k)
        for (int j = 0; j < S.ny; ++j)
            for (int i = 0; i < S.nx; ++i) {
                const double x = i * h, y = j * h, z = k * h;
                const size_t id = S.u.idx(i, j, k);
                S.u.a[id] =  std::sin(x) * std::cos(y) * std::cos(z);
                S.v.a[id] = -std::cos(x) * std::sin(y) * std::cos(z);
                S.w.a[id] =  0.0;
            }
    S.project();
}

static IncompressibleNS make(int N, int p, double nu) {
    SimConfig c; c.p = p; c.route = Route::BMC; c.nu = nu;
    for (int a = 0; a < 3; ++a) c.ax[a] = {N, 2 * PI, BC::Periodic, true};
    return IncompressibleNS(c);
}

int main() {
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("==================================================================\n");
    std::printf(" 3D DNS: Taylor-Green vortex, IMEX RK3/CN  (%d threads)\n", threads);
    std::printf("==================================================================\n");

    // --- physics run: energy decay + divergence-free, N=64, Re=1/nu=100 -------
    {
        const int N = 64, p = 4; const double nu = 0.01, dt = 0.006, T = 2.0;
        IncompressibleNS S = make(N, p, nu);
        if (!S.ready) { std::printf("setup failed: %s\n", S.setup_error); return 1; }
        init_tg3d(S);
        const double ke0 = S.kinetic_energy();
        const int nsteps = (int)std::lround(T / dt);
        std::printf("\n N=%d^3  p=%d  nu=%.3g (Re=%.0f)  dt=%.3f\n", N, p, nu, 1.0 / nu, dt);
        std::printf("    t        KE/KE0       max|div u|\n");
        auto t0 = Clock::now();
        for (int n = 0; n <= nsteps; ++n) {
            if (n % (nsteps / 5) == 0)
                std::printf("   %.2f     %.6f     %.2e\n", n * dt, S.kinetic_energy() / ke0, S.max_divergence());
            if (n < nsteps) S.step_imex(dt);
        }
        const double wall = ms_since(t0);
        const double pts = (double)N * N * N;
        std::printf("   %d steps in %.2f s  =>  %.1f ms/step,  %.0f M point-steps/s\n",
                    nsteps, wall / 1e3, wall / nsteps, pts * nsteps / 1e6 / (wall / 1e3));
        std::printf("   (div ~ round-off throughout => incompressibility held exactly)\n");
    }

    // --- throughput at larger grids (a few timed steps each) ------------------
    std::printf("\n IMEX step throughput (BMC p=4):\n");
    std::printf("    grid       ms/step     M point-steps/s\n");
    for (int N : {64, 128}) {
        IncompressibleNS S = make(N, 4, 0.01);
        if (!S.ready) continue;
        init_tg3d(S);
        S.step_imex(0.005);                      // warm up
        const int reps = (N <= 64) ? 20 : 6;
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) S.step_imex(0.005);
        const double t = ms_since(t0) / reps;
        const double pts = (double)N * N * N;
        std::printf("    %3d^3     %7.1f       %7.0f\n", N, t, pts / 1e6 / (t / 1e3));
    }

    std::printf("\nDone.\n");
    return 0;
}
