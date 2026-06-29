// comp_kh.cpp
//
// Compressible Kelvin-Helmholtz instability: a doubly-periodic shear layer rolls
// up into the classic train of vortices. 2D, fully periodic, on the split
// skew-symmetric compressible solver (compressible.h). A clean nonlinear
// showcase of the energy-conserving scheme -- the roll-up stays sharp because
// nothing is artificially dissipated. Dumps a vorticity + density movie.
//
//   bin/comp_kh [N=256] [Mach=0.3] [Re=20000] [T=2.5] [nframes=120]
//
// Render:  uv run ... python viz/render.py --pvd data/kh/kh.pvd --field vorticity
//          --out figures/comp_kh.mp4 --cmap RdBu_r --symmetric

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

int main(int argc, char** argv) {
    const int    N    = (argc > 1) ? std::atoi(argv[1]) : 256;
    const double Mach = (argc > 2) ? std::atof(argv[2]) : 0.3;
    const double Re   = (argc > 3) ? std::atof(argv[3]) : 20000.0;
    const double T    = (argc > 4) ? std::atof(argv[4]) : 2.5;
    const int nframes = (argc > 5) ? std::atoi(argv[5]) : 120;

    const double gamma = 1.4, R = 1.0, L = 1.0, rho0 = 1.0, U = 0.5, delta = 1.0 / 30.0;
    const double c0 = U / Mach, p0 = rho0 * c0 * c0 / gamma;
    const double mu = rho0 * U * L / Re;

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("=================================================================\n");
    std::printf(" Compressible Kelvin-Helmholtz  N=%d^2  M=%.2f  Re=%.0f  (%d thr)\n",
                N, Mach, Re, threads);
    std::printf("=================================================================\n");

    SimConfig c; c.p = 4; c.route = Route::BMC;
    c.ax[0] = {N, L, BC::Periodic, true};
    c.ax[1] = {N, L, BC::Periodic, true};
    c.ax[2] = {1, 1.0, BC::Periodic, false};            // 2D
    CompressibleNS S(c, gamma, R, mu, 0.71);
    if (!S.ready) { std::printf("setup failed: %s\n", S.setup_error); return 1; }

    const double h = L / N, dv = 0.05;
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h;
        const double u = (y < 0.5) ? U * std::tanh((y - 0.25) / delta)
                                   : U * std::tanh((0.75 - y) / delta);
        const double v = dv * std::sin(2 * PI * x / L);
        const size_t id = S.rho.idx(i, j, 0);
        S.rho.a[id] = rho0; S.mx.a[id] = rho0 * u; S.my.a[id] = rho0 * v; S.mz.a[id] = 0;
        S.Egy.a[id] = p0 / (gamma - 1.0) + 0.5 * rho0 * (u * u + v * v);
    }

    const double dt = 0.35 * h / (c0 + U);
    const int nsteps = (int)std::lround(T / dt);
    const int frame_every = std::max(1, nsteps / nframes);

    std::vector<double> omega((size_t)N * N), dens((size_t)N * N);
    vtk::PvdSeries series("data/kh", "kh");
    auto dump = [&](double t) {
        S.rhs();                                         // refresh g (gradients)
        for (int q = 0; q < N * N; ++q) {
            omega[q] = S.g[1][0].a[q] - S.g[0][1].a[q];   // wz = dv/dx - du/dy
            dens[q] = S.rho.a[q];
        }
        series.add_frame(t, N, N, 1, h, h, h, {{"vorticity", &omega}, {"density", &dens}});
    };

    std::printf("\n dt=%.5f  nsteps=%d  ~%d frames\n", dt, nsteps, nsteps / frame_every + 1);
    std::printf("    t      KE          enstrophy    min rho\n");
    auto t0 = Clock::now();
    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % frame_every == 0) {
            dump(n * dt);
            double ens = 0; for (double w : omega) ens += w * w;
            std::printf("  %5.3f   %.6f   %9.2f    %.4f\n",
                        n * dt, S.kinetic_energy(), 0.5 * ens / (N * N), S.min_density());
            std::fflush(stdout); frame++;
            if (!(S.min_density() > 0)) { std::printf("  ** UNSTABLE **\n"); return 2; }
        }
        if (n < nsteps) S.step(dt);
    }
    series.write_pvd();
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("\n wrote %d frames to data/kh/kh.pvd in %.1f s (%.3f s/step)\n",
                frame, wall, wall / nsteps);
    std::printf(" render: uv run ... python viz/render.py --pvd data/kh/kh.pvd "
                "--field vorticity --out figures/comp_kh.mp4 --cmap RdBu_r\n");
    return 0;
}
