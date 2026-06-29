// dns_movie.cpp
//
// 3D Taylor-Green vortex DNS that DUMPS A MOVIE. The classic transition to
// turbulence (smooth vortex sheets roll up, stretch, and break into a tangle of
// vorticity worms). Energy-conserving skew-symmetric BMC advection + implicit
// IMEX viscous (ns.h). Every few steps it computes the vorticity magnitude and
// the Q-criterion and writes a ParaView .vti frame (io/vtk.h); a .pvd ties them
// into an animation. viz/render.py turns the same frames into an mp4.
//
//   bin/dns_movie [N=128] [Re=1600] [T=10] [nframes=100]
//
// Output: data/turb/turb_XXXX.vti  +  data/turb/turb.pvd
// Fields per frame: "vorticity" (|omega|), "Q" (Q-criterion).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include "ns.h"
#include "vtk.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;
static double s_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static IncompressibleNS make(int N, double nu) {
    SimConfig c; c.p = 4; c.route = Route::BMC; c.nu = nu;
    for (int a = 0; a < 3; ++a) c.ax[a] = {N, 2 * PI, BC::Periodic, true};
    return IncompressibleNS(c);
}

static void init_tg3d(IncompressibleNS& S) {
    const double h = 2 * PI / S.nx;
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

int main(int argc, char** argv) {
    const int    N       = (argc > 1) ? std::atoi(argv[1]) : 128;
    const double Re      = (argc > 2) ? std::atof(argv[2]) : 1600.0;
    const double T       = (argc > 3) ? std::atof(argv[3]) : 10.0;
    const int    nframes = (argc > 4) ? std::atoi(argv[4]) : 100;
    const double nu = 1.0 / Re;

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("=================================================================\n");
    std::printf(" Taylor-Green DNS movie  N=%d^3  Re=%.0f  T=%.1f  (%d threads)\n",
                N, Re, T, threads);
    std::printf("=================================================================\n");

    IncompressibleNS S = make(N, nu);
    if (!S.ready) { std::printf("setup failed: %s\n", S.setup_error); return 1; }
    init_tg3d(S);

    const double dx = 2 * PI / N;
    const double dt = 0.4 * dx;                       // advective CFL (|u|<=1)
    const int nsteps = (int)std::lround(T / dt);
    const int frame_every = std::max(1, nsteps / nframes);

    // vorticity / Q work fields (the 9 velocity gradients reused per frame).
    Field3D g[3][3];
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) g[a][b] = Field3D(N, N, N);
    std::vector<double> omag((size_t)N * N * N), qcrit((size_t)N * N * N);

    vtk::PvdSeries series("data/turb", "turb");

    auto compute_fields = [&]() {
        const Field3D* U[3] = {&S.u, &S.v, &S.w};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) S.D.deriv(*U[a], g[a][b], b, 1);  // du_a/dx_b
        const size_t M = omag.size();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            const double ox = g[2][1].a[i] - g[1][2].a[i];   // w_y - v_z
            const double oy = g[0][2].a[i] - g[2][0].a[i];   // u_z - w_x
            const double oz = g[1][0].a[i] - g[0][1].a[i];   // v_x - u_y
            omag[i] = std::sqrt(ox * ox + oy * oy + oz * oz);
            // Q = -1/2 sum_ij g_ij g_ji
            double q = 0.0;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) q += g[a][b].a[i] * g[b][a].a[i];
            qcrit[i] = -0.5 * q;
        }
    };
    auto enstrophy = [&]() {
        double s = 0.0; for (double x : omag) s += x * x; return 0.5 * s / omag.size();
    };

    const double ke0 = S.kinetic_energy();
    std::printf("\n dt=%.4f  nsteps=%d  frame_every=%d  -> ~%d frames\n",
                dt, nsteps, frame_every, nsteps / frame_every + 1);
    std::printf("    t      KE/KE0    enstrophy    max|div|    frame\n");

    auto t0 = Clock::now();
    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % frame_every == 0) {
            compute_fields();
            series.add_frame(n * dt, N, N, N, dx, dx, dx,
                             {{"vorticity", &omag}, {"Q", &qcrit}});
            std::printf("  %5.2f   %7.4f   %9.3f   %.1e   #%d\n",
                        n * dt, S.kinetic_energy() / ke0, enstrophy(),
                        S.max_divergence(), frame++);
            std::fflush(stdout);
        }
        if (n < nsteps) S.step_imex(dt);
    }
    series.write_pvd();
    std::printf("\n wrote %d frames to data/turb/turb.pvd in %.1f s (%.2f s/step)\n",
                frame, s_since(t0), s_since(t0) / nsteps);
    std::printf(" open data/turb/turb.pvd in ParaView, or: make video\n");
    return 0;
}
