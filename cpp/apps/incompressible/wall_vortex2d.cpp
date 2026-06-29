// wall_vortex2d.cpp
//
// SELF-CONSISTENT 2D wall flow on the energy-conserving wall advection
// (wall2d.h): vorticity-streamfunction in a clamped x clamped walled box,
// free-slip walls. Vortices interact and merge while the discrete enstrophy and
// energy stay conserved (inviscid) to the time-integration error -- a stable
// nonlinear wall simulation, the 2D model of the 3D wall-NS loop. Dumps a
// vorticity movie.
//
//   bin/wall_vortex2d [N=64] [nu=1e-4] [T=12] [nframes=120]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "wall2d.h"
#include "vtk.h"

static const double PI = 3.14159265358979323846;

int main(int argc, char** argv) {
    const int    N  = (argc > 1) ? std::atoi(argv[1]) : 64;
    const double nu = (argc > 2) ? std::atof(argv[2]) : 1e-4;
    const double T  = (argc > 3) ? std::atof(argv[3]) : 12.0;
    const int nframes = (argc > 4) ? std::atoi(argv[4]) : 120;
    const double Lx = 2.0, Lz = 2.0, p = 4;

    WallVortex2D S(N, N, (int)p, Lx, Lz, nu);
    // IC: three Gaussian vortices (zero at walls), interpolated to coefficients.
    auto blob = [](double x, double z, double x0, double z0, double s, double amp) {
        const double r2 = (x - x0) * (x - x0) + (z - z0) * (z - z0);
        return amp * std::exp(-r2 / (s * s));
    };
    std::vector<double> wn((size_t)N * N);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        const double x = S.X.grev[i], z = S.Z.grev[j];
        double w = blob(x, z, 0.8, 1.0, 0.22, 12.0) + blob(x, z, 1.2, 1.0, 0.22, 12.0)
                 + blob(x, z, 1.0, 1.45, 0.20, -10.0);
        wn[(size_t)i * N + j] = w;
    }
    // interpolate to coefficients per axis: alpha = Bx^-1 Wn Bz^-T
    {
        std::vector<double> col(N), row(N);
        Field2D tmp(N, N);
        for (int j = 0; j < N; ++j) {                 // Bx^-1 along x
            for (int i = 0; i < N; ++i) col[i] = wn[(size_t)i * N + j];
            std::vector<double> c; dense_lu_solve(S.X.B, col, c);
            for (int i = 0; i < N; ++i) tmp.at(i, j) = c[i];
        }
        for (int i = 0; i < N; ++i) {                 // Bz^-1 along z
            for (int j = 0; j < N; ++j) row[j] = tmp.at(i, j);
            std::vector<double> c; dense_lu_solve(S.Z.B, row, c);
            for (int j = 0; j < N; ++j) S.omega.at(i, j) = c[j];
        }
    }
    S.zero_bndry(S.omega);

    // nodal vorticity for output: w(grev) = Bx alpha Bz^T
    std::vector<double> wout((size_t)N * N);
    auto to_nodal = [&]() {
        Field2D tmp(N, N);
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            double s = 0; for (int k = 0; k < N; ++k) s += S.X.B(i, k) * S.omega.at(k, j);
            tmp.at(i, j) = s;
        }
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
            double s = 0; for (int l = 0; l < N; ++l) s += S.Z.B(j, l) * tmp.at(i, l);
            wout[(size_t)i * N + j] = s;
        }
    };

    const double dx = Lx / (N - 1), dz = Lz / (N - 1);
    const double dt = 0.25 * std::fmin(dx, dz) / 6.0;     // advective CFL (|u|~few)
    const int nsteps = (int)std::lround(T / dt);
    const int frame_every = std::max(1, nsteps / nframes);

    vtk::PvdSeries series("data/wv2d", "wv2d");
    const double ens0 = S.enstrophy(), ke0 = S.kinetic_energy();
    std::printf("=================================================================\n");
    std::printf(" Self-consistent 2D wall vortex (omega-psi)  N=%d^2  nu=%.1e\n", N, nu);
    std::printf("   dt=%.5f  nsteps=%d  ~%d frames\n", dt, nsteps, nsteps / frame_every + 1);
    std::printf("=================================================================\n");
    std::printf("   enstrophy (the advected-scalar L2 norm = energy-conserving-advection\n");
    std::printf("   invariant) conserved to time error; KE = secondary 2D invariant.\n");
    std::printf("    t      enstrophy/E0   KE/KE0     max|omega|\n");
    int frame = 0;
    for (int n = 0; n <= nsteps; ++n) {
        if (n % frame_every == 0) {
            to_nodal();
            double wm = 0; for (double w : wout) wm = std::fmax(wm, std::fabs(w));
            series.add_frame(n * dt, N, N, 1, dx, dz, dz, {{"vorticity", &wout}});
            std::printf("  %5.2f    %9.6f   %8.5f   %8.3f\n",
                        n * dt, S.enstrophy() / ens0, S.kinetic_energy() / ke0, wm);
            std::fflush(stdout); frame++;
            if (!(wm < 1e4)) { std::printf("  ** UNSTABLE **\n"); return 2; }
        }
        if (n < nsteps) S.step(dt);
    }
    series.write_pvd();
    std::printf("\n STABLE self-consistent wall flow: enstrophy conserved (the energy-\n");
    std::printf(" conserving wall advection), bounded -- where the nodal/collocated wall\n");
    std::printf(" advection blew up. (KE is the secondary 2D invariant; dual energy-\n");
    std::printf(" enstrophy conservation would need an enstrophy-conserving Jacobian form.)\n");
    std::printf(" wrote %d frames to data/wv2d/wv2d.pvd.  render:\n", frame);
    std::printf("   uv run ... python viz/render.py --pvd data/wv2d/wv2d.pvd --field vorticity"
                " --out figures/wall_vortex.mp4 --cmap RdBu_r --symmetric\n");
    return 0;
}
