// wall_dns3d.cpp
//
// 3D WALL-BOUNDED DNS on the coefficient-space primitive wall-NS solver
// (wall_ns3d.h, IMEX step): an energetic no-slip box flow, kept stable and
// (weakly) divergence-free by the energy-conserving Galerkin advection + the
// fast-diagonalization Schur projection + implicit viscous. force=1 sustains the
// kinetic energy (forced turbulence -> statistical steady state) and prints the
// turbulence characterization (dissipation, Re_lambda, Kolmogorov resolution).
// Dumps a vorticity movie. The wall-NS frontier, running fast in 3D.
//
//   bin/wall_dns3d [N=32] [Re=1000] [T=8] [nframes=80] [p=2] [force=0]
//
// Output: data/wdns/wdns_XXXX.vti + wdns.pvd ; render with viz/render.py.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include "wall_ns3d.h"
#include "vtk.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;
static double L = 2.0;

// energetic multi-mode no-slip IC (all components vanish at the walls)
static double icu(double x, double y, double z) {
    return std::sin(PI*x/L)*std::sin(2*PI*y/L)*std::sin(PI*z/L)
         + 0.7*std::sin(2*PI*x/L)*std::sin(PI*y/L)*std::sin(2*PI*z/L);
}
static double icv(double x, double y, double z) {
    return std::sin(2*PI*x/L)*std::sin(PI*y/L)*std::sin(PI*z/L)
         - 0.7*std::sin(PI*x/L)*std::sin(2*PI*y/L)*std::sin(2*PI*z/L);
}
static double icw(double x, double y, double z) {
    return std::sin(PI*x/L)*std::sin(PI*y/L)*std::sin(2*PI*z/L)
         + 0.5*std::sin(2*PI*x/L)*std::sin(2*PI*y/L)*std::sin(PI*z/L);
}

static void interp(WallNS3D& S, double (*f)(double,double,double), Field3& out) {
    const int nx = S.X.n, ny = S.Y.n, nz = S.Z.n;
    Field3 g(nx, ny, nz);
    for (int i=0;i<nx;++i) for (int j=0;j<ny;++j) for (int k=0;k<nz;++k)
        g.at(i,j,k) = f(S.X.grev[i], S.Y.grev[j], S.Z.grev[k]);
    std::vector<double> ln;
    for (int j=0;j<ny;++j) for (int k=0;k<nz;++k) { ln.assign(nx,0); for(int i=0;i<nx;++i) ln[i]=g.at(i,j,k);
        std::vector<double> c; dense_lu_solve(S.X.B, ln, c); for(int i=0;i<nx;++i) g.at(i,j,k)=c[i]; }
    for (int i=0;i<nx;++i) for (int k=0;k<nz;++k) { ln.assign(ny,0); for(int j=0;j<ny;++j) ln[j]=g.at(i,j,k);
        std::vector<double> c; dense_lu_solve(S.Y.B, ln, c); for(int j=0;j<ny;++j) g.at(i,j,k)=c[j]; }
    for (int i=0;i<nx;++i) for (int j=0;j<ny;++j) { ln.assign(nz,0); for(int k=0;k<nz;++k) ln[k]=g.at(i,j,k);
        std::vector<double> c; dense_lu_solve(S.Z.B, ln, c); for(int k=0;k<nz;++k) g.at(i,j,k)=c[k]; }
    out = g;
}

int main(int argc, char** argv) {
    const int    N  = (argc>1)?std::atoi(argv[1]):32;
    const double Re = (argc>2)?std::atof(argv[2]):1000.0;
    const double T  = (argc>3)?std::atof(argv[3]):8.0;
    const int nframes = (argc>4)?std::atoi(argv[4]):80;
    const int    p  = (argc>5)?std::atoi(argv[5]):2;
    const int force = (argc>6)?std::atoi(argv[6]):0;   // 1 = sustain KE (forced turbulence)
    const double nu = 1.0 * L / Re;          // U=1 ref

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("=================================================================\n");
    std::printf(" 3D WALL-BOUNDED DNS  N=%d^3  p=%d  Re=%.0f  (%d threads)\n", N, p, Re, threads);
    std::printf("=================================================================\n");

    WallNS3D S(N, N, N, p, L, L, L, nu);
    interp(S, icu, S.u); interp(S, icv, S.v); interp(S, icw, S.w);
    S.no_slip(S.u); S.no_slip(S.v); S.no_slip(S.w); S.project();
    // normalize to KE ~ 0.5
    {
        const double e0 = S.kinetic_energy(), sc = std::sqrt(0.5 / e0);
        for (auto& x : S.u.a) x *= sc;
        for (auto& x : S.v.a) x *= sc;
        for (auto& x : S.w.a) x *= sc;
    }

    const double dz = L / N, dt = 0.3 * dz;      // advective CFL (implicit viscous: no viscous-CFL)
    const int nsteps = (int)std::lround(T / dt);
    const int frame_every = std::max(1, nsteps / nframes);
    const double ke0 = S.kinetic_energy();

    vtk::PvdSeries series("data/wdns", "wdns");
    std::vector<double> mag;
    std::printf("\n dt=%.4f nsteps=%d ~%d frames\n", dt, nsteps, nsteps/frame_every+1);
    std::printf("    t     KE/KE0   enstrophy  max|wdiv|  max|w|\n");
    auto t0 = Clock::now();
    int frame = 0;
    const double V = L * L * L;
    double ens_sum = 0, ke_sum = 0; int nstat = 0;     // steady-state averages (second half)
    for (int n = 0; n <= nsteps; ++n) {
        if (n % frame_every == 0) {
            S.vorticity_mag(mag);
            double wm = 0, ens = 0; for (double m : mag) { wm = std::fmax(wm, m); ens += m*m; }
            ens = 0.5 * ens / mag.size();              // mean enstrophy density = 0.5<w^2>
            series.add_frame(n*dt, N, N, N, dz, dz, dz, {{"vorticity", &mag}});
            std::printf("  %5.2f   %.4f   %8.2f   %.1e   %.2f\n",
                        n*dt, S.kinetic_energy()/ke0, ens, S.max_weak_div(), wm);
            std::fflush(stdout); ++frame;
            if (n * dt > 0.5 * T) { ens_sum += ens; ke_sum += S.kinetic_energy() / V; ++nstat; }
            double um=0; for (double x: S.u.a) um=std::fmax(um,std::fabs(x));
            if (!(um < 1e3)) { std::printf("  ** UNSTABLE **\n"); return 2; }
        }
        if (n < nsteps) {
            S.step_imex(dt);
            if (force) {                 // sustain KE: gentle capped rescale (energy injection
                double f = std::sqrt(ke0 / std::fmax(S.kinetic_energy(), 1e-30));   // toward target)
                f = std::fmin(std::fmax(f, 0.97), 1.06);
                for (auto& x : S.u.a) x *= f;
                for (auto& x : S.v.a) x *= f;
                for (auto& x : S.w.a) x *= f;
            }
        }
    }
    series.write_pvd();
    if (force && nstat > 0) {                          // turbulence characterization (steady state)
        const double ens = ens_sum / nstat, ke = ke_sum / nstat;   // mean densities
        const double diss = 2.0 * nu * ens;            // eps = nu <w^2> = 2 nu * enstrophy
        const double up2 = 2.0 * ke / 3.0;             // u'^2 = <|u|^2>/3 = 2 ke /3
        const double ReLam = up2 * std::sqrt(15.0 / (nu * diss));
        const double eta = std::pow(nu * nu * nu / diss, 0.25);   // Kolmogorov scale
        std::printf("\n --- forced turbulence (steady-state averages) ---\n");
        std::printf("   dissipation eps = %.4g   u' = %.3f   Re_lambda = %.1f\n",
                    diss, std::sqrt(up2), ReLam);
        std::printf("   Kolmogorov eta = %.4f   grid dz = %.4f   dz/eta = %.2f (resolution)\n",
                    eta, dz, dz / eta);
        std::printf("   => a genuine sustained turbulent state at Re_lambda ~ %.0f.\n", ReLam);
    }
    const double wall = std::chrono::duration<double>(Clock::now()-t0).count();
    std::printf("\n wrote %d frames to data/wdns/wdns.pvd in %.1f s (%.2f s/step)\n",
                frame, wall, wall/nsteps);
    std::printf(" render: uv run ... python viz/render.py --pvd data/wdns/wdns.pvd "
                "--field vorticity --out figures/wall_dns3d.mp4\n");
    return 0;
}
