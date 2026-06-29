// channel_dns.cpp
//
// TURBULENT CHANNEL DNS -- the canonical wall-bounded benchmark -- on the
// coefficient-space primitive wall-NS solver (wall_ns3d.h): x,y PERIODIC, z
// NO-SLIP WALLS. The fast-diagonalization automatically uses Fourier modes in
// the periodic directions (eigenvectors of the circulant operators) and pins the
// const + Nyquist pressure nulls (the checkerboard); the wall (z) direction is
// the inf-sup-stable Galerkin Schur projection. Energy-conserving sum-factorized
// advection + implicit viscous. force=1 sustains the turbulence. Vorticity movie
// + turbulence statistics.
//
//   bin/channel_dns [N=32] [Re=2000] [T=10] [nframes=90] [p=3] [force=1]

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
static double Lp = 2 * PI, Lz = 2.0;             // periodic box (x,y), wall gap (z)

// periodic in x,y; vanishes at the z-walls (no-slip)
static double icu(double x, double y, double z) {
    return (std::sin(x)*std::cos(y) + 0.6*std::sin(2*x)*std::cos(2*y)) * std::sin(PI*z/Lz);
}
static double icv(double x, double y, double z) {
    return (std::cos(x)*std::sin(y) - 0.6*std::cos(2*x)*std::sin(2*y)) * std::sin(PI*z/Lz);
}
static double icw(double x, double y, double z) {
    return std::sin(x)*std::sin(y) * std::sin(2*PI*z/Lz);
}
static void interp(WallNS3D& S, double (*f)(double,double,double), Field3& out) {
    const int nx=S.X.n, ny=S.Y.n, nz=S.Z.n; Field3 g(nx,ny,nz);
    for (int i=0;i<nx;++i) for (int j=0;j<ny;++j) for (int k=0;k<nz;++k)
        g.at(i,j,k)=f(S.X.grev[i],S.Y.grev[j],S.Z.grev[k]);
    std::vector<double> ln;
    for (int j=0;j<ny;++j) for (int k=0;k<nz;++k){ ln.assign(nx,0); for(int i=0;i<nx;++i) ln[i]=g.at(i,j,k);
        std::vector<double> c; dense_lu_solve(S.X.B,ln,c); for(int i=0;i<nx;++i) g.at(i,j,k)=c[i]; }
    for (int i=0;i<nx;++i) for (int k=0;k<nz;++k){ ln.assign(ny,0); for(int j=0;j<ny;++j) ln[j]=g.at(i,j,k);
        std::vector<double> c; dense_lu_solve(S.Y.B,ln,c); for(int j=0;j<ny;++j) g.at(i,j,k)=c[j]; }
    for (int i=0;i<nx;++i) for (int j=0;j<ny;++j){ ln.assign(nz,0); for(int k=0;k<nz;++k) ln[k]=g.at(i,j,k);
        std::vector<double> c; dense_lu_solve(S.Z.B,ln,c); for(int k=0;k<nz;++k) g.at(i,j,k)=c[k]; }
    out=g;
}

int main(int argc, char** argv) {
    const int    N  = (argc>1)?std::atoi(argv[1]):32;
    const double Re = (argc>2)?std::atof(argv[2]):2000.0;
    const double T  = (argc>3)?std::atof(argv[3]):10.0;
    const int nframes = (argc>4)?std::atoi(argv[4]):90;
    const int    p  = (argc>5)?std::atoi(argv[5]):3;       // ODD p (periodic basis)
    const int force = (argc>6)?std::atoi(argv[6]):1;
    const double nu = Lz * 1.0 / Re;

    int threads=1;
#ifdef _OPENMP
    threads=omp_get_max_threads();
#endif
    std::printf("=================================================================\n");
    std::printf(" TURBULENT CHANNEL DNS  N=%d^3  p=%d  Re=%.0f  (x,y periodic, z wall, %d thr)\n",
                N, p, Re, threads);
    std::printf("=================================================================\n");

    WallNS3D S(N, N, N, p, Lp, Lp, Lz, nu, /*px*/true, /*py*/true, /*pz*/false);
    interp(S, icu, S.u); interp(S, icv, S.v); interp(S, icw, S.w);
    S.no_slip(S.u); S.no_slip(S.v); S.no_slip(S.w); S.project();
    {
        const double e0=S.kinetic_energy(), sc=std::sqrt(0.5/e0);
        for (auto& x:S.u.a) x*=sc;
        for (auto& x:S.v.a) x*=sc;
        for (auto& x:S.w.a) x*=sc;
    }

    const double dx=Lp/N, dz=Lz/N, dt=0.3*std::fmin(dx,dz);
    const int nsteps=(int)std::lround(T/dt);
    const int frame_every=std::max(1,nsteps/nframes);
    const double ke0=S.kinetic_energy();
    const double V=Lp*Lp*Lz;
    double ens_sum=0, ke_sum=0; int nstat=0;

    vtk::PvdSeries series("data/chan", "chan");
    std::vector<double> mag;
    std::printf("\n dt=%.4f nsteps=%d ~%d frames\n", dt, nsteps, nsteps/frame_every+1);
    std::printf("    t     KE/KE0   enstrophy  max|wdiv|  max|w|\n");
    auto t0=Clock::now(); int frame=0;
    for (int n=0;n<=nsteps;++n){
        if (n%frame_every==0){
            S.vorticity_mag(mag);
            double wm=0,ens=0; for(double m:mag){wm=std::fmax(wm,m); ens+=m*m;} ens=0.5*ens/mag.size();
            series.add_frame(n*dt, N,N,N, dx,dx,dz, {{"vorticity",&mag}});
            std::printf("  %5.2f   %.4f   %8.2f   %.1e   %.2f\n", n*dt, S.kinetic_energy()/ke0, ens, S.max_weak_div(), wm);
            std::fflush(stdout); ++frame;
            if (n*dt>0.5*T){ ens_sum+=ens; ke_sum+=S.kinetic_energy()/V; ++nstat; }
            double um=0; for(double x:S.u.a) um=std::fmax(um,std::fabs(x));
            if (!(um<1e3)){ std::printf("  ** UNSTABLE **\n"); return 2; }
        }
        if (n<nsteps){
            S.step_imex(dt);
            if (force){
                double f=std::sqrt(ke0/std::fmax(S.kinetic_energy(),1e-30));
                f=std::fmin(std::fmax(f,0.97),1.06);
                for(auto&x:S.u.a)x*=f;
                for(auto&x:S.v.a)x*=f;
                for(auto&x:S.w.a)x*=f;
            }
        }
    }
    series.write_pvd();
    if (force && nstat>0){
        const double ens=ens_sum/nstat, ke=ke_sum/nstat;
        const double diss=2*nu*ens, up2=2*ke/3, ReLam=up2*std::sqrt(15.0/(nu*diss));
        std::printf("\n --- channel turbulence (steady state) --- Re_lambda=%.1f  eps=%.4g  u'=%.3f\n",
                    ReLam, diss, std::sqrt(up2));
    }
    const double wall=std::chrono::duration<double>(Clock::now()-t0).count();
    std::printf(" wrote %d frames to data/chan/chan.pvd in %.1f s (%.2f s/step)\n", frame, wall, wall/nsteps);
    std::printf(" render: uv run ... python viz/render.py --pvd data/chan/chan.pvd "
                "--field vorticity --out figures/channel.mp4\n");
    return 0;
}
