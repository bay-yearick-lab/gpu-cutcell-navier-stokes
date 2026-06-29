// ns_reconnection.cpp
//
// FREAKY DNS #1: antiparallel vortex-tube RECONNECTION -- the canonical hard DNS
// whose peak-vorticity amplification omega_max(t)/omega_0 is acutely resolution
// sensitive (thin vorticity bridges form). The cost story: our spectral-class
// banded scheme captures the peak at a COARSE grid where matched-order finite
// differences badly under-predict it, matching a fine reference -> same physics,
// far fewer points (the reason to use these schemes for scalable / wall-bound DNS,
// where Fourier pseudospectral cannot go).
//
// This program sets up the flow and reports the resolution-sensitive diagnostics
// (peak |omega|, enstrophy, energy, divergence). Initial condition: two Gaussian
// antiparallel tubes along x at y = +/- b, sinusoidally drawn together near x=0;
// built divergence-free via omega -> (clean div) -> psi = lap^{-1}(-omega) ->
// u = curl(psi).

#include <cstdio>
#include <cmath>
#include <string>
#include "ns.h"

static const double PI = 3.14159265358979323846;

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    std::string rstr = (argc > 2) ? argv[2] : "bmc";
    Route route = (rstr=="fd6")?Route::FD6 : (rstr=="fd4")?Route::FD4 : (rstr=="fd2")?Route::FD2
                : (rstr=="c6")?Route::C6 : (rstr=="drp")?Route::DRP
                : (rstr=="bmg")?Route::BMG : Route::BMC;
    const double Re = (argc > 3) ? std::atof(argv[3]) : 2000.0;
    const int pdeg = (argc > 5) ? std::atoi(argv[5]) : 5;  // spline degree (BMC p: 5,7,9,11 = BMG2,3,4,5)
    const double L = 2*PI, Gam = 1.0, nu = Gam/Re;
    const double b = 0.9, acore = 0.45, del = 0.45;       // separation, core, perturbation
    SimConfig c; c.p = pdeg; c.nu = nu; c.route = route;  // higher degree -> higher order + resolution
    c.ax[0] = {N, L, BC::Periodic, true};
    c.ax[1] = {N, L, BC::Periodic, true};
    c.ax[2] = {N, L, BC::Periodic, true};
    std::printf("=================================================================\n");
    std::printf(" Antiparallel vortex reconnection DNS  (N=%d, Re=%.0f, route=%s p=%d)\n",
                N, Re, route_name(route), c.p);
    std::printf("=================================================================\n");

    IncompressibleNS ns(c);
    if (!ns.ready) { std::printf("setup failed: %s\n", ns.setup_error); return 1; }
    const char* why = nullptr; SeparablePoisson lap;
    if (!lap.build(c, why, /*divgrad=*/false)) { std::printf("poisson: %s\n", why); return 1; }
    const double h = L/N;
    Field3D wx(N,N,N), wy(N,N,N), wz(N,N,N), px(N,N,N), py(N,N,N), pz(N,N,N), tA(N,N,N), tB(N,N,N);

    // Deposit vorticity along the two perturbed tube tangents (antiparallel).
    for (int k=0;k<N;++k) for (int j=0;j<N;++j) for (int i=0;i<N;++i) {
        const double x=i*h, y=j*h-PI, z=k*h-PI;                 // centered y,z in [-pi,pi)
        double Wx=0,Wy=0,Wz=0;
        for (int s=-1;s<=1;s+=2) {                              // s=+1 upper tube, s=-1 lower
            const double yc = s*(b - del*std::cos(x));          // centerline drawn together near x=0
            const double dyc = s*del*std::sin(x);               // dyc/dx
            double tx=1.0, ty=dyc, tz=0.0, tn=std::sqrt(tx*tx+ty*ty+tz*tz);
            tx/=tn; ty/=tn; tz/=tn;                             // unit tangent
            const double r2=(y-yc)*(y-yc)+z*z;                  // distance^2 from centerline
            const double mag = s*(Gam/(PI*acore*acore))*std::exp(-r2/(acore*acore));
            Wx+=mag*tx; Wy+=mag*ty; Wz+=mag*tz;
        }
        wx.a[wx.idx(i,j,k)]=Wx; wy.a[wy.idx(i,j,k)]=Wy; wz.a[wz.idx(i,j,k)]=Wz;
    }
    // Clean divergence of omega: chi = lap^{-1}(div omega); omega -= grad chi.
    ns.D.divergence(wx,wy,wz,tA,tB); lap.solve(tA,tB,0.0,1.0);
    ns.D.deriv(tB,tA,0,1); for(size_t q=0;q<wx.a.size();++q) wx.a[q]-=tA.a[q];
    ns.D.deriv(tB,tA,1,1); for(size_t q=0;q<wy.a.size();++q) wy.a[q]-=tA.a[q];
    ns.D.deriv(tB,tA,2,1); for(size_t q=0;q<wz.a.size();++q) wz.a[q]-=tA.a[q];
    // psi_i = lap^{-1}(-omega_i)
    for(size_t q=0;q<wx.a.size();++q){ wx.a[q]=-wx.a[q]; wy.a[q]=-wy.a[q]; wz.a[q]=-wz.a[q]; }
    lap.solve(wx,px,0.0,1.0); lap.solve(wy,py,0.0,1.0); lap.solve(wz,pz,0.0,1.0);
    // u = curl(psi)
    ns.D.deriv(pz,tA,1,1); ns.D.deriv(py,tB,2,1); for(size_t q=0;q<ns.u.a.size();++q) ns.u.a[q]=tA.a[q]-tB.a[q];
    ns.D.deriv(px,tA,2,1); ns.D.deriv(pz,tB,0,1); for(size_t q=0;q<ns.v.a.size();++q) ns.v.a[q]=tA.a[q]-tB.a[q];
    ns.D.deriv(py,tA,0,1); ns.D.deriv(px,tB,1,1); for(size_t q=0;q<ns.w.a.size();++q) ns.w.a[q]=tA.a[q]-tB.a[q];
    ns.project();

    // Vorticity diagnostics: peak |omega|, enstrophy = sum|omega|^2 * h^3, energy.
    auto vortdiag = [&](double& peak, double& ens){
        ns.D.deriv(ns.w,tA,1,1); ns.D.deriv(ns.v,tB,2,1); for(size_t q=0;q<wx.a.size();++q) wx.a[q]=tA.a[q]-tB.a[q];
        ns.D.deriv(ns.u,tA,2,1); ns.D.deriv(ns.w,tB,0,1); for(size_t q=0;q<wy.a.size();++q) wy.a[q]=tA.a[q]-tB.a[q];
        ns.D.deriv(ns.v,tA,0,1); ns.D.deriv(ns.u,tB,1,1); for(size_t q=0;q<wz.a.size();++q) wz.a[q]=tA.a[q]-tB.a[q];
        peak=0; ens=0; for(size_t q=0;q<wx.a.size();++q){ double m2=wx.a[q]*wx.a[q]+wy.a[q]*wy.a[q]+wz.a[q]*wz.a[q];
            ens+=m2; peak=std::fmax(peak,std::sqrt(m2)); } ens*=h*h*h;
    };
    double peak0,ens0; vortdiag(peak0,ens0);
    const double E0 = ns.kinetic_energy();
    std::printf("\n init: peak|omega|=%.4f  enstrophy=%.4f  KE=%.4f  div=%.2e\n",
                peak0, ens0, E0, ns.max_divergence());

    const double umax = peak0*acore;                          // ~ velocity scale
    const double dt = 0.4 * h / std::fmax(umax, 0.2);
    const double Tend = (argc > 4) ? std::atof(argv[4]) : 50.0;
    const int nst = (int)std::lround(Tend/dt);
    std::printf(" dt=%.4f steps=%d  (advance to t=%.0f)\n", dt, nst, Tend);
    std::printf("\n   t      peak|omega|   enstrophy   KE/KE0    max|div|   (peak0=%.4f)\n", peak0);
    for (int n=0;n<=nst;++n){
        if (n % (nst/25) == 0) {
            double pk,en; vortdiag(pk,en);
            std::printf("  %5.1f    %.5f     %.5f    %.5f   %.1e\n",
                        n*dt, pk, en, ns.kinetic_energy()/E0, ns.max_divergence());
        }
        if (n<nst) ns.step(dt);          // explicit SSP-RK3 (cheaper; viscous CFL fine at these Re)
    }
    std::printf("\nDone.\n");
    return 0;
}
