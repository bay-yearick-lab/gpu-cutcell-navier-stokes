// acoustic2d.cpp
//
// PHASE 2 verification of the radiation engine (acoustic2d.h): a compact monopole
// (mass source) must radiate a clean 2D cylindrical wave -- amplitude ~ 1/sqrt(r),
// stable, absorbed by the sponge with little reflection. Confirms the open-domain
// propagation before Phase 3 drives the engine with the vortex Lighthill source.

#include "acoustic2d.h"

int main() {
    std::printf("=================================================================\n");
    std::printf(" 2D acoustic radiation engine (monopole) -- sponge open boundary\n");
    std::printf(" verify: 1/sqrt(r) cylindrical spreading + bounded (stable)\n");
    std::printf("=================================================================\n");
    const int N = 128; const double L = 4.0, c = 1.0;
    Acoustic2D ac(N, L, c, 2, A_BMG);                 // BMG2 = BMC5 (6th order, low dispersion)
    const double h = ac.h, x0 = L/2, y0 = L/2;
    const double lam = L/6.0, omega = 2*ACPI*c/lam, ppw = lam/h;
    const double sig = 1.5*h, A = 1.0;
    Field p(N),u(N),v(N),S(N),fz(N),dp(N),du(N),dv(N),t1(N),t2(N);
    Field p1(N),u1(N),v1(N), kp(N),ku(N),kv(N), prms2(N);
    auto srcprofile = [&](Field& Sf, double t){
        for (int j=0;j<N;++j) for (int i=0;i<N;++i){ double x=i*h-x0,y=j*h-y0,r2=x*x+y*y;
            Sf(i,j) = A*std::exp(-r2/(2*sig*sig))*std::sin(omega*t); } };
    const double dt = 0.3*h/c, Tp = 2*ACPI/omega;
    const int nst = (int)std::lround((5.0*Tp + 0.7*L/c)/dt);
    const int nrms = (int)std::lround(Tp/dt);
    std::printf("\n N=%d  lambda=%.3f  PPW=%.1f  dt=%.4f  steps=%d\n", N, lam, ppw, dt, nst);
    auto step = [&](double tn){
        srcprofile(S, tn);            ac.rhs(p,u,v,S,fz,fz,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ kp.a[k]=dp.a[k];ku.a[k]=du.a[k];kv.a[k]=dv.a[k];
            p1.a[k]=p.a[k]+0.5*dt*dp.a[k];u1.a[k]=u.a[k]+0.5*dt*du.a[k];v1.a[k]=v.a[k]+0.5*dt*dv.a[k]; }
        srcprofile(S, tn+0.5*dt);     ac.rhs(p1,u1,v1,S,fz,fz,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ kp.a[k]+=2*dp.a[k];ku.a[k]+=2*du.a[k];kv.a[k]+=2*dv.a[k];
            p1.a[k]=p.a[k]+0.5*dt*dp.a[k];u1.a[k]=u.a[k]+0.5*dt*du.a[k];v1.a[k]=v.a[k]+0.5*dt*dv.a[k]; }
        srcprofile(S, tn+0.5*dt);     ac.rhs(p1,u1,v1,S,fz,fz,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ kp.a[k]+=2*dp.a[k];ku.a[k]+=2*du.a[k];kv.a[k]+=2*dv.a[k];
            p1.a[k]=p.a[k]+dt*dp.a[k];u1.a[k]=u.a[k]+dt*du.a[k];v1.a[k]=v.a[k]+dt*dv.a[k]; }
        srcprofile(S, tn+dt);         ac.rhs(p1,u1,v1,S,fz,fz,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ p.a[k]+=dt/6*(kp.a[k]+dp.a[k]); u.a[k]+=dt/6*(ku.a[k]+du.a[k]); v.a[k]+=dt/6*(kv.a[k]+dv.a[k]); }
    };
    for (int n=0;n<nst;++n){ step(n*dt); if (n>=nst-nrms) for(size_t k=0;k<p.a.size();++k) prms2.a[k]+=p.a[k]*p.a[k]; }
    for (size_t k=0;k<prms2.a.size();++k) prms2.a[k]=std::sqrt(prms2.a[k]/nrms);

    std::printf("\n   r/lambda   p_rms(r)     p_rms*sqrt(r)   (const => 1/sqrt(r) 2D radiation)\n");
    const int nang=720;
    for (double rr : {0.6,0.9,1.2,1.5}) {
        const double r=rr*lam; double s=0; int cnt=0;
        for (int a=0;a<nang;++a){ double th=2*ACPI*a/nang, x=x0+r*std::cos(th), y=y0+r*std::sin(th);
            int i=(int)std::lround(x/h), j=(int)std::lround(y/h);
            if(i>=0&&i<N&&j>=0&&j<N){ s+=prms2(i,j); ++cnt; } }
        const double prms=s/cnt;
        std::printf("   %.2f       %.3e   %.3e\n", rr, prms, prms*std::sqrt(r));
    }
    double emax=0; for(double z:p.a) emax=std::fmax(emax,std::fabs(z));
    std::printf("\n max|p| = %.3e  (bounded => stable + sponge absorbing)\n", emax);
    std::printf("\nDone.\n");
    return 0;
}
