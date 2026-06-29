// acoustic_vortex.cpp
//
// PHASE 3: sound radiated by a CO-ROTATING VORTEX PAIR -- the canonical
// computational-aeroacoustics verification case (a standard
// aeroacoustics benchmark). Two equal Lamb-Oseen vortices orbit at angular
// rate Omega and radiate a spinning QUADRUPOLE at frequency 2*Omega.
//
// Hybrid (Lighthill analogy): the incompressible near-field U of the rotating pair
// is prescribed analytically; its Lighthill body-force source f = -(U.grad)U (the
// flow's own self-advection) drives the acoustic engine (acoustic2d.h), whose
// low-dispersion B-spline mimetic D1 propagates the faint sound to the far field.
//
// This program verifies the radiated field has the rotating-quadrupole signature:
//   (1) frequency 2*Omega at a far observer,
//   (2) azimuthal order m = 2 (four lobes) dominating the far-field circle,
//   (3) cylindrical 1/sqrt(r) decay of the acoustic amplitude.

#include "acoustic2d.h"

int main() {
    std::printf("=================================================================\n");
    std::printf(" Co-rotating vortex pair sound (Lighthill) -- spinning quadrupole\n");
    std::printf(" verify quadrupole signature: freq 2*Omega, azimuthal m=2, 1/sqrt(r)\n");
    std::printf("=================================================================\n");
    const int N = 160; const double L = 24.0, c = 1.0;
    Acoustic2D ac(N, L, c, 2, A_BMG);
    const double h = ac.h, x0 = L/2, y0 = L/2;
    const double r0 = 1.0, rc = 0.4, Gam = 2*ACPI;          // Lamb-Oseen pair
    const double Om = Gam/(4*ACPI*r0*r0);                   // co-rotation rate
    const double Mach = Om*r0/c, lam = ACPI*c/Om, ppw = lam/h;
    std::printf("\n N=%d  Omega=%.3f  Mach=%.2f  lambda_ac=%.2f  PPW=%.1f\n", N, Om, Mach, lam, ppw);

    Field p(N),u(N),v(N),S(N),fx(N),fy(N),dp(N),du(N),dv(N),t1(N),t2(N);
    Field Ux(N),Uy(N),Axx(N),Axy(N),Ayx(N),Ayy(N);
    Field p1(N),u1(N),v1(N),kp(N),ku(N),kv(N),Pre(N),Pim(N);
    auto win = [&](double r){ const double a=4*r0,b=6*r0; if(r<a)return 1.0; if(r>b)return 0.0; double t=(r-a)/(b-a); return 1.0-t*t*(3-2*t); };

    // Lighthill source f = -(U.grad)U from the rotating pair at time t (frozen per step).
    auto source = [&](double t){
        for (int j=0;j<N;++j) for (int i=0;i<N;++i){ double x=i*h-x0,y=j*h-y0,Uxx=0,Uyy=0;
            for (int s=-1;s<=1;s+=2){ double cx=s*r0*std::cos(Om*t),cy=s*r0*std::sin(Om*t);
                double dx=x-cx,dy=y-cy,s2=dx*dx+dy*dy+1e-12,sd=std::sqrt(s2);
                double ut=(Gam/(2*ACPI*sd))*(1-std::exp(-s2/(rc*rc)));
                Uxx += ut*(-dy/sd); Uyy += ut*(dx/sd); }
            Ux(i,j)=Uxx; Uy(i,j)=Uyy; }
        ac.dx(Ux,Axx); ac.dy(Ux,Axy); ac.dx(Uy,Ayx); ac.dy(Uy,Ayy);
        for (int j=0;j<N;++j) for (int i=0;i<N;++i){ double x=i*h-x0,y=j*h-y0,w=win(std::sqrt(x*x+y*y));
            fx(i,j) = -w*(Ux(i,j)*Axx(i,j)+Uy(i,j)*Axy(i,j));
            fy(i,j) = -w*(Ux(i,j)*Ayx(i,j)+Uy(i,j)*Ayy(i,j)); }
    };
    const double dt = 0.3*h/c, Trot = 2*ACPI/Om, Tp = ACPI/Om;   // Tp = acoustic period (2*Omega)
    const double wrad = 2*Om;                                     // radiated angular frequency
    const int nst = (int)std::lround((0.7*L/c + 3*Trot)/dt);
    const int nacc = 2*(int)std::lround(Tp/dt);                  // extract over last 2 acoustic periods
    // far observer for the frequency check
    const double robs = 1.0*lam; const int io=(int)std::lround((x0+robs)/h), jo=(int)std::lround(y0/h);
    std::vector<double> ptime;
    auto step = [&](double tn){
        source(tn);                          // frozen-per-step source (smooth in time)
        ac.rhs(p,u,v,S,fx,fy,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ kp.a[k]=dp.a[k];ku.a[k]=du.a[k];kv.a[k]=dv.a[k];
            p1.a[k]=p.a[k]+0.5*dt*dp.a[k];u1.a[k]=u.a[k]+0.5*dt*du.a[k];v1.a[k]=v.a[k]+0.5*dt*dv.a[k]; }
        ac.rhs(p1,u1,v1,S,fx,fy,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ kp.a[k]+=2*dp.a[k];ku.a[k]+=2*du.a[k];kv.a[k]+=2*dv.a[k];
            p1.a[k]=p.a[k]+0.5*dt*dp.a[k];u1.a[k]=u.a[k]+0.5*dt*du.a[k];v1.a[k]=v.a[k]+0.5*dt*dv.a[k]; }
        ac.rhs(p1,u1,v1,S,fx,fy,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ kp.a[k]+=2*dp.a[k];ku.a[k]+=2*du.a[k];kv.a[k]+=2*dv.a[k];
            p1.a[k]=p.a[k]+dt*dp.a[k];u1.a[k]=u.a[k]+dt*du.a[k];v1.a[k]=v.a[k]+dt*dv.a[k]; }
        ac.rhs(p1,u1,v1,S,fx,fy,dp,du,dv,t1,t2);
        for(size_t k=0;k<p.a.size();++k){ p.a[k]+=dt/6*(kp.a[k]+dp.a[k]); u.a[k]+=dt/6*(ku.a[k]+du.a[k]); v.a[k]+=dt/6*(kv.a[k]+dv.a[k]); }
    };
    for (int n=0;n<nst;++n){ step(n*dt); ptime.push_back(p(io,jo));
        if (n>=nst-nacc){ const double tt=(n+1)*dt, cw=std::cos(wrad*tt), sw=std::sin(wrad*tt);
            for(size_t k=0;k<p.a.size();++k){ Pre.a[k]+=p.a[k]*cw; Pim.a[k]+=p.a[k]*sw; } } }
    // |P| = amplitude of the 2*Omega (radiating) component -- filters out the huge
    // STEADY hydrodynamic near-field, leaving the faint radiated sound.
    Field Pmag(N);
    for (size_t k=0;k<Pre.a.size();++k){ Pre.a[k]*=2.0/nacc; Pim.a[k]*=2.0/nacc;
        Pmag.a[k]=std::sqrt(Pre.a[k]*Pre.a[k]+Pim.a[k]*Pim.a[k]); }

    // (1) frequency at the far observer: count zero-crossings over the last 2 rotations.
    int n0 = (int)std::lround((nst*dt - 2*Trot)/dt), zc=0;
    for (int n=n0+1;n<(int)ptime.size();++n) if (ptime[n-1]*ptime[n]<0) ++zc;
    const double Tmeas = 2.0*(2*Trot)/zc;                 // period = 2*span/zerocrossings
    std::printf("\n (1) far-field period: measured %.3f  vs  2*Omega -> %.3f  (ratio %.3f)\n",
                Tmeas, Tp, Tmeas/Tp);

    // (2) azimuthal structure of the radiated (2*Omega) field on a far circle: the
    // in-phase component Pre ~ A(r) cos(2 theta) so m=2 must dominate (quadrupole).
    const int na=360; std::vector<double> ring(na);
    const double rr=1.0*lam;
    for (int a=0;a<na;++a){ double th=2*ACPI*a/na; int i=(int)std::lround((x0+rr*std::cos(th))/h),j=(int)std::lround((y0+rr*std::sin(th))/h);
        ring[a]= (i>=0&&i<N&&j>=0&&j<N)? Pre(i,j):0.0; }
    auto modeamp=[&](int m){ double cr=0,ci=0; for(int a=0;a<na;++a){ double th=2*ACPI*a/na; cr+=ring[a]*std::cos(m*th); ci+=ring[a]*std::sin(m*th);} return std::sqrt(cr*cr+ci*ci)*2.0/na; };
    std::printf(" (2) radiated-field azimuthal amplitudes: m0=%.2e m1=%.2e m2=%.2e m3=%.2e m4=%.2e\n",
                modeamp(0),modeamp(1),modeamp(2),modeamp(3),modeamp(4));

    // (3) cylindrical spreading of the radiated (2*Omega) acoustic amplitude |P|.
    std::printf(" (3) r/lambda   |P|*sqrt(r)  (const => 1/sqrt(r) radiation)\n");
    for (double q : {0.7,1.0,1.3}) { double r=q*lam,s=0;int cnt=0;
        for(int a=0;a<na;++a){ double th=2*ACPI*a/na; int i=(int)std::lround((x0+r*std::cos(th))/h),j=(int)std::lround((y0+r*std::sin(th))/h);
            if(i>=0&&i<N&&j>=0&&j<N){ s+=Pmag(i,j);++cnt; } }
        std::printf("       %.2f      %.3e\n", q, (s/cnt)*std::sqrt(r)); }
    double emax=0; for(double z:p.a) emax=std::fmax(emax,std::fabs(z));
    std::printf("\n max|p|=%.3e (bounded => stable)\n\nDone.\n", emax);
    return 0;
}
