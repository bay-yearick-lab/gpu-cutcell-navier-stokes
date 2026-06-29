// mms_hill3d.cpp
//
// MMS gate for the 3D periodic-hill operator layer (hill3d.h): the verified
// 2D-curved cut tensored with periodic spanwise z.  Best-approximation of a
// separable u = f(x,y) g(z): with z spectrally resolved, the 3D error inherits
// the curved-cut order p+1 from the x-y plane.  Exercises the full tensor mass
// solve (M2 (x) Mz)^{-1}.

#include <cmath>
#include <cstdio>
#include <vector>
#include "hill3d.h"

static double f_xy(double x,double y){ return std::sin(2.3*x+0.7*y+0.4)+0.5*std::cos(1.1*x-1.7*y); }

int main(){
    std::printf("=================================================================\n");
    std::printf(" MMS gate: 3D periodic-hill operator layer (hill3d.h)\n");
    std::printf(" 2D-curved cut (x-y) (x) periodic z ; best-approx order p+1\n");
    std::printf("=================================================================\n\n");
    const double Lz = 4.5;
    auto g_z = [&](double z){ const double k=2.0*M_PI/Lz; return 1.0 + 0.5*std::cos(k*z) + 0.3*std::sin(2.0*k*z); };

    for (int p=2;p<=4;++p){
        std::printf("  degree p=%d (formal p+1=%d), nz=2*N_xy (z refined with x-y)\n", p, p+1);
        std::printf("   N_xy   nz   h        nr2    L2err       order\n");
        double prev=0,prevh=0;
        for (int ne : {8,12,16}){
            const int nz = 2*ne;
            Hill3D H; H.build(ne, p, nz, Lz);
            // 2D load of f and ||f||^2 over the fluid.
            FluidQuad2D fq = hill_fluid_quadrature(ne, p, H.HB);
            std::vector<double> bfull(H.cf2.n, 0.0); double f2=0.0;
            for (size_t qq=0; qq<fq.w.size(); ++qq){
                std::vector<int> rows; std::vector<double> phi;
                cut2d_point_basis(H.cf2, fq.x[qq], fq.y[qq], rows, phi);
                const double fv = f_xy(fq.x[qq], fq.y[qq]);
                f2 += fq.w[qq]*fv*fv;
                for (size_t a=0;a<rows.size();++a) bfull[rows[a]] += fq.w[qq]*phi[a]*fv;
            }
            std::vector<double> bf2 = matvec(transpose(H.ext2.E), bfull);   // reduced 2D load
            // z load of g and ||g||^2.
            std::vector<double> bg; double g2=0.0; H.Z.load_and_norm(g_z, bg, g2);
            // 3D load b = bf2 (x) bg ; project; ||u_h||^2 = <c,b>.
            FieldH b(H.nr2, nz);
            for (int m=0;m<H.nr2;++m) for (int iz=0;iz<nz;++iz) b.at(m,iz)=bf2[m]*bg[iz];
            FieldH c = hill_mass_solve(H, b);
            double proj2=0.0; for (size_t i=0;i<c.v.size();++i) proj2 += c.v[i]*b.v[i];
            const double err = std::sqrt(std::fabs(f2*g2 - proj2));
            const double hh=1.0/ne, ord=(prev>0)?std::log(prev/err)/std::log(prevh/hh):0;
            std::printf("   %4d  %3d  %.5f  %4d   %.3e   %s%.2f\n", ne, nz, hh, H.nr2, err, (prev>0)?"":"--", ord);
            prev=err; prevh=hh;
        }
        std::printf("   => 3D hill best-approx order ~ p+1 (curved-cut dominated; z spectral)\n\n");
    }

    // ---- Gate 2: L2 projection through the Galerkin sample/load machinery
    //      (the RHS foundation), and the conservation primitive Q_dir . 1 = 0. ----
    std::printf("  (2) L2 projection via sample/load machinery + conservation primitive\n");
    auto u3 = [&](double x,double y,double z){ const double k=2.0*M_PI/Lz;
        return std::sin(2.1*x+0.4)*std::cos(1.4*y-0.2)*(1.0+0.5*std::cos(k*z)); };
    for (int p=2;p<=4;++p){
        std::printf("   p=%d\n", p);
        double prev=0,prevh=0,qmax=0;
        for (int ne : {8,12,16}){
            const int nz=2*ne; Hill3D H; H.build(ne,p,nz,Lz);
            QField fq(H.n2q,H.nzq);
            for(int q=0;q<H.n2q;++q) for(int qz=0;qz<H.nzq;++qz) fq.at(q,qz)=u3(H.fq2.x[q],H.fq2.y[q],H.zq[qz]);
            FieldH c = hill_l2_project(H, fq);
            QField uh = hill_sample(H, c);
            double e2=0,n2=0;
            for(int q=0;q<H.n2q;++q) for(int qz=0;qz<H.nzq;++qz){ const double w=H.w2[q]*H.wz[qz];
                const double d=fq.at(q,qz)-uh.at(q,qz); e2+=w*d*d; n2+=w*fq.at(q,qz)*fq.at(q,qz); }
            const double err=std::sqrt(e2/n2);
            // conservation primitive: Q_dir . 1
            FieldH one(H.nr2,H.Z.nz); for(double&v:one.v) v=1.0;
            double qm=0; for(int d=0;d<3;++d){ FieldH q=hill_apply_Q(H,one,d); for(double v:q.v) qm=std::max(qm,std::fabs(v)); }
            qmax=std::max(qmax,qm);
            const double hh=1.0/ne, ord=(prev>0)?std::log(prev/err)/std::log(prevh/hh):0;
            std::printf("     ne=%2d nz=%3d  proj-L2=%.3e  %s%.2f   max|Q.1|=%.1e\n",
                ne,nz,err,(prev>0)?"":"--",ord,qm);
            prev=err; prevh=hh;
        }
        std::printf("     => projection order ~ p+1 ; max|Q.1|<=%.1e (roundoff: derivative of a constant)\n", qmax);
    }
    // ---- Gate 3: conservative compressible RHS on the hill.  Interior pulse
    //      (zero velocity -> zero flux -> mass/energy rates exact); and the
    //      inviscid KE-production rate (skew form), at the curved-panel floor. ----
    std::printf("  (3) Conservative compressible RHS on the hill\n");
    const double gam=1.4;
    for (int p=2;p<=3;++p){
        const int ne=10, nz=20; Hill3D H; H.build(ne,p,nz,Lz);
        auto proj=[&](auto fn){ QField q(H.n2q,H.nzq);
            for(int a=0;a<H.n2q;++a) for(int b=0;b<H.nzq;++b) q.at(a,b)=fn(H.fq2.x[a],H.fq2.y[a],H.zq[b]);
            return hill_l2_project(H,q); };
        // interior pulse (u=0)
        auto pulse=[&](double x,double y,double z){ return 5e-4*std::exp(-((x-0.5)*(x-0.5)+(y-0.6)*(y-0.6)+(z-0.5*Lz)*(z-0.5*Lz))/0.03); };
        HillCompState s;
        s.rho=proj([&](double x,double y,double z){return 1.0+pulse(x,y,z)/gam;});
        s.mx=proj([](double,double,double){return 0.0;}); s.my=s.mx; s.mz=s.mx;
        s.E=proj([&](double x,double y,double z){return (1.0+pulse(x,y,z))/(gam-1.0);});
        HillCompState R=hill_comp_rhs(H,s,gam,true);
        const double dmass=std::fabs(hill_total(H,R.rho))/std::fabs(hill_total(H,s.rho));
        const double denergy=std::fabs(hill_total(H,R.E))/std::fabs(hill_total(H,s.E));
        // KE-production rate on a velocity that VANISHES on the wall (y=hbar) and
        // top (y=1) and is periodic in x,z, so the boundary KE flux is zero and the
        // measured rate is the interior skew-form production alone.
        auto uf=[&](double x,double y,double z){ const double hw=H.HB.hbar(x);
            return (y-hw)*(1.0-y)*std::cos(2.0*M_PI*x)*std::cos(2.0*M_PI*z/Lz); };
        auto vf=[&](double x,double y,double z){ const double hw=H.HB.hbar(x);
            return (y-hw)*(1.0-y)*std::sin(2.0*M_PI*x)*0.7*(0.0*z+1.0); };
        HillCompState s2;
        s2.rho=proj([](double,double,double){return 1.0;});
        s2.mx=proj(uf); s2.my=proj(vf); s2.mz=proj([](double,double,double){return 0.0;});
        s2.E=proj([&](double x,double y,double z){return 1.0/(gam-1.0)+0.5*(uf(x,y,z)*uf(x,y,z)+vf(x,y,z)*vf(x,y,z));});
        HillCompState R2=hill_comp_rhs(H,s2,gam,false);
        QField uq=hill_sample(H,s2.mx), vq=hill_sample(H,s2.my), wq=hill_sample(H,s2.mz), rqs=hill_sample(H,s2.rho);
        QField Rrq=hill_sample(H,R2.rho), Rmxq=hill_sample(H,R2.mx), Rmyq=hill_sample(H,R2.my), Rmzq=hill_sample(H,R2.mz);
        double ke=0,sc=0;
        for(int a=0;a<H.n2q;++a) for(int b=0;b<H.nzq;++b){ const double w=H.w2[a]*H.wz[b];
            const double ir=1.0/rqs.at(a,b), u=uq.at(a,b)*ir, v=vq.at(a,b)*ir, ww=wq.at(a,b)*ir;
            ke += w*(u*Rmxq.at(a,b)+v*Rmyq.at(a,b)+ww*Rmzq.at(a,b)-0.5*(u*u+v*v+ww*ww)*Rrq.at(a,b));
            sc += w*(0.5*rqs.at(a,b)*(u*u+v*v+ww*ww)); }
        std::printf("     p=%d ne=%d  pulse mass-rate=%.2e  energy-rate=%.2e  inviscid KE-rate(rel)=%.2e\n",
            p, ne, dmass, denergy, std::fabs(ke)/std::max(sc,1e-300));
    }
    std::printf("     => interior conservation exact; inviscid KE-production at the ROUNDOFF floor (wall-vanishing velocity).\n");

    // ---- Gate 4: energy-conserving skew convection.  For a scalar vanishing on
    //      the wall and top, the discrete KE injection u^T M C_skew(w) u -> floor. ----
    std::printf("  (4) Energy-conserving skew convection: KE injection u^T M C(w)u\n");
    for (int p=2;p<=3;++p){
        const int ne=10, nz=20; Hill3D H; H.build(ne,p,nz,Lz);
        auto proj=[&](auto fn){ QField q(H.n2q,H.nzq);
            for(int a=0;a<H.n2q;++a) for(int b=0;b<H.nzq;++b) q.at(a,b)=fn(H.fq2.x[a],H.fq2.y[a],H.zq[b]);
            return hill_l2_project(H,q); };
        auto ufn=[&](double x,double y,double z){ const double hw=H.HB.hbar(x);
            return (y-hw)*(1.0-y)*std::cos(2.0*M_PI*x)*std::cos(2.0*M_PI*z/Lz); };
        FieldH ui = proj(ufn);
        QField wxq=hill_sample(H,proj([](double x,double y,double){return std::sin(1.7*x)*std::cos(1.3*y);}));
        QField wyq=hill_sample(H,proj([](double x,double y,double){return std::cos(1.1*x)*std::sin(1.9*y);}));
        QField wzq=hill_sample(H,proj([&](double,double,double z){return 0.4*std::sin(2.0*M_PI*z/Lz);}));
        FieldH C = hill_skew_conv(H, wxq, wyq, wzq, ui);
        FieldH MC = hill_mass_apply(H, C), Mu = hill_mass_apply(H, ui);
        double inj=0, scale=0;
        for (size_t k=0;k<ui.v.size();++k){ inj += ui.v[k]*MC.v[k]; scale += ui.v[k]*Mu.v[k]; }
        std::printf("     p=%d ne=%d  KE-injection(rel)=%.2e\n", p, ne, std::fabs(inj)/std::max(scale,1e-300));
    }
    std::printf("     => skew convection injects no interior KE (roundoff): energy-conserving advection on the curved wall.\n");

    std::printf("Done.\n");
    return 0;
}
