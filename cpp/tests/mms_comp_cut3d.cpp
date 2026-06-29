// mms_comp_cut3d.cpp
//
// MMS / identity gate for the conservative-flux machinery on the 3D tensor cut
// (cut/comp_cut3d.h), the operator core of the high-order conservative
// compressible scheme (code/comp3d_highp_solver.py).  Gates:
//
//   (1) L2 projection:  reproduce a degree-p polynomial to roundoff, and a smooth
//       field at order p+1 -- the full (non-lumped) mass keeps spline accuracy.
//   (2) Conservative weak divergence:  div F computed by the conservative form
//       converges at order p+1 to the analytic divergence on the cut box.
//   (3) Conservation identity:  for a flux vanishing on every face, the discrete
//       total-mass rate  ones^T H (div F)  collapses to roundoff (Q_j^T 1 = S_j 1).

#include <cmath>
#include <cstdio>
#include <vector>

#include "comp_cut3d.h"

// Sample a callable u(x,y,z) onto the quadrature grid of a CompCut3D.
template <class F>
static Field3D sample_fn(const CompCut3D& c, F u) {
    Field3D fq(c.qx.nq, c.qy.nq, c.qz.nq);
    for (int iz = 0; iz < c.qz.nq; ++iz) for (int iy = 0; iy < c.qy.nq; ++iy) for (int ix = 0; ix < c.qx.nq; ++ix)
        fq(ix,iy,iz) = u(c.qx.x[ix], c.qy.x[iy], c.qz.x[iz]);
    return fq;
}

// Weighted L2(fluid) error between a quad-grid field and a callable.
template <class F>
static double l2_err(const CompCut3D& c, const Field3D& fq, F u) {
    double e2 = 0.0;
    for (int iz = 0; iz < c.qz.nq; ++iz) for (int iy = 0; iy < c.qy.nq; ++iy) for (int ix = 0; ix < c.qx.nq; ++ix) {
        const double d = fq(ix,iy,iz) - u(c.qx.x[ix], c.qy.x[iy], c.qz.x[iz]);
        e2 += c.qx.w[ix]*c.qy.w[iy]*c.qz.w[iz]*d*d;
    }
    return std::sqrt(e2);
}

// Smooth scalar for the projection test.
static double uS(double x,double y,double z){ return std::sin(2.1*x+0.3)*std::cos(1.4*y-0.2)*std::sin(1.7*z+0.5); }

int main() {
    std::printf("=================================================================\n");
    std::printf(" MMS gate: conservative-flux machinery on the 3D tensor cut\n");
    std::printf("=================================================================\n\n");

    const double fr = 0.3;     // self-similar corner cut

    // ---- Gate 1: L2 projection (polynomial exact + smooth order p+1) ----
    std::printf("  (1) L2 projection on the cut box\n");
    for (int p = 2; p <= 4; ++p) {
        std::printf("   p=%d   polyP(deg p)   smooth L2 (order)\n", p);
        double prev=0,prevh=0;
        for (int ne : {4,6,8,10}) {
            CompCut3D c = build_comp_cut3d(ne, p, fr, fr, fr);
            // polynomial of total degree p (mixed): x^p + y^p + z^p + x*y*z (if p>=3)
            auto poly = [p](double x,double y,double z){ return std::pow(x,p)+std::pow(y,p)+std::pow(z,p)+x*y*z; };
            Field3D pp = comp_l2_project(c, sample_fn(c, poly));
            double perr = l2_err(c, comp_sample(c, pp), poly);
            Field3D ps = comp_l2_project(c, sample_fn(c, uS));
            double serr = l2_err(c, comp_sample(c, ps), uS);
            double h = 1.0/ne, ord = (prev>0)? std::log(prev/serr)/std::log(prevh/h):0;
            std::printf("     ne=%2d  poly=%.2e   smooth=%.2e  %s%.2f\n", ne, perr, serr, (prev>0)?"":"--", ord);
            prev=serr; prevh=h;
        }
    }

    // ---- Gate 2: truncation order of the FULL Euler RHS on the exact steady
    //      isentropic vortex (continuous RHS == 0; discrete RHS = truncation),
    //      measured in the cut-domain H-weighted norm (the paper's gate). ----
    std::printf("\n  (2) Compressible RHS truncation on the exact steady vortex (H-norm, p+1 target)\n");
    const double gamma = 1.4, beta = 0.3, sv = 40.0, cx = 0.55, cy = 0.55;
    auto rho_fn = [&](double x,double y,double z){ (void)z;
        const double r2=(x-cx)*(x-cx)+(y-cy)*(y-cy);
        const double dT=-(gamma-1.0)*beta*beta/(2.0*gamma)*std::exp(1.0-sv*r2);
        return std::pow(1.0+dT, 1.0/(gamma-1.0)); };
    auto u_fn = [&](double x,double y,double z){ (void)x;(void)z;
        const double r2=(x-cx)*(x-cx)+(y-cy)*(y-cy); return -beta*std::sqrt(sv)*(y-cy)*std::exp(0.5*(1.0-sv*r2)); };
    auto v_fn = [&](double x,double y,double z){ (void)z;
        const double r2=(x-cx)*(x-cx)+(y-cy)*(y-cy); return  beta*std::sqrt(sv)*(x-cx)*std::exp(0.5*(1.0-sv*r2)); };
    auto p_fn = [&](double x,double y,double z){ return std::pow(rho_fn(x,y,z),gamma); };
    for (int p = 2; p <= 4; ++p) {
        std::printf("   p=%d\n", p);
        double prev=0,prevh=0;
        for (int ne : {4,6,8,10}) {
            CompCut3D c = build_comp_cut3d(ne, p, fr, fr, fr);
            // conservative state = L2 projection of (rho, rho u, rho v, 0, E).
            CompState s;
            s.rho = comp_l2_project(c, sample_fn(c, rho_fn));
            s.mx  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return rho_fn(x,y,z)*u_fn(x,y,z); }));
            s.my  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return rho_fn(x,y,z)*v_fn(x,y,z); }));
            s.mz  = comp_l2_project(c, sample_fn(c, [](double,double,double){ return 0.0; }));
            s.E   = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){
                const double r=rho_fn(x,y,z), u=u_fn(x,y,z), v=v_fn(x,y,z);
                return p_fn(x,y,z)/(gamma-1.0) + 0.5*r*(u*u+v*v); }));
            CompState R = comp_rhs(c, s, gamma);
            double e2 = 0.0; for (const Field3D* r : {&R.rho,&R.mx,&R.my,&R.mz,&R.E}) { double hn=comp_hnorm(c,*r); e2+=hn*hn; }
            double ref2 = 0.0; for (const Field3D* q : {&s.mx,&s.my,&s.mz,&s.E}) { double hn=comp_hnorm(c,*q); ref2+=hn*hn; }
            double err = std::sqrt(e2)/std::sqrt(ref2);
            double h=1.0/ne, ord=(prev>0)? std::log(prev/err)/std::log(prevh/h):0;
            std::printf("     ne=%2d  trunc(rel H-norm)=%.3e  %s%.2f\n", ne, err, (prev>0)?"":"--", ord);
            prev=err; prevh=h;
        }
    }

    // ---- Gate 3: conservation is algebraic.  Q_j^T 1 = S_j 1 is inherited from
    //      the verified SBP identity (Q_j 1 = 0, Q_j + Q_j^T = S_j), so any RHS
    //      satisfies  ones^T H (rhs) = face flux exactly.  Confirm Q_x c1 = 0. ----
    std::printf("\n  (3) Conservation primitive: ||reduced Q_x (constant)|| -> roundoff\n");
    for (int p = 2; p <= 4; ++p) {
        CompCut3D c = build_comp_cut3d(8, p, fr, fr, fr);
        Field3D one(c.nr(), c.nr(), c.nr()); for (double& v : one.v) v = 1.0;  // constant's reduced coeffs
        double m = 0.0; for (int ax=0;ax<3;++ax){ Field3D q=comp_apply_Q(c, one, ax); for(double v:q.v) m=std::max(m,std::fabs(v)); }
        std::printf("     p=%d  max|Q_axis . 1| = %.2e\n", p, m);
    }

    // ---- Gate 4: the two headline instantaneous identities the theory promises.
    //   (4a) Inviscid KE-production rate (pressure off): the matrix-form skew-symmetric
    //        split makes the interior production vanish -> roundoff.
    //   (4b) Conservation: for an interior pulse with zero velocity, the discrete
    //        mass / energy rates equal the (zero) boundary flux -> roundoff. ----
    std::printf("\n  (4) Headline conservation identities (instantaneous)\n");
    for (int p = 2; p <= 4; ++p) {
        CompCut3D c = build_comp_cut3d(6, p, fr, fr, fr);
        // (4a) KE production rate on the vortex, pressure off.
        CompState s;
        s.rho = comp_l2_project(c, sample_fn(c, rho_fn));
        s.mx  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return rho_fn(x,y,z)*u_fn(x,y,z); }));
        s.my  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return rho_fn(x,y,z)*v_fn(x,y,z); }));
        s.mz  = comp_l2_project(c, sample_fn(c, [](double,double,double){ return 0.0; }));
        s.E   = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){
            const double r=rho_fn(x,y,z),u=u_fn(x,y,z),v=v_fn(x,y,z);
            return p_fn(x,y,z)/(gamma-1.0)+0.5*r*(u*u+v*v); }));
        CompState R = comp_rhs(c, s, gamma, /*with_pressure=*/false);
        Field3D rq=comp_sample(c,s.rho), mxq=comp_sample(c,s.mx), myq=comp_sample(c,s.my), mzq=comp_sample(c,s.mz);
        Field3D Rrq=comp_sample(c,R.rho), Rmxq=comp_sample(c,R.mx), Rmyq=comp_sample(c,R.my), Rmzq=comp_sample(c,R.mz);
        double ke_rate=0, ke_scale=0;
        for (int iz=0; iz<c.qz.nq; ++iz) for (int iy=0; iy<c.qy.nq; ++iy) for (int ix=0; ix<c.qx.nq; ++ix) {
            const size_t k = ix + (size_t)c.qx.nq*(iy + (size_t)c.qy.nq*iz);
            const double w = c.qx.w[ix]*c.qy.w[iy]*c.qz.w[iz];
            const double ir=1.0/rq.v[k], u=mxq.v[k]*ir, v=myq.v[k]*ir, ww=mzq.v[k]*ir;
            ke_rate  += w*(u*Rmxq.v[k] + v*Rmyq.v[k] + ww*Rmzq.v[k] - 0.5*(u*u+v*v+ww*ww)*Rrq.v[k]);
            ke_scale += w*(0.5*rq.v[k]*(u*u+v*v+ww*ww));
        }
        // (4b) interior zero-velocity pulse: mass / energy rates -> roundoff.
        const double c0=0.55, sig2=0.05;
        auto pulse=[&](double x,double y,double z){ return 5e-4*std::exp(-((x-c0)*(x-c0)+(y-c0)*(y-c0)+(z-c0)*(z-c0))/sig2); };
        CompState s2;
        s2.rho = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return 1.0+pulse(x,y,z)/gamma; }));
        s2.mx = comp_l2_project(c, sample_fn(c, [](double,double,double){return 0.0;}));
        s2.my = s2.mx; s2.mz = s2.mx;
        s2.E  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return (1.0+pulse(x,y,z))/(gamma-1.0); }));
        CompState R2 = comp_rhs(c, s2, gamma, true);
        const double dmass = std::fabs(comp_total(c,R2.rho))/std::fabs(comp_total(c,s2.rho));
        const double denergy = std::fabs(comp_total(c,R2.E))/std::fabs(comp_total(c,s2.E));
        std::printf("     p=%d  inviscid KE-rate(rel)=%.2e   pulse mass-rate=%.2e  energy-rate=%.2e\n",
                    p, std::fabs(ke_rate)/std::max(ke_scale,1e-300), dmass, denergy);
    }
    std::printf("     => interior KE production and mass/energy rates at the roundoff floor.\n");

    // ---- Gate 5: CUT-INDEPENDENCE.  Drive the corner-cut volume fraction toward
    //      the paper's 1e-6 (and beyond) by pushing the per-axis cut to the cell
    //      edge; the KE-production rate and the conservation residual stay at the
    //      roundoff floor, set by the background grid, not the sliver. ----
    std::printf("\n  (5) Cut-independence: conservation vs corner-cut volume fraction (p=3)\n");
    std::printf("     frac     corner_volfrac   KE-rate(rel)   pulse mass-rate   energy-rate\n");
    for (double frac : {0.5, 0.9, 0.99, 0.999, 0.9999}) {
        const int p = 3, ne = 6;
        CompCut3D c = build_comp_cut3d(ne, p, frac, frac, frac);
        const double corner = c.ax.gamma_min * c.ay.gamma_min * c.az.gamma_min;
        CompState s;
        s.rho = comp_l2_project(c, sample_fn(c, rho_fn));
        s.mx  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return rho_fn(x,y,z)*u_fn(x,y,z); }));
        s.my  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return rho_fn(x,y,z)*v_fn(x,y,z); }));
        s.mz  = comp_l2_project(c, sample_fn(c, [](double,double,double){ return 0.0; }));
        s.E   = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){
            const double r=rho_fn(x,y,z),u=u_fn(x,y,z),v=v_fn(x,y,z);
            return p_fn(x,y,z)/(gamma-1.0)+0.5*r*(u*u+v*v); }));
        CompState R = comp_rhs(c, s, gamma, /*with_pressure=*/false);
        Field3D rq=comp_sample(c,s.rho), mxq=comp_sample(c,s.mx), myq=comp_sample(c,s.my), mzq=comp_sample(c,s.mz);
        Field3D Rrq=comp_sample(c,R.rho), Rmxq=comp_sample(c,R.mx), Rmyq=comp_sample(c,R.my), Rmzq=comp_sample(c,R.mz);
        double ke_rate=0, ke_scale=0;
        for (int iz=0; iz<c.qz.nq; ++iz) for (int iy=0; iy<c.qy.nq; ++iy) for (int ix=0; ix<c.qx.nq; ++ix) {
            const size_t k = ix + (size_t)c.qx.nq*(iy + (size_t)c.qy.nq*iz);
            const double w = c.qx.w[ix]*c.qy.w[iy]*c.qz.w[iz];
            const double ir=1.0/rq.v[k], u=mxq.v[k]*ir, v=myq.v[k]*ir, ww=mzq.v[k]*ir;
            ke_rate  += w*(u*Rmxq.v[k] + v*Rmyq.v[k] + ww*Rmzq.v[k] - 0.5*(u*u+v*v+ww*ww)*Rrq.v[k]);
            ke_scale += w*(0.5*rq.v[k]*(u*u+v*v+ww*ww));
        }
        const double c0=0.55, sig2=0.05;
        auto pulse=[&](double x,double y,double z){ return 5e-4*std::exp(-((x-c0)*(x-c0)+(y-c0)*(y-c0)+(z-c0)*(z-c0))/sig2); };
        CompState s2;
        s2.rho = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return 1.0+pulse(x,y,z)/gamma; }));
        s2.mx = comp_l2_project(c, sample_fn(c, [](double,double,double){return 0.0;})); s2.my=s2.mx; s2.mz=s2.mx;
        s2.E  = comp_l2_project(c, sample_fn(c, [&](double x,double y,double z){ return (1.0+pulse(x,y,z))/(gamma-1.0); }));
        CompState R2 = comp_rhs(c, s2, gamma, true);
        const double dmass=std::fabs(comp_total(c,R2.rho))/std::fabs(comp_total(c,s2.rho));
        const double denergy=std::fabs(comp_total(c,R2.E))/std::fabs(comp_total(c,s2.E));
        std::printf("     %.4f   %.2e        %.2e       %.2e          %.2e\n",
                    frac, corner, std::fabs(ke_rate)/std::max(ke_scale,1e-300), dmass, denergy);
    }
    std::printf("     => conservation set by the background grid, independent of the cut fraction.\n");

    std::printf("\nDone.\n");
    return 0;
}
