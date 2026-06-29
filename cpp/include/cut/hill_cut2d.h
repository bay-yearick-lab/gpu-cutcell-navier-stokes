// hill_cut2d.h
//
// Two-dimensional sharp-cut B-spline forms on the periodic-hill curved wall, the
// geometry for the S6 capstone (sec:s6-hill).  Port of code/hill_cut_quadrature.py.
//
// The fluid domain is { y >= h(x) } of the box [0,Lx] x [0,Ly], rescaled to the
// unit square [0,1]^2 by x' = x/Lx, y' = y/Ly, so the wall becomes the height
// function  hbar(x') = h(Lx x') / Ly.  Working on the unit square lets the cut
// forms reuse the verified cut2d aggregation / congruence / gates unchanged; only
// the QUADRATURE changes from polygon clipping to a height-function rule:
//
//   * volume:  for each cell, x-Gauss over sub-panels split at wall kinks and at
//     the crossings hbar(x) = cell_y0 / cell_y1, then for each x-node a y-Gauss
//     from max(cell_y0, hbar(x)) to the cell top (skipping fully-solid columns);
//   * wall:    arc-length quadrature along (x, hbar(x)) with the contribution
//     n ds = (-hbar', 1) dx, plus the three box edges as straight boundaries.
//
// The order is scale-invariant, so the p+1 best-approximation gate on the unit
// square certifies the curved-cut accuracy claimed for the hill.

#ifndef HILL_CUT2D_H
#define HILL_CUT2D_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "bspline.h"
#include "cut2d.h"
#include "hill_geom.h"

// Rescaled wall height and slope on the unit square.
struct HillBar {
    HillProfile hill; double Lx = 9.0, Ly = 3.036;
    bool physical = false;   // false: unit-square forms (scale-invariant accuracy gates);
                             // true: PHYSICAL Lx:Ly metric (gradients /Lx,/Ly, volume Jac Lx*Ly,
                             // physical wall arc-length + normal) for a DNS matching the reference stations.
    double hbar(double xp) const { return hill.h(Lx * xp) / Ly; }
    double hbarp(double xp) const { return hill.hp(Lx * xp) * Lx / Ly; }   // d/dx'
    std::vector<double> kinks_unit() const {                                // wall kinks in x'
        std::vector<double> k = hill.kinks(); for (double& v : k) v /= Lx; return k;
    }
};

// Bracketed bisection for x in (xa,xb) with hbar(x) = yk (the wall crossing a
// cell edge), scanning for sign changes of hbar-yk and refining each bracket.
inline void hill_crossings(const HillBar& H, double xa, double xb, double yk,
                           std::vector<double>& out, int nscan = 24) {
    double xprev = xa, gprev = H.hbar(xa) - yk;
    for (int i = 1; i <= nscan; ++i) {
        const double xc = xa + (xb - xa) * i / (double)nscan;
        const double gc = H.hbar(xc) - yk;
        if (gprev == 0.0 || gprev * gc < 0.0) {
            double lo = xprev, hi = xc, glo = gprev;
            for (int it = 0; it < 60; ++it) {
                const double m = 0.5*(lo+hi), gm = H.hbar(m) - yk;
                if (glo * gm <= 0.0) hi = m; else { lo = m; glo = gm; }
            }
            const double r = 0.5*(lo+hi);
            if (r > xa + 1e-12 && r < xb - 1e-12) out.push_back(r);
        }
        xprev = xc; gprev = gc;
    }
}

// Height-function fluid quadrature points (x', y', w) on the unit square.
inline FluidQuad2D hill_fluid_quadrature(int ne, int p, const HillBar& H, int q = -1) {
    if (q < 0) q = std::max(12, 2*p + 8);
    const double h = 1.0 / ne;
    std::vector<double> gx, gw; gauss_legendre(q, gx, gw);
    std::vector<double> ku = H.kinks_unit();
    FluidQuad2D fq;
    for (int iy = 0; iy < ne; ++iy) for (int ix = 0; ix < ne; ++ix) {
        const double x0 = ix*h, x1 = (ix+1)*h, y0 = iy*h, y1 = (iy+1)*h;
        // x sub-panel breaks: cell edges + wall kinks + crossings of y0 and y1.
        std::vector<double> br{x0, x1};
        for (double k : ku) if (k > x0 + 1e-12 && k < x1 - 1e-12) br.push_back(k);
        hill_crossings(H, x0, x1, y0, br);
        hill_crossings(H, x0, x1, y1, br);
        std::sort(br.begin(), br.end());
        br.erase(std::unique(br.begin(), br.end(), [](double a,double b){return std::fabs(a-b)<1e-12;}), br.end());
        for (size_t s = 0; s + 1 < br.size(); ++s) {
            const double xa = br[s], xb = br[s+1]; if (xb <= xa) continue;
            const double hx = 0.5*(xb-xa), mx = 0.5*(xb+xa);
            for (int gi = 0; gi < q; ++gi) {
                const double xp = mx + hx*gx[gi], wx = gw[gi]*hx;
                const double lower = std::max(y0, H.hbar(xp));
                if (lower >= y1 - 1e-14) continue;            // column fully solid
                const double hy = 0.5*(y1-lower), my = 0.5*(y1+lower);
                for (int gj = 0; gj < q; ++gj) {
                    fq.x.push_back(xp); fq.y.push_back(my + hy*gx[gj]);
                    fq.w.push_back(wx * gw[gj]*hy);
                }
            }
        }
    }
    return fq;
}

// Assemble the cut forms (H, Qx, Qy, K volume; Sx, Sy wall + box edges; Hbg) on
// the hill domain.  Returns a CutForms2D usable by cut2d_active_sets / extension.
inline CutForms2D assemble_hill_cut_2d(int ne, int p, const HillBar& H, int q = -1) {
    if (q < 0) q = std::max(12, 2*p + 8);
    const int nb = ne + p, n = nb*nb;
    std::vector<double> knots = open_uniform_knots(nb, p, 1.0);
    std::vector<double> grev = greville_points(knots, p, nb);
    CutForms2D cf;
    cf.nx = cf.ny = nb; cf.n = n; cf.degree = p; cf.n_elements = ne; cf.knots = knots;
    cf.H=Mat(n,n); cf.Qx=Mat(n,n); cf.Qy=Mat(n,n); cf.K=Mat(n,n); cf.Sx=Mat(n,n); cf.Sy=Mat(n,n); cf.Hbg=Mat(n,n);
    cf.gx.resize(n); cf.gy.resize(n);
    for (int iy=0; iy<nb; ++iy) for (int ix=0; ix<nb; ++ix){ cf.gx[iy*nb+ix]=grev[ix]; cf.gy[iy*nb+ix]=grev[iy]; }

    // Physical Lx:Ly metric (opt-in via H.physical): map the parametric unit-square
    // forms to physical x=Lx x', y=Ly y' -- gradients carry 1/Lx,1/Ly (chain rule),
    // volume integrals carry the Jacobian Lx*Ly.  H.physical=false => Lx=Ly=1 (the
    // scale-invariant unit-square forms; existing accuracy gates unchanged).
    const double Lx = H.physical ? H.Lx : 1.0, Ly = H.physical ? H.Ly : 1.0;
    const double isx = 1.0/Lx, isy = 1.0/Ly, jacv = Lx*Ly;

    auto accum_vol = [&](double xq, double yq, double wq){
        std::vector<double> vx,dvx,vy,dvy;
        const int lox = local_1d(knots,p,nb,xq,vx,dvx), loy = local_1d(knots,p,nb,yq,vy,dvy);
        const double wj = jacv*wq;
        for (int ja=0;ja<=p;++ja) for (int ia=0;ia<=p;++ia){
            const int ra=(loy+ja)*nb+(lox+ia);
            const double pa=vy[ja]*vx[ia], dxa=isx*vy[ja]*dvx[ia], dya=isy*dvy[ja]*vx[ia];
            for (int jb=0;jb<=p;++jb) for (int ib=0;ib<=p;++ib){
                const int rb=(loy+jb)*nb+(lox+ib);
                const double pb=vy[jb]*vx[ib], dxb=isx*vy[jb]*dvx[ib], dyb=isy*dvy[jb]*vx[ib];
                cf.H(ra,rb)+=wj*pa*pb; cf.Qx(ra,rb)+=wj*pa*dxb; cf.Qy(ra,rb)+=wj*pa*dyb;
                cf.K(ra,rb)+=wj*(dxa*dxb+dya*dyb);
            }
        }
    };
    auto accum_bdy = [&](double xq, double yq, double nxw, double nyw){   // nxw,nyw already include w*ds
        std::vector<double> vx,dvx,vy,dvy;
        const int lox = local_1d(knots,p,nb,xq,vx,dvx), loy = local_1d(knots,p,nb,yq,vy,dvy);
        for (int ja=0;ja<=p;++ja) for (int ia=0;ia<=p;++ia){
            const int ra=(loy+ja)*nb+(lox+ia); const double pa=vy[ja]*vx[ia];
            for (int jb=0;jb<=p;++jb) for (int ib=0;ib<=p;++ib){
                const int rb=(loy+jb)*nb+(lox+ib); const double m=pa*(vy[jb]*vx[ib]);
                cf.Sx(ra,rb)+=nxw*m; cf.Sy(ra,rb)+=nyw*m;
            }
        }
    };

    // Volume.
    FluidQuad2D fq = hill_fluid_quadrature(ne, p, H, q);
    for (size_t i=0;i<fq.w.size();++i) accum_vol(fq.x[i], fq.y[i], fq.w[i]);

    // Wall boundary: along (x, hbar(x)), n ds = (-hbar', 1) dx, split at kinks.
    std::vector<double> gx,gw; gauss_legendre(q,gx,gw);
    {
        std::vector<double> br{0.0,1.0}; for (double k : H.kinks_unit()) if (k>1e-12&&k<1-1e-12) br.push_back(k);
        // also split where the wall enters/exits the box top (hbar=1), rare; and at cell edges for accuracy
        for (int e=1;e<ne;++e) br.push_back(e*(1.0/ne));
        std::sort(br.begin(),br.end()); br.erase(std::unique(br.begin(),br.end(),[](double a,double b){return std::fabs(a-b)<1e-12;}),br.end());
        for (size_t s=0;s+1<br.size();++s){ const double xa=br[s],xb=br[s+1]; if(xb<=xa)continue;
            const double hh=0.5*(xb-xa),mm=0.5*(xb+xa);
            for(int gi=0;gi<q;++gi){ const double xp=mm+hh*gx[gi], w=gw[gi]*hh; const double yp=H.hbar(xp);
                if (yp<0.0-1e-12||yp>1.0+1e-12) continue;
                accum_bdy(xp,yp, Ly*H.hbarp(xp)*w, -Lx*w); }     // physical outward n ds=(Ly h',-Lx)dx'
        }
    }
    // Box edges integrated over the EXACT fluid extent (split at the wall height,
    // not by skipping nodes).  Left x=0 n=(-1,0); right x=1 n=(1,0); top y=1 n=(0,1).
    auto edge = [&](bool vertical, double fixed, double nxv, double nyv){
        const double lo = vertical ? H.hbar(fixed) : 0.0, hi = 1.0;
        const double eds = vertical ? Ly : Lx;   // physical length element: vertical edge runs in y (Ly), top edge in x (Lx)
        std::vector<double> b2{lo,hi};
        for(int e=1;e<ne;++e){ const double ce=e*(1.0/ne); if(ce>lo+1e-12&&ce<hi-1e-12) b2.push_back(ce); }
        std::sort(b2.begin(),b2.end());
        for(size_t s=0;s+1<b2.size();++s){ const double ta=b2[s],tb=b2[s+1]; if(tb<=ta)continue;
            const double hh=0.5*(tb-ta),mm=0.5*(tb+ta);
            for(int gi=0;gi<q;++gi){ const double t=mm+hh*gx[gi], w=gw[gi]*hh;
                if(vertical) accum_bdy(fixed,t, nxv*eds*w, nyv*eds*w); else accum_bdy(t,fixed, nxv*eds*w, nyv*eds*w); } }
    };
    edge(true, 0.0, -1.0, 0.0); edge(true, 1.0, 1.0, 0.0); edge(false, 1.0, 0.0, 1.0);

    // Background (uncut) tensor mass -- scaled by the same Jacobian as cf.H so the
    // support ratio gamma=diag(H)/diag(Hbg) (the active-set classifier) is metric-invariant.
    CutForms1D m1 = assemble_cut_1d(ne, p, 0.0, 1.0);
    for (int iy=0;iy<nb;++iy) for (int ix=0;ix<nb;++ix){ const int ra=iy*nb+ix;
        for(int jy=0;jy<nb;++jy) for(int jx=0;jx<nb;++jx){ const int rb=jy*nb+jx; cf.Hbg(ra,rb)=jacv*m1.H(iy,jy)*m1.H(ix,jx); } }
    return cf;
}

// No-slip weak-penalty wall traces on the hill (full 2D): the wall mass
// Mw_ij = \int_wall phi_i phi_j ds (penalty) and the normal-derivative trace
// Bn_ij = \int_wall (n.grad phi_i) phi_j ds, on the curved bottom wall y=hbar(x)
// (outward n ds = (hbar',-1) dx) AND the flat top y=1 (n=(0,1), ds=dx).  These
// assemble the symmetric weak-penalty stiffness K - Bn - Bn^T + (beta/h) Mw.
struct WallTraces2D { Mat Mw, Bn; };
inline WallTraces2D assemble_hill_wall_traces(const CutForms2D& cf, const HillBar& H, int ne, int q = -1) {
    const int p = cf.degree, n = cf.n;
    if (q < 0) q = std::max(12, 2*p + 8);
    std::vector<double> gx, gw; gauss_legendre(q, gx, gw);
    WallTraces2D wt; wt.Mw = Mat(n,n); wt.Bn = Mat(n,n);
    // Physical metric (opt-in): wall arc-length ds=sqrt(Lx^2+(Ly h')^2)dx', and the
    // normal-derivative trace uses the physical gradient (dpx/Lx,dpy/Ly) with the
    // physical outward normal.  H.physical=false => Lx=Ly=1 (unchanged).
    const double Lx = H.physical ? H.Lx : 1.0, Ly = H.physical ? H.Ly : 1.0;
    auto accum = [&](double xq, double yq, double wds, double nx, double ny, double wraw){
        // wds = ds weight (for Mw); (nx,ny)*wraw = n ds weight (for Bn, un-normalized)
        std::vector<int> r1,r2; std::vector<double> phi, dpx, dpy;
        cut2d_point_basis(cf, xq, yq, r1, phi);
        cut2d_point_grad (cf, xq, yq, r2, dpx, dpy);
        for (size_t a=0;a<r1.size();++a) for (size_t b=0;b<r1.size();++b){
            wt.Mw(r1[a],r1[b]) += wds * phi[a]*phi[b];
            wt.Bn(r1[a],r1[b]) += wraw * (nx*dpx[a] + ny*dpy[a]) * phi[b];
        }
    };
    // breaks for the curved wall: kinks + cell edges.
    std::vector<double> br{0.0,1.0}; for (double k : H.kinks_unit()) if (k>1e-12&&k<1-1e-12) br.push_back(k);
    for (int e=1;e<ne;++e) br.push_back(e*(1.0/ne));
    std::sort(br.begin(),br.end()); br.erase(std::unique(br.begin(),br.end(),[](double a,double b){return std::fabs(a-b)<1e-12;}),br.end());
    for (size_t s=0;s+1<br.size();++s){ const double xa=br[s],xb=br[s+1]; if(xb<=xa)continue;
        const double hh=0.5*(xb-xa),mm=0.5*(xb+xa);
        for(int gi=0;gi<q;++gi){ const double xp=mm+hh*gx[gi], w=gw[gi]*hh, yp=H.hbar(xp), hp=H.hbarp(xp);
            const double Lp=std::sqrt(Lx*Lx+(Ly*hp)*(Ly*hp));   // physical wall arc-length |T|
            // curved bottom: ds=Lp dx' ; physical n.grad ds folds 1/Lx,1/Ly into (nx,ny):
            // d phi/d n_phys ds = ((Ly/Lx) h' dpx - (Lx/Ly) dpy) dx'
            accum(xp,yp, Lp*w, (Ly/Lx)*hp, -(Lx/Ly), w);
        }
    }
    // flat top y=1: n=(0,1), ds=dx, split at cell edges.
    std::vector<double> bt{0.0,1.0}; for(int e=1;e<ne;++e) bt.push_back(e*(1.0/ne)); std::sort(bt.begin(),bt.end());
    for (size_t s=0;s+1<bt.size();++s){ const double xa=bt[s],xb=bt[s+1]; const double hh=0.5*(xb-xa),mm=0.5*(xb+xa);
        for(int gi=0;gi<q;++gi){ const double xp=mm+hh*gx[gi], w=gw[gi]*hh; accum(xp,1.0, Lx*w, 0.0, Lx/Ly, w); }   // top y=Ly: ds=Lx dx', d phi/dn_phys ds=(Lx/Ly)dpy dx'
    }
    return wt;
}

#endif // HILL_CUT2D_H
