// comp_cut3d.h
//
// Conservative-flux machinery on the 3D tensor-cut reduced space (cut3d.h), the
// C++ port of code/comp3d_highp_solver.py.  Everything is built from per-axis 1D
// operators and tensor quadrature, so every operation is a stack of 1D line
// applies (the 3d_to_1d reduction) and nothing dense-3D is ever formed.
//
// The conservative weak divergence is the heart of the high-order conservative
// compressible scheme: each flux component is L2-projected to spline coefficients
// f_j = H^{-1}(PHIr W F_j) and differentiated by the FULL cut weak derivative,
//   -div F = -H^{-1} sum_j Q_j f_j ,
// with Q_j + Q_j^T = S_j.  Because Q_j^T 1 = S_j 1, the discrete total changes by
// exactly the face flux: mass/momentum/energy conserve to roundoff, while the
// full (non-lumped) derivative keeps order p+1 (breaks the order-2 lumped cap).

#ifndef COMP_CUT3D_H
#define COMP_CUT3D_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "bspline.h"
#include "cut1d.h"
#include "cut3d.h"

// Per-axis cut quadrature: the sample matrix V (nq x n, V(q,i)=N_i(x_q)) on the
// fluid interval [left,1] and the Gauss weights, used for Galerkin products.
struct CutAxisQuad {
    int nq = 0, n = 0;
    Mat V;                       // nq x n
    std::vector<double> w, x;
};

inline CutAxisQuad build_axis_quad(const CutAxis1D& ax, int q = -1) {
    const int p = ax.degree, n = ax.n;
    if (q < 0) q = 2 * p + 4;
    std::vector<double> gx, gw; gauss_legendre(q, gx, gw);
    std::vector<double> brk = cut1d_panels(ax.knots, ax.left, 1.0);
    CutAxisQuad aq; aq.n = n;
    std::vector<std::vector<double>> rows;
    for (size_t s = 0; s + 1 < brk.size(); ++s) {
        const double lo = brk[s], hi = brk[s+1]; if (hi <= lo) continue;
        const double half = 0.5*(hi-lo), mid = 0.5*(hi+lo);
        for (int g = 0; g < q; ++g) {
            aq.x.push_back(mid + half*gx[g]); aq.w.push_back(gw[g]*half);
            rows.push_back(cut1d_eval_all(ax.knots, p, n, mid + half*gx[g], 0));
        }
    }
    aq.nq = (int)rows.size();
    aq.V = Mat(aq.nq, n);
    for (int qq = 0; qq < aq.nq; ++qq) for (int i = 0; i < n; ++i) aq.V(qq,i) = rows[qq][i];
    return aq;
}

// A tensor-cut compressible context: the three reduced axes plus their quadrature.
struct CompCut3D {
    CutAxis1D ax, ay, az;
    CutAxisQuad qx, qy, qz;
    int nr() const { return ax.nr; }       // (cubic box: same per axis)
    int nq() const { return qx.nq; }
};

inline CompCut3D build_comp_cut3d(int n_elements, int p, double fx, double fy, double fz) {
    CompCut3D c;
    const double h = 1.0 / n_elements;
    c.ax = build_cut_axis(n_elements, p, fx*h);
    c.ay = build_cut_axis(n_elements, p, fy*h);
    c.az = build_cut_axis(n_elements, p, fz*h);
    c.qx = build_axis_quad(c.ax); c.qy = build_axis_quad(c.ay); c.qz = build_axis_quad(c.az);
    return c;
}

// Sample a reduced-coefficient field at the tensor quadrature nodes:
//   f_q = PHIr^T c = sample(expand(c)).
inline Field3D comp_sample(const CompCut3D& c, const Field3D& cr) {
    Field3D full = tensor_expand(cr, c.ax, c.ay, c.az);
    Field3D t = apply_axis(full, 0, c.qx.V);
    t = apply_axis(t, 1, c.qy.V);
    t = apply_axis(t, 2, c.qz.V);
    return t;                                // dims (nqx,nqy,nqz)
}

// Reduced Galerkin load of a quadrature-grid field:  PHIr (W f_q)  =  E^T PHI W f.
inline Field3D comp_load_reduced(const CompCut3D& c, const Field3D& Fq) {
    Field3D wf = Fq;                          // apply separable quad weights
    for (int iz = 0; iz < c.qz.nq; ++iz) for (int iy = 0; iy < c.qy.nq; ++iy) for (int ix = 0; ix < c.qx.nq; ++ix)
        wf(ix,iy,iz) *= c.qx.w[ix]*c.qy.w[iy]*c.qz.w[iz];
    Field3D full = apply_axis(wf, 0, transpose(c.qx.V));   // quad -> basis
    full = apply_axis(full, 1, transpose(c.qy.V));
    full = apply_axis(full, 2, transpose(c.qz.V));
    return tensor_reduce(full, c.ax, c.ay, c.az);          // -> reduced
}

// L2 projection of a quadrature-grid field onto the reduced cut space.
inline Field3D comp_l2_project(const CompCut3D& c, const Field3D& Fq) {
    return reduced_mass_solve(comp_load_reduced(c, Fq), c.ax, c.ay, c.az);
}

// Apply the reduced 3D Q_axis = (reduced 1D Q on that axis) (x) (reduced 1D H on
// the other two axes) to a reduced field.
inline Field3D comp_apply_Q(const CompCut3D& c, const Field3D& f, int axis) {
    if (axis == 0) { Field3D t = apply_axis(f,0,c.ax.Qr); t = apply_axis(t,1,c.ay.Hr); return apply_axis(t,2,c.az.Hr); }
    if (axis == 1) { Field3D t = apply_axis(f,0,c.ax.Hr); t = apply_axis(t,1,c.ay.Qr); return apply_axis(t,2,c.az.Hr); }
    /* axis 2 */     Field3D t = apply_axis(f,0,c.ax.Hr); t = apply_axis(t,1,c.ay.Hr); return apply_axis(t,2,c.az.Qr);
}

// Conservative weak divergence of a flux field given at the quadrature nodes per
// axis:  -div F = -H^{-1} sum_j Q_j H^{-1}(PHIr W F_j).
inline Field3D comp_wdiv(const CompCut3D& c, const Field3D& Fx, const Field3D& Fy, const Field3D& Fz) {
    Field3D acc(c.nr(), c.nr(), c.nr());
    const Field3D* F[3] = { &Fx, &Fy, &Fz };
    for (int j = 0; j < 3; ++j) {
        Field3D fc = reduced_mass_solve(comp_load_reduced(c, *F[j]), c.ax, c.ay, c.az);
        Field3D Qf = comp_apply_Q(c, fc, j);
        for (size_t i = 0; i < acc.size(); ++i) acc.v[i] += Qf.v[i];
    }
    return reduced_mass_solve(acc, c.ax, c.ay, c.az);
}

// Total integral over the fluid of a reduced field:  ones^T (H_red c) = \int E c.
inline double comp_total(const CompCut3D& c, const Field3D& cr) {
    Field3D Hc = reduced_mass_apply(cr, c.ax, c.ay, c.az);
    double s = 0.0; for (double v : Hc.v) s += v; return s;
}

// Conservative state and RHS (reduced coefficients of rho, rho u, rho v, rho w, E).
struct CompState { Field3D rho, mx, my, mz, E; };

// H-weighted norm of a reduced residual:  sqrt(c . H_red c).
inline double comp_hnorm(const CompCut3D& c, const Field3D& r) {
    Field3D Hr = reduced_mass_apply(r, c.ax, c.ay, c.az);
    double s = 0.0; for (size_t i = 0; i < r.size(); ++i) s += r.v[i]*Hr.v[i];
    return std::sqrt(std::fabs(s));
}

// The conservative compressible Euler RHS (matrix-form skew-symmetric, full mass).
// Conservative weak divergence of every flux, plus the weak pressure gradient.
inline CompState comp_rhs(const CompCut3D& c, const CompState& s, double gamma = 1.4,
                          bool with_pressure = true) {
    // Primitives at the quadrature nodes (exact pointwise division).
    Field3D rq = comp_sample(c, s.rho), mxq = comp_sample(c, s.mx), myq = comp_sample(c, s.my),
            mzq = comp_sample(c, s.mz), Eq = comp_sample(c, s.E);
    const size_t Nq = rq.size();
    Field3D uq = rq, vq = rq, wq = rq, pq = rq;
    for (size_t i = 0; i < Nq; ++i) {
        const double ir = 1.0 / rq.v[i];
        uq.v[i] = mxq.v[i]*ir; vq.v[i] = myq.v[i]*ir; wq.v[i] = mzq.v[i]*ir;
        const double ke = 0.5*rq.v[i]*(uq.v[i]*uq.v[i] + vq.v[i]*vq.v[i] + wq.v[i]*wq.v[i]);
        pq.v[i] = (gamma - 1.0)*(Eq.v[i] - ke);
    }
    auto neg = [](Field3D f){ for (double& v : f.v) v = -v; return f; };
    auto mul = [&](const Field3D& a, const Field3D& b){ Field3D o = a; for (size_t i=0;i<Nq;++i) o.v[i]=a.v[i]*b.v[i]; return o; };

    CompState R;
    // continuity: -div(m)
    R.rho = neg(comp_wdiv(c, mxq, myq, mzq));
    // pressure projection coefficients (reused by all three momentum components)
    Field3D pcoef = reduced_mass_solve(comp_load_reduced(c, pq), c.ax, c.ay, c.az);
    const Field3D* mq[3] = { &mxq, &myq, &mzq };
    Field3D* Rm[3] = { &R.mx, &R.my, &R.mz };
    for (int i = 0; i < 3; ++i) {
        Field3D conv = comp_wdiv(c, mul(*mq[i], uq), mul(*mq[i], vq), mul(*mq[i], wq));
        Field3D gradp = reduced_mass_solve(comp_apply_Q(c, pcoef, i), c.ax, c.ay, c.az);
        const double pw = with_pressure ? 1.0 : 0.0;
        *Rm[i] = conv;                     // allocate with the right dims, then fill
        for (size_t k = 0; k < conv.size(); ++k) Rm[i]->v[k] = -conv.v[k] - pw*gradp.v[k];
    }
    // energy: -div((E+p) u)
    Field3D aE = Eq; for (size_t i=0;i<Nq;++i) aE.v[i] = Eq.v[i] + pq.v[i];
    R.E = neg(comp_wdiv(c, mul(aE, uq), mul(aE, vq), mul(aE, wq)));
    return R;
}

// SSP-RK3 step of the conservative compressible system.
inline void comp_ssprk3_step(const CompCut3D& c, CompState& s, double dt, double gamma = 1.4) {
    auto axpby = [](double a, const Field3D& X, double b, const Field3D& Y){
        Field3D o = X; for (size_t i = 0; i < o.size(); ++i) o.v[i] = a*X.v[i] + b*Y.v[i]; return o; };
    auto stage = [&](const CompState& u, const CompState& L, double dtt){
        CompState o;
        o.rho = axpby(1.0,u.rho, dtt,L.rho); o.mx = axpby(1.0,u.mx, dtt,L.mx);
        o.my  = axpby(1.0,u.my,  dtt,L.my);  o.mz = axpby(1.0,u.mz, dtt,L.mz);
        o.E   = axpby(1.0,u.E,   dtt,L.E);  return o; };
    auto comb = [&](double a, const CompState& A, double b, const CompState& B){
        CompState o;
        o.rho = axpby(a,A.rho,b,B.rho); o.mx = axpby(a,A.mx,b,B.mx); o.my = axpby(a,A.my,b,B.my);
        o.mz  = axpby(a,A.mz,b,B.mz);   o.E  = axpby(a,A.E, b,B.E);  return o; };

    CompState L0 = comp_rhs(c, s, gamma);
    CompState u1 = stage(s, L0, dt);
    CompState L1 = comp_rhs(c, u1, gamma);
    CompState u2 = comb(0.75, s, 0.25, stage(u1, L1, dt));
    CompState L2 = comp_rhs(c, u2, gamma);
    s = comb(1.0/3.0, s, 2.0/3.0, stage(u2, L2, dt));
}

// Greville coordinates of the reduced (interior) nodes per axis, for ICs/MMS.
inline std::vector<double> comp_axis_coords(const CutAxis1D& ax) {
    std::vector<double> grev = greville_points(ax.knots, ax.degree, ax.n), out;
    for (int g : ax.interior) out.push_back(grev[g]);
    return out;
}

#endif // COMP_CUT3D_H
