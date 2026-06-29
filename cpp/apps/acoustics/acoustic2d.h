#ifndef ACOUSTIC2D_H
#define ACOUSTIC2D_H
//
// acoustic2d.h -- 2D open-domain linear-acoustics radiation engine for the
// vortex-sound (computational aeroacoustics) study. First-order system on a
// periodic grid (so the fast B-spline mimetic D1 = gradient/divergence applies),
// with a SPONGE ring near the edges absorbing outgoing waves -> open boundaries:
//
//   p_t = -c^2 (u_x + v_y) + S            (mass source S; monopole test)
//   u_t = -p_x + fx ,  v_t = -p_y + fy    (momentum source f; Lighthill vortex sound)
//   minus sigma(x) * (p,u,v) in the sponge band.
//
// Lighthill aeroacoustic source as a body force: f = -(u.grad)u of the
// incompressible near-field -- the flow's own nonlinear self-advection radiates.
// Space: BMG (spline mimetic Galerkin D1, the low-dispersion wave route) or E6.

#include <cstdio>
#include <cmath>
#include <vector>
#include "bm_apply.h"

static const double ACPI = 3.14159265358979323846;
enum AScheme { A_BMG, A_E6 };

struct Field {
    int N; std::vector<double> a;
    Field(int N_ = 0) : N(N_), a((size_t)N_ * N_, 0.0) {}
    double& operator()(int i, int j)       { return a[(size_t)j * N + i]; }
    double  operator()(int i, int j) const { return a[(size_t)j * N + i]; }
    void zero() { std::fill(a.begin(), a.end(), 0.0); }
};

struct Acoustic2D {
    int N; double L, h, c; AScheme sc; FastDeriv f;
    std::vector<double> spng;
    Acoustic2D(int N_, double L_, double c_, int p, AScheme sc_)
        : N(N_), L(L_), h(L_ / N_), c(c_), sc(sc_),
          f(FastDeriv::build(true, N_, p, L_)), spng((size_t)N_ * N_, 0.0) {
        const double wsp = 0.18 * L, smax = 18.0 * c / wsp;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            const double x = i * h, y = j * h;
            const double d = std::min(std::min(x, L - x), std::min(y, L - y));
            double s = 0.0; if (d < wsp) { const double t = (wsp - d) / wsp; s = smax * t * t; }
            spng[(size_t)j * N + i] = s;
        }
    }
    void d1_line(const std::vector<double>& u, std::vector<double>& y) const {
        if (sc == A_BMG) { y = f.bmg_d1(u); return; }
        const double ih = 1.0 / h; const int n = (int)u.size();
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c1 = 3.0/4, c2 = -3.0/20, c3 = 1.0/60;
        for (int i = 0; i < n; ++i) y[i] = ih*(c1*(U(i,1)-U(i,-1))+c2*(U(i,2)-U(i,-2))+c3*(U(i,3)-U(i,-3)));
    }
    void dx(const Field& A, Field& Ax) const {
        std::vector<double> line(N), out(N);
        for (int j=0;j<N;++j){ for(int i=0;i<N;++i) line[i]=A(i,j); d1_line(line,out); for(int i=0;i<N;++i) Ax(i,j)=out[i]; }
    }
    void dy(const Field& A, Field& Ay) const {
        std::vector<double> line(N), out(N);
        for (int i=0;i<N;++i){ for(int j=0;j<N;++j) line[j]=A(i,j); d1_line(line,out); for(int j=0;j<N;++j) Ay(i,j)=out[j]; }
    }
    // rhs with mass source S and momentum source (fx,fy). Pass zeroed fields for unused.
    void rhs(const Field& p,const Field& u,const Field& v,
             const Field& S,const Field& fx,const Field& fy,
             Field& dp,Field& du,Field& dv,Field& t1,Field& t2) const {
        dx(u,t1); dy(v,t2);
        for (size_t k=0;k<dp.a.size();++k) dp.a[k] = -c*c*(t1.a[k]+t2.a[k]) + S.a[k] - spng[k]*p.a[k];
        dx(p,t1); for (size_t k=0;k<du.a.size();++k) du.a[k] = -t1.a[k] + fx.a[k] - spng[k]*u.a[k];
        dy(p,t2); for (size_t k=0;k<dv.a.size();++k) dv.a[k] = -t2.a[k] + fy.a[k] - spng[k]*v.a[k];
    }
};

#endif // ACOUSTIC2D_H
