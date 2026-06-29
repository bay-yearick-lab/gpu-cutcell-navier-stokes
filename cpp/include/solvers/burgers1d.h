// burgers1d.h
//
// Energy-CONSERVING wall advection on a bounded B-spline space -- the kernel of
// the wall-NS frontier, demonstrated on 1D Burgers, u_t + (u^2/2)_x = nu u_xx.
//
// WHY THIS WORKS WHERE THE COLLOCATED WALL FAILED. The B-spline mimetic
// operators carry a FULL (non-diagonal) energy norm H = B^-T M B^-1. The usual
// NODAL split-form energy conservation telescopes only for a DIAGONAL norm, so
// the collocated wall advection injected energy. Here we advect in COEFFICIENT
// space (isogeometric Galerkin): u = sum_j alpha_j N_j, and the convective term
// is the Galerkin trilinear form
//
//   C_i = (1/3)[ \int N_i d_x(u^2) dx + \int N_i u u_x dx ]
//       = (1/3)[ ([N_i u^2])_0^L - \int N_i' u^2 + \int N_i u u_x ].
//
// Its energy contribution is EXACT by the weak integration by parts:
//   alpha^T C = (1/3)\int (u d_x(u^2) + u^2 u_x) dx = \int u^2 u_x dx
//             = [u^3/3]_0^L            (the physical WALL FLUX, nothing else),
// because the two cubic volume integrals cancel identically (the discrete
// tensors are Gauss-exact). With u=0 (no-slip) walls the flux is zero, so the
// semi-discrete kinetic energy 1/2 alpha^T M alpha is conserved to ROUND-OFF on
// the bounded domain -- independent of the norm being non-diagonal. This is the
// energy-conserving wall advection; in 3D NS it applies per coordinate axis
// (the 3d_to_1d collapse), one coefficient-space line solve at a time.

#ifndef BURGERS1D_H
#define BURGERS1D_H

#include <vector>
#include <map>
#include <cmath>
#include "bspline.h"
#include "linalg.h"

// Convective 3-tensor  D_ikl = \int (N_i N_k N_l' - N_i' N_k N_l) dx  on the
// clamped degree-p basis. Banded (i,k,l within one span), Gauss-exact.
struct ConvTensor {
    struct E { int i, k, l; double v; };
    std::vector<E> ent;
};
inline ConvTensor assemble_conv_clamped(int n, int p,
                                        const std::vector<double>& knots) {
    std::map<long long, double> acc;
    auto key = [n](int i, int k, int l) {
        return (static_cast<long long>(i) * n + k) * n + l;
    };
    std::vector<double> gx, gw;
    gauss_legendre(2 * p, gx, gw);          // exact to degree 4p-1 >= 3p-1
    std::vector<std::vector<double>> ders;
    for (int s = p; s < n; ++s) {
        const double a = knots[s], b = knots[s + 1];
        if (b <= a) continue;
        const double half = 0.5 * (b - a), mid = 0.5 * (a + b);
        for (size_t q = 0; q < gx.size(); ++q) {
            const double xq = mid + half * gx[q], wq = gw[q] * half;
            ders_basis_funs(s, xq, p, 1, knots, ders);
            for (int aa = 0; aa <= p; ++aa) {
                const int i = s - p + aa;
                const double Ni = ders[0][aa], dNi = ders[1][aa];
                for (int bb = 0; bb <= p; ++bb) {
                    const int k = s - p + bb;
                    const double Nk = ders[0][bb];
                    for (int cc = 0; cc <= p; ++cc) {
                        const int l = s - p + cc;
                        const double Nl = ders[0][cc], dNl = ders[1][cc];
                        acc[key(i, k, l)] += wq * (Ni * Nk * dNl - dNi * Nk * Nl);
                    }
                }
            }
        }
    }
    ConvTensor T;
    for (const auto& kv : acc)
        if (std::fabs(kv.second) > 1e-14) {
            long long kk = kv.first;
            const int l = kk % n; kk /= n;
            const int k = kk % n;
            const int i = kk / n;
            T.ent.push_back({i, k, l, kv.second});
        }
    return T;
}

struct BurgersBounded {
    int n, p;
    double L, nu;
    std::vector<double> knots, grev;
    Mat M, S;
    BandSolver Mfac;
    ConvTensor T;
    DenseMatrix B;                      // sample N_j(greville_i), for I/O
    std::vector<double> alpha, a0, kk;  // state + RK scratch
    bool dirichlet = true;              // pin u=0 at both walls (no-slip)

    BurgersBounded(int n_, int p_, double L_, double nu_)
        : n(n_), p(p_), L(L_), nu(nu_) {
        knots = open_uniform_knots(n, p, L);
        grev = greville_points(knots, p, n);
        CoeffForms cf = assemble_forms_clamped(n, p, knots);
        M = cf.M; S = cf.S;
        DenseMatrix Md(n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) Md(i, j) = M(i, j);
        Mfac.set_from_dense(Md, p, p); Mfac.factor();
        T = assemble_conv_clamped(n, p, knots);
        B = DenseMatrix(n);
        std::vector<std::vector<double>> ders;
        for (int r = 0; r < n; ++r) {
            const int sp = find_span(n - 1, p, grev[r], knots);
            ders_basis_funs(sp, grev[r], p, 0, knots, ders);
            for (int a = 0; a <= p; ++a) B(r, sp - p + a) = ders[0][a];
        }
        alpha.assign(n, 0.0); a0 = alpha; kk = alpha;
    }

    // C_i = (1/3)[ sum_kl D_ikl a_k a_l + wall_i ],  wall_0=-u(0)^2, wall_{n-1}=u(L)^2
    void conv(const std::vector<double>& a, std::vector<double>& C) const {
        C.assign(n, 0.0);
        for (const auto& e : T.ent) C[e.i] += e.v * a[e.k] * a[e.l];
        C[0]     += -a[0] * a[0];
        C[n - 1] +=  a[n - 1] * a[n - 1];
        for (double& c : C) c *= (1.0 / 3.0);
    }

    // semi-discrete dE/dt from the spatial scheme (inviscid part):
    //   alpha^T M dalpha/dt|_conv = -alpha^T C = -(wall flux) = -[u^3/3]_0^L.
    double dEdt_conv(const std::vector<double>& a) const {
        std::vector<double> C; conv(a, C);
        double s = 0.0; for (int i = 0; i < n; ++i) s += a[i] * C[i];
        return -s;
    }
    double wall_flux(const std::vector<double>& a) const {     // -[u^3/3]_0^L
        return -(a[n - 1] * a[n - 1] * a[n - 1] - a[0] * a[0] * a[0]) / 3.0;
    }

    // dalpha/dt = M^{-1}( -C - nu S alpha )   (natural / zero-stress viscous walls)
    void rhs(const std::vector<double>& a, std::vector<double>& da) const {
        std::vector<double> C; conv(a, C);
        da.assign(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double Sa = 0.0;
            if (nu != 0.0) for (int j = 0; j < n; ++j) Sa += S(i, j) * a[j];
            da[i] = -C[i] - nu * Sa;
        }
        Mfac.solve_inplace(da);
    }

    void pin() { if (dirichlet) { alpha[0] = 0.0; alpha[n - 1] = 0.0; } }
    void step(double dt) {
        a0 = alpha;
        rhs(alpha, kk); for (int i = 0; i < n; ++i) alpha[i] = a0[i] + dt * kk[i]; pin();
        rhs(alpha, kk); for (int i = 0; i < n; ++i) alpha[i] = 0.75 * a0[i] + 0.25 * (alpha[i] + dt * kk[i]); pin();
        rhs(alpha, kk); for (int i = 0; i < n; ++i) alpha[i] = (1.0/3.0) * a0[i] + (2.0/3.0) * (alpha[i] + dt * kk[i]); pin();
    }

    double energy() const {                       // 1/2 alpha^T M alpha = 1/2 \int u^2
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            double Ma = 0.0; for (int j = 0; j < n; ++j) Ma += M(i, j) * alpha[j];
            e += alpha[i] * Ma;
        }
        return 0.5 * e;
    }
    std::vector<double> nodal() const {           // u(greville) = B alpha
        std::vector<double> u(n, 0.0);
        for (int i = 0; i < n; ++i) { double s = 0; for (int j = 0; j < n; ++j) s += B(i, j) * alpha[j]; u[i] = s; }
        return u;
    }
    void set_nodal(const std::vector<double>& u) { dense_lu_solve(B, u, alpha); pin(); }
};

#endif  // BURGERS1D_H
