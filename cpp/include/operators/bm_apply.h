// bm_apply.h
//
// The PRODUCTION application path for the B-spline mimetic derivatives: the
// reduce-solve-expand chain (see cpp/README.md). The dense operators in
// bm_operators.h are built only to verify identities and measure order; here the
// same maps are applied without ever materializing an n x n matrix.
//
//   BMG D1 = B M^{-1} K B^{-1}   applied to nodal u as
//       u --> alpha (solve B alpha = u)        banded / cyclic solve, O(n*bw^2) once
//          --> r = K alpha                     banded matvec,         O(n*bw)
//          --> beta (solve M beta = r)         banded / cyclic solve
//          --> y = B beta                      banded matvec
//
//   BMC D1 = B' B^{-1},  BMC D2 = B'' B^{-1}   solve B, then one banded matvec.
//   BMG D2 (periodic) = -B M^{-1} S B^{-1}     same chain with S, negated.
//   BMF G  = Bhat_eta Delta B^{-1}             solve B, Delta (bidiagonal), matvec.
//
// PERIODIC operators are circulant: each is one stencil of length 2*bw+1 and the
// solves use CyclicSolver::set_from_stencil. Nothing is O(n^2) -- storage and
// per-apply cost are O(n*bw), so a single line scales to millions of points.
// BOUNDED operators are banded; they keep dense-backed storage here because the
// bounded path is only exercised for correctness at modest n. B and M are
// factored ONCE and reused for every right-hand side -- every grid line, every
// time step. The bounded wide second derivatives (BMF Dv G and BMG flux form)
// are not pure reduce-solve-expand chains and stay in bm_operators.h.

#ifndef BM_APPLY_H
#define BM_APPLY_H

#include <vector>
#include "bspline.h"

// A circulant operator stored as one centered stencil st[o + bw], o in [-bw,bw].
// The matvec splits into a wrap-free interior (the bulk, a stride-1 dot product
// the compiler vectorizes with FMA) and the 2*bw boundary rows that wrap. Keeping
// the modulo out of the hot loop is what makes this fast.
struct Circulant {
    int bw = 0;
    std::vector<double> st;          // length 2*bw+1
    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        const int n = (int)x.size();
        const int w = 2 * bw + 1;
        const double* s = st.data();
        for (int i = bw; i < n - bw; ++i) {              // interior: no wrap
            const double* xp = &x[i - bw];
            double acc = 0.0;
            for (int o = 0; o < w; ++o) acc += s[o] * xp[o];
            y[i] = acc;
        }
        auto wrap_row = [&](int i) {
            double acc = 0.0;
            for (int o = -bw; o <= bw; ++o) acc += s[o + bw] * x[((i + o) % n + n) % n];
            y[i] = acc;
        };
        for (int i = 0; i < bw && i < n; ++i)      wrap_row(i);
        for (int i = (n - bw > bw ? n - bw : bw); i < n; ++i) wrap_row(i);
    }
    std::vector<double> matvec(const std::vector<double>& x) const {
        std::vector<double> y(x.size());
        matvec(x, y);
        return y;
    }
    // Batched matvec: x, y are n x W row-major; W lanes processed together.
    void matvec_block(const double* x, double* y, int n, int W) const {
        const int wd = 2 * bw + 1;
        const double* s = st.data();
        for (int i = bw; i < n - bw; ++i) {              // interior: no wrap
            double* yi = y + static_cast<size_t>(i) * W;
            for (int l = 0; l < W; ++l) yi[l] = 0.0;
            const double* xp = x + static_cast<size_t>(i - bw) * W;
            for (int o = 0; o < wd; ++o) {
                const double so = s[o];
                const double* xo = xp + static_cast<size_t>(o) * W;
                for (int l = 0; l < W; ++l) yi[l] += so * xo[l];
            }
        }
        auto wrap_row = [&](int i) {
            double* yi = y + static_cast<size_t>(i) * W;
            for (int l = 0; l < W; ++l) yi[l] = 0.0;
            for (int o = -bw; o <= bw; ++o) {
                const int j = ((i + o) % n + n) % n;
                const double so = s[o + bw];
                const double* xj = x + static_cast<size_t>(j) * W;
                for (int l = 0; l < W; ++l) yi[l] += so * xj[l];
            }
        };
        for (int i = 0; i < bw && i < n; ++i) wrap_row(i);
        for (int i = (n - bw > bw ? n - bw : bw); i < n; ++i) wrap_row(i);
    }
};

// Banded matvec y = A x touching only the half-bandwidth bw (bounded case).
inline void matvec_banded(const Mat& A, const std::vector<double>& x, int bw,
                          std::vector<double>& y) {
    const int n = A.r;
    for (int i = 0; i < n; ++i) {
        const int jlo = (i - bw > 0) ? i - bw : 0;
        const int jhi = (i + bw < n - 1) ? i + bw : n - 1;
        double s = 0.0;
        for (int j = jlo; j <= jhi; ++j) s += A(i, j) * x[j];
        y[i] = s;
    }
}
inline std::vector<double> matvec_banded(const Mat& A, const std::vector<double>& x, int bw) {
    std::vector<double> y(A.r);
    matvec_banded(A, x, bw, y);
    return y;
}

// Factored, banded application of the mimetic derivatives on one 1D line. Build
// once per (domain, degree, n); apply to as many fields as needed.
struct FastDeriv {
    bool periodic = false;
    int n = 0, p = 0; double h = 0.0;          // grid spacing (for explicit-FD baseline)

    // periodic (circulant stencil) storage -- O(n*bw)
    Circulant cB, cBp, cBpp, cK, cS;
    CyclicSolver Bcyc, Mcyc;

    // local high-resolution baselines (periodic): compact Pade C6
    // LHS tridiagonal solve + RHS stencil; DRP explicit 7-point D1.
    CyclicSolver L1cyc, L2cyc;               // compact LHS (tridiagonal) for D1, D2
    std::vector<double> cr1, cr2, drp1;      // compact RHS (offsets -2..2); DRP D1 (-3..3)

    // bounded (banded, dense-backed) storage
    Mat B, Bp, Bpp, K, S;
    int bw = 0;
    DerivativeSpace ds;
    BandSolver Bband, Mband;

    mutable std::vector<double> s1, s2;     // reusable scratch for the in-place path

    static FastDeriv build(bool periodic, int n, int p, double L) {
        FastDeriv f;
        f.periodic = periodic; f.n = n; f.p = p; f.bw = p; f.h = L / n;
        f.s1.assign(n, 0.0); f.s2.assign(n, 0.0);
        if (periodic) {
            const double h = L / n, inv_h = 1.0 / h;
            const int bwB = (p + 1) / 2;            // value/derivative support radius
            // Circulant stencils computed directly from the cardinal kernels --
            // no n x n matrix. Offset o = j - i; build_periodic uses t = -o.
            f.cB.bw = bwB;   f.cB.st.resize(2 * bwB + 1);
            f.cBp.bw = bwB;  f.cBp.st.resize(2 * bwB + 1);
            f.cBpp.bw = bwB; f.cBpp.st.resize(2 * bwB + 1);
            for (int o = -bwB; o <= bwB; ++o) {
                const double t = -o;
                f.cB.st[o + bwB]   = beta_value(p, t);
                f.cBp.st[o + bwB]  = (p >= 1) ? beta_deriv1(p, t) * inv_h : 0.0;
                f.cBpp.st[o + bwB] = (p >= 2) ? beta_deriv2(p, t) * inv_h * inv_h : 0.0;
            }
            f.cK.bw = p; f.cK.st.resize(2 * p + 1);
            f.cS.bw = p; f.cS.st.resize(2 * p + 1);
            std::vector<double> stM(2 * p + 1, 0.0);
            for (int o = -p; o <= p; ++o) {
                stM[o + p]        = h * beta_overlap(p, o, 0, 0);
                f.cK.st[o + p]    = (p >= 1) ? beta_overlap(p, o, 0, 1) : 0.0;
                f.cS.st[o + p]    = (p >= 1) ? beta_overlap(p, o, 1, 1) * inv_h : 0.0;
            }
            f.Bcyc.set_from_stencil(n, bwB, f.cB.st);
            f.Mcyc.set_from_stencil(n, p, stM);
            // compact Pade C6: D1 alpha=1/3,a=14/9,b=1/9 ; D2 alpha=2/11,a=12/11,b=3/11
            { const double a1=14.0/9, b1=1.0/9, al1=1.0/3;
              f.cr1.assign(5, 0.0);
              f.cr1[2+1]= a1*inv_h/2;  f.cr1[2-1]=-a1*inv_h/2;
              f.cr1[2+2]= b1*inv_h/4;  f.cr1[2-2]=-b1*inv_h/4;
              f.L1cyc.set_from_stencil(n, 1, std::vector<double>{al1, 1.0, al1});
              const double a2=12.0/11, b2=3.0/11, al2=2.0/11, ih2=inv_h*inv_h;
              f.cr2.assign(5, 0.0);
              f.cr2[2+1]= a2*ih2;  f.cr2[2-1]= a2*ih2;
              f.cr2[2+2]= b2*ih2/4; f.cr2[2-2]= b2*ih2/4;
              f.cr2[2+0]= -(2*a2 + b2/2)*ih2;
              f.L2cyc.set_from_stencil(n, 1, std::vector<double>{al2, 1.0, al2}); }
            // DRP explicit 7-point first derivative (antisymmetric), the original
            // published DRP coefficients.
            { const double d1=0.79926643, d2=-0.18941314, d3=0.02651995;
              f.drp1.assign(7, 0.0);
              f.drp1[3+1]= d1*inv_h; f.drp1[3-1]=-d1*inv_h;
              f.drp1[3+2]= d2*inv_h; f.drp1[3-2]=-d2*inv_h;
              f.drp1[3+3]= d3*inv_h; f.drp1[3-3]=-d3*inv_h; }
        } else {
            SplineOperators op = build_clamped(n, p, L);
            std::vector<double> knots = open_uniform_knots(n, p, L);
            CoeffForms cf = assemble_forms_clamped(n, p, knots);
            f.B = to_mat(op.B); f.Bp = to_mat(op.Bp); f.Bpp = to_mat(op.Bpp);
            f.K = cf.K; f.S = cf.S;
            f.ds = build_derivative_space(n, p, knots);
            f.Bband.set_from_dense(op.B, p, p); f.Bband.factor();
            f.Mband.set_from_dense(to_dense(cf.M), p, p); f.Mband.factor();
        }
        return f;
    }

    // --- dispatched primitives (periodic stencil vs bounded band) -----------
    std::vector<double> applyB(const std::vector<double>& x) const {
        return periodic ? cB.matvec(x) : matvec_banded(B, x, bw);
    }
    std::vector<double> applyBp(const std::vector<double>& x) const {
        return periodic ? cBp.matvec(x) : matvec_banded(Bp, x, bw);
    }
    std::vector<double> applyBpp(const std::vector<double>& x) const {
        return periodic ? cBpp.matvec(x) : matvec_banded(Bpp, x, bw);
    }
    std::vector<double> applyK(const std::vector<double>& x) const {
        return periodic ? cK.matvec(x) : matvec_banded(K, x, bw);
    }
    std::vector<double> applyS(const std::vector<double>& x) const {
        return periodic ? cS.matvec(x) : matvec_banded(S, x, bw);
    }
    std::vector<double> solveB(const std::vector<double>& r) const {
        return periodic ? Bcyc.solve(r) : Bband.solve(r);
    }
    std::vector<double> solveM(const std::vector<double>& r) const {
        return periodic ? Mcyc.solve(r) : Mband.solve(r);
    }

    // --- operators ----------------------------------------------------------
    // BMC same-point derivatives: reduce to coefficients, sample the derivative.
    std::vector<double> bmc_d1(const std::vector<double>& u) const { return applyBp(solveB(u)); }
    std::vector<double> bmc_d2(const std::vector<double>& u) const { return applyBpp(solveB(u)); }

    // BMG Galerkin first derivative: reduce-solve-expand through the mass metric.
    std::vector<double> bmg_d1(const std::vector<double>& u) const {
        return applyB(solveM(applyK(solveB(u))));
    }
    // Zero-allocation periodic version with caller-provided scratch (a, b) and
    // output (y). Thread-safe: the factorization is read-only, so many threads
    // can share one FastDeriv and apply it to independent lines in parallel,
    // each with its own buffers. This is the production hot path.
    void bmg_d1(const std::vector<double>& u, std::vector<double>& y,
                std::vector<double>& a, std::vector<double>& b) const {
        a = u;                             // reduce: a = B^{-1} u
        reduceB(a);
        into_K(a, b);                      // b = K a
        reduceM(b);                        // solve: b = M^{-1} b
        expandB(b, y);                     // expand: y = B b
    }
    // Convenience single-thread overload using the member scratch.
    void bmg_d1(const std::vector<double>& u, std::vector<double>& y) const {
        bmg_d1(u, y, s1, s2);
    }
    // BMG periodic direct second derivative: -B M^{-1} S B^{-1}.
    std::vector<double> bmg_d2_periodic(const std::vector<double>& u) const {
        std::vector<double> y = applyB(solveM(applyS(solveB(u))));
        for (double& v : y) v = -v;
        return y;
    }

    // BMF staggered gradient G = Bhat_eta Delta B^{-1}: primal -> derivative grid.
    std::vector<double> bmf_grad(const std::vector<double>& u) const {
        std::vector<double> a = solveB(u);
        std::vector<double> d(ds.m, 0.0);                 // Delta is bidiagonal
        for (int i = 0; i < ds.m; ++i)
            d[i] = ds.Delta(i, i) * a[i] + ds.Delta(i, i + 1) * a[i + 1];
        return matvec_banded(ds.Bhat, d, p);
    }

    // -----------------------------------------------------------------------
    // Zero-allocation, dispatched (periodic stencil / bounded band) primitives
    // and operators, all taking caller-provided scratch. These are the methods
    // the 3D engine calls per line, in parallel, with per-thread buffers.
    // -----------------------------------------------------------------------
    void reduceB(std::vector<double>& x) const {
        if (periodic) Bcyc.solve_inplace(x); else Bband.solve_inplace(x);
    }
    void reduceM(std::vector<double>& x) const {
        if (periodic) Mcyc.solve_inplace(x); else Mband.solve_inplace(x);
    }
    void expandB(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cB.matvec(x, y); else matvec_banded(B, x, bw, y);
    }
    void into_Bp(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cBp.matvec(x, y); else matvec_banded(Bp, x, bw, y);
    }
    void into_Bpp(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cBpp.matvec(x, y); else matvec_banded(Bpp, x, bw, y);
    }
    void into_K(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cK.matvec(x, y); else matvec_banded(K, x, bw, y);
    }
    void into_S(const std::vector<double>& x, std::vector<double>& y) const {
        if (periodic) cS.matvec(x, y); else matvec_banded(S, x, bw, y);
    }

    // BMC same-point derivatives: reduce, then sample. scratch a (size n).
    void bmc_d1(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a) const {
        a = u; reduceB(a); into_Bp(a, y);
    }
    void bmc_d2(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a) const {
        a = u; reduceB(a); into_Bpp(a, y);
    }
    // BMG periodic direct second derivative: -B M^{-1} S B^{-1}. scratch a, b.
    void bmg_d2(const std::vector<double>& u, std::vector<double>& y,
                std::vector<double>& a, std::vector<double>& b) const {
        a = u; reduceB(a); into_S(a, b); reduceM(b); expandB(b, y);
        for (double& v : y) v = -v;
    }

    // Explicit centered periodic finite differences (2nd/4th/6th order) -- the FD
    // baselines for the cost comparison (no banded solve; cheap but dispersive).
    void fd2_d1(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih = 1.0/h; y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        for (int i = 0; i < n; ++i) y[i] = ih*0.5*(U(i,1)-U(i,-1));
    }
    void fd2_d2(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0/(h*h); y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        for (int i = 0; i < n; ++i) y[i] = ih2*(U(i,1)-2*U(i,0)+U(i,-1));
    }
    void fd4_d1(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih = 1.0/h; y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c1 = 2.0/3, c2 = -1.0/12;
        for (int i = 0; i < n; ++i) y[i] = ih*(c1*(U(i,1)-U(i,-1)) + c2*(U(i,2)-U(i,-2)));
    }
    void fd4_d2(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0/(h*h); y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c0 = -5.0/2, c1 = 4.0/3, c2 = -1.0/12;
        for (int i = 0; i < n; ++i) y[i] = ih2*(c0*U(i,0) + c1*(U(i,1)+U(i,-1)) + c2*(U(i,2)+U(i,-2)));
    }
    // Compact Pade C6 (periodic): solve LHS f' = RHS u (tridiagonal cyclic solve).
    void compact_d1(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t(n);
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int o = -2; o <= 2; ++o) s += cr1[o+2]*u[((i+o)%n+n)%n];
            t[i] = s;
        }
        y = L1cyc.solve(t);
    }
    void compact_d2(const std::vector<double>& u, std::vector<double>& y) const {
        std::vector<double> t(n);
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int o = -2; o <= 2; ++o) s += cr2[o+2]*u[((i+o)%n+n)%n];
            t[i] = s;
        }
        y = L2cyc.solve(t);
    }
    // DRP explicit first derivative; DRP-family diffusion uses E6 d2.
    void drp_d1(const std::vector<double>& u, std::vector<double>& y) const {
        y.resize(n);
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int o = -3; o <= 3; ++o) s += drp1[o+3]*u[((i+o)%n+n)%n];
            y[i] = s;
        }
    }
    void fd6_d1(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih = 1.0 / h; y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c1 = 3.0/4, c2 = -3.0/20, c3 = 1.0/60;
        for (int i = 0; i < n; ++i)
            y[i] = ih*(c1*(U(i,1)-U(i,-1)) + c2*(U(i,2)-U(i,-2)) + c3*(U(i,3)-U(i,-3)));
    }
    void fd6_d2(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0 / (h*h); y.resize(n);
        auto U = [&](int i, int o){ return u[((i+o)%n+n)%n]; };
        const double c0 = -49.0/18, c1 = 3.0/2, c2 = -3.0/20, c3 = 1.0/90;
        for (int i = 0; i < n; ++i)
            y[i] = ih2*(c0*U(i,0) + c1*(U(i,1)+U(i,-1)) + c2*(U(i,2)+U(i,-2)) + c3*(U(i,3)+U(i,-3)));
    }

    // -----------------------------------------------------------------------
    // Batched periodic BMC apply: W lines at once (in, out, scratch are n x W
    // row-major). The reduce is one circulant block solve; the expand is one
    // block matvec. This is the cache- and SIMD-efficient path for the strided
    // (y, z) directions of a 3D field, where W adjacent lines are contiguous.
    // -----------------------------------------------------------------------
    void bmc_d2_block(const double* in, double* out, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Bcyc.solve_block(sc, W);
        cBpp.matvec_block(sc, out, n, W);
    }
    void bmc_d1_block(const double* in, double* out, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Bcyc.solve_block(sc, W);
        cBp.matvec_block(sc, out, n, W);
    }
    // One coefficient solve feeds both derivatives (advection + diffusion).
    void bmc_d1d2_block(const double* in, double* d1, double* d2, double* sc, int W) const {
        const size_t nW = static_cast<size_t>(n) * W;
        for (size_t i = 0; i < nW; ++i) sc[i] = in[i];
        Bcyc.solve_block(sc, W);
        cBp.matvec_block(sc, d1, n, W);
        cBpp.matvec_block(sc, d2, n, W);
    }
};

#endif // BM_APPLY_H
