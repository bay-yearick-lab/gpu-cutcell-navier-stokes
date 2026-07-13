// mms_tgdecay.cpp
//
// QUANTITATIVE Taylor-Green no-slip decay gate for the walled-box primitive
// NS loop (wall_ns2d.h). The qualitative MMS suite (stable, monotone decay,
// div held) let two silent defects through in 2026-07; this test pins the
// actual decay physics against externally verified reference values.
//
// Setup: psi = sin^2(pi x) sin^2(pi z) / pi on [0,1]^2, Re = 400 (nu = 1/400):
//   u =  sin^2(pi x) sin(2 pi z),   w = -sin(2 pi x) sin^2(pi z)
// (exactly no-slip on all four walls). March dt = 0.04/N to T = 0.48, so
// nsteps = 12 N exactly.
//
// Reference values (cross-verified against two independent solvers):
//   * initial dissipation  nu * int |grad u|^2 = 2 nu pi^2 = 4.93480220e-2
//   * KE(0)   = 3/16 = 0.1875 exactly
//   * KE(0.48) = 0.1647438
//
// Gates:
//   (G0) nu * u^T stiff(u) called EXACTLY as rhs() calls it (out = t1) must
//        match an independent Gauss-quadrature integral of nu*|grad u_h|^2 to
//        round-off: they are the same bilinear form. Catches scratch-aliasing
//        corruption of the viscous operator at unit level.
//   (G1) quadrature dissipation of the projected IC -> 2 nu pi^2.
//   (G2) KE(0.48) within 5e-4 of 0.1647438 at the finest N AND the error
//        non-increasing along the ladder. The 0.1647438 reference is the
//        BMF constrained-trace solver (N=33/49,
//        final_KE 0.16474380/0.16474373); the pinned fractional-step scheme
//        here carries a wall-splitting floor ~1.5e-4 against it, which is
//        the verified reference behavior (N=16/24/32 -> 2.0e-4/1.6e-4/
//        1.5e-4, decreasing). The two 2026-07 defects sat far outside this
//        envelope: KE errors 1.9e-3 -> 3.4e-3, GROWING with N.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "wall_ns2d.h"

static const double PI = 3.14159265358979323846;
static const double REF_DISS = 2.0 * (1.0 / 400.0) * PI * PI;   // 4.93480220e-2
static const double REF_KE_T = 0.1647438;                        // KE at T = 0.48

// Greville-sample the analytic IC and interpolate to spline coefficients
static void tg_ic(WallNS2D& S) {
    const int nx = S.X.n, nz = S.Z.n;
    for (int i = 0; i < nx; ++i) for (int j = 0; j < nz; ++j) {
        const double x = S.X.grev[i], z = S.Z.grev[j];
        const double sx = std::sin(PI * x), cx = std::cos(PI * x);
        const double sz = std::sin(PI * z), cz = std::cos(PI * z);
        S.u.at(i, j) = 2.0 * sx * sx * sz * cz;
        S.w.at(i, j) = -2.0 * sx * cx * sz * sz;
    }
    std::vector<double> line, sol;
    for (Field2D* F : {&S.u, &S.w}) {
        line.assign(nx, 0.0);
        for (int j = 0; j < nz; ++j) {
            for (int i = 0; i < nx; ++i) line[i] = F->at(i, j);
            dense_lu_solve(S.X.B, line, sol);
            for (int i = 0; i < nx; ++i) F->at(i, j) = sol[i];
        }
        line.assign(nz, 0.0);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nz; ++j) line[j] = F->at(i, j);
            dense_lu_solve(S.Z.B, line, sol);
            for (int j = 0; j < nz; ++j) F->at(i, j) = sol[j];
        }
    }
}

// independent Gauss-quadrature integral of |grad u_h|^2 + |grad w_h|^2.
// Deliberately does NOT go through stiff(): an adversarial recomputation of
// the same bilinear form through a different code path.
static double grad_sq_quad(const WallNS2D& S) {
    const int px = S.X.p, pz = S.Z.p;
    std::vector<double> gx, gw; gauss_legendre(2 * std::max(px, pz), gx, gw);
    std::vector<std::vector<double>> dx, dz;
    double acc = 0.0;
    for (int sx = px; sx < S.X.n; ++sx) {
        const double ax = S.X.knots[sx], bx = S.X.knots[sx + 1]; if (bx <= ax) continue;
        const double hx = 0.5 * (bx - ax), mx = 0.5 * (ax + bx);
        for (int sz = pz; sz < S.Z.n; ++sz) {
            const double az = S.Z.knots[sz], bz = S.Z.knots[sz + 1]; if (bz <= az) continue;
            const double hz = 0.5 * (bz - az), mz = 0.5 * (az + bz);
            for (size_t qx = 0; qx < gx.size(); ++qx) {
                ders_basis_funs(sx, mx + hx * gx[qx], px, 1, S.X.knots, dx);
                for (size_t qz = 0; qz < gx.size(); ++qz) {
                    ders_basis_funs(sz, mz + hz * gx[qz], pz, 1, S.Z.knots, dz);
                    const double wq = gw[qx] * gw[qz] * hx * hz;
                    double Ux = 0, Uz = 0, Wx = 0, Wz = 0;
                    for (int a = 0; a <= px; ++a) {
                        const int gi = sx - px + a;
                        for (int b = 0; b <= pz; ++b) {
                            const int gj = sz - pz + b;
                            const double uu = S.u.at(gi, gj), ww = S.w.at(gi, gj);
                            const double dNx = dx[1][a] * dz[0][b], dNz = dx[0][a] * dz[1][b];
                            Ux += uu * dNx; Uz += uu * dNz;
                            Wx += ww * dNx; Wz += ww * dNz;
                        }
                    }
                    acc += wq * (Ux * Ux + Uz * Uz + Wx * Wx + Wz * Wz);
                }
            }
        }
    }
    return acc;
}

struct RunResult { double d0_quad, d0_stiff, keT; };

static RunResult run_case(int N, int p) {
    const double nu = 1.0 / 400.0;
    WallNS2D S(N, N, p, 1.0, 1.0, nu);
    tg_ic(S);
    S.no_slip(S.u); S.no_slip(S.w); S.project();

    RunResult r;
    r.d0_quad = nu * grad_sq_quad(S);
    // the viscous term EXACTLY as rhs() computes it: stiff into the shared t1
    S.stiff(S.u, S.t1);
    double us = 0; for (size_t q = 0; q < S.u.a.size(); ++q) us += S.u.a[q] * S.t1.a[q];
    S.stiff(S.w, S.t1);
    for (size_t q = 0; q < S.w.a.size(); ++q) us += S.w.a[q] * S.t1.a[q];
    r.d0_stiff = nu * us;

    const double dt = 0.04 / N;
    const int nsteps = 12 * N;                       // T = 0.48 exactly
    for (int n = 0; n < nsteps; ++n) S.step(dt);
    r.keT = S.kinetic_energy();
    return r;
}

int main(int argc, char** argv) {
    const int p = 4;
    std::vector<int> ladder = {16, 24, 32};
    if (argc > 1) { ladder.clear(); for (int a = 1; a < argc; ++a) ladder.push_back(std::atoi(argv[a])); }

    std::printf("\n TAYLOR-GREEN NO-SLIP DECAY GATE (wall_ns2d.h, p=%d, Re=400, T=0.48)\n", p);
    std::printf(" reference: diss0 = 2 nu pi^2 = %.8e, KE(T) = %.7f\n\n", REF_DISS, REF_KE_T);
    std::printf("    N    nu*quad|grad|^2   nu*u'stiff(u)    KE(T)       |KE err|\n");

    int fail = 0;
    double prev_err = 0.0; bool have_prev = false;
    double last_order = 0.0;
    for (size_t c = 0; c < ladder.size(); ++c) {
        const int N = ladder[c];
        const RunResult r = run_case(N, p);
        const double ke_err = std::fabs(r.keT - REF_KE_T);
        std::printf("  %4d   %.8e   %.8e   %.7f   %.3e\n",
                    N, r.d0_quad, r.d0_stiff, r.keT, ke_err);

        // (G0) stiff-as-called-by-rhs == independent quadrature, to round-off
        if (std::fabs(r.d0_stiff - r.d0_quad) > 1e-10 * std::fabs(r.d0_quad)) {
            std::printf("   ** G0 FAIL: stiff() (out=t1) disagrees with quadrature "
                        "(rel %.2e) -- scratch aliasing **\n",
                        std::fabs(r.d0_stiff - r.d0_quad) / std::fabs(r.d0_quad));
            fail = 1;
        }
        // (G1) projected-IC dissipation at the finest N
        if (c + 1 == ladder.size() &&
            std::fabs(r.d0_quad - REF_DISS) > 1e-3 * REF_DISS) {
            std::printf("   ** G1 FAIL: initial dissipation %.6e vs %.6e **\n", r.d0_quad, REF_DISS);
            fail = 1;
        }
        // (G2) KE(T) accuracy + error must not grow under refinement
        if (have_prev && ke_err > 0) {
            last_order = std::log(prev_err / ke_err) /
                         std::log((double)N / (double)ladder[c - 1]);
            if (ke_err > prev_err * 1.05) {
                std::printf("   ** G2 FAIL: KE(T) error grew under refinement "
                            "(%.3e -> %.3e) **\n", prev_err, ke_err);
                fail = 1;
            }
        }
        if (c + 1 == ladder.size() && ke_err > 5e-4) {
            std::printf("   ** G2 FAIL: |KE(T) - %.7f| = %.3e > 5e-4 **\n", REF_KE_T, ke_err);
            fail = 1;
        }
        prev_err = ke_err; have_prev = true;
    }
    if (ladder.size() > 1)
        std::printf("\n  KE(T) error vs BMF reference, order (last pair): %.2f "
                    "(splitting floor saturates this; informational)\n", last_order);
    std::printf("\n %s\n\n", fail ? "** TG DECAY GATE: FAIL **" : "TG decay gate: PASS");
    return fail;
}
