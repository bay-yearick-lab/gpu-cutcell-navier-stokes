// wall_ns2d.h
//
// PRIMITIVE-variable incompressible Navier-Stokes in a 2D WALLED box, in
// coefficient space -- the model for the 3D wall-NS loop, assembling the two
// de-risked pieces in one (compatible) representation:
//
//   * the energy-conserving GALERKIN advection (the burgers1d/wall2d kernel,
//     here the skew-symmetric vector form 1/2[u.grad u_i + div(u u_i)]),
//   * the inf-sup-stable consistent GALERKIN projection (mms_infsup): the Schur
//     complement L = D Mv^{-1} D^T (NOT the collocated sum G_a^2), which makes
//     the WEAK divergence vanish EXACTLY (idempotent) and is inf-sup stable for
//     the smooth equal-order B-spline pairing.
//
// No-slip walls (u = w = 0 on the box boundary). Velocity u, w and pressure phi
// are equal-order degree-p tensor-product spline coefficients. div u = 0 in the
// weak sense; KE is conserved by the advection (inviscid) to the div-free error.

#ifndef WALL_NS2D_H
#define WALL_NS2D_H

#include <vector>
#include <cmath>
#include "wall2d.h"

struct WallNS2D {
    Axis X, Z;
    double nu;
    bool advect = true;
    Field2D u, w, u0, w0, ku, kw, phi, dvg, t1, t2;
    Field2D ts1, ts2;   // stiff()-private scratch: callers may pass t1/t2 as out

    WallNS2D(int nx, int nz, int p, double Lx, double Lz, double nu_)
        : X(nx, p, Lx), Z(nz, p, Lz), nu(nu_),
          u(nx, nz), w(nx, nz), u0(nx, nz), w0(nx, nz), ku(nx, nz), kw(nx, nz),
          phi(nx, nz), dvg(nx, nz), t1(nx, nz), t2(nx, nz),
          ts1(nx, nz), ts2(nx, nz) {}

    bool is_bndry(int i, int j) const { return i == 0 || i == X.n - 1 || j == 0 || j == Z.n - 1; }
    void no_slip(Field2D& f) const {
        for (int i = 0; i < X.n; ++i) for (int j = 0; j < Z.n; ++j) if (is_bndry(i, j)) f.at(i, j) = 0.0;
    }

    // weak divergence  d = (Kx (x) Mz) u + (Mx (x) Kz) w
    void weak_div(const Field2D& uu, const Field2D& ww, Field2D& d) {
        apply_z(Z.M, uu, t1); apply_x(X.K, t1, d);
        apply_z(Z.K, ww, t1); apply_x(X.M, t1, t2);
        for (size_t q = 0; q < d.a.size(); ++q) d.a[q] += t2.a[q];
    }
    // Schur Laplacian  L phi = (Kx Mx^-1 Kx^T (x) Mz) phi + (Mx (x) Kz Mz^-1 Kz^T) phi
    void schur(const Field2D& f, Field2D& out) {
        apply_z(Z.M, f, t1); apply_xT(X.K, t1, t2); minv_x(X, t2); apply_x(X.K, t2, out);
        apply_x(X.M, f, t1); apply_zT(Z.K, t1, t2); minv_z(Z, t2); apply_z(Z.K, t2, t1);
        for (size_t q = 0; q < out.a.size(); ++q) out.a[q] += t1.a[q];
    }
    // CG solve  L phi = rhs  (symmetric PSD)
    //
    // NOTE: two details of the CG solve matter here.
    // (1) Stopping test: the original `rr > 1e-22*rr0` (1e-11 relative on the
    //     SQUARED residual) sits below the round-off noise floor of the dense
    //     schur() apply, so the loop never satisfied its own exit test and
    //     silently burned the full 6*N^2 iteration cap on every project().
    //     Replaced by a reachable relative tolerance (1e-12 on ||r||) plus a
    //     stagnation safeguard; accuracy unchanged.
    // (2) Null-space handling: the original remove_mean(r) / remove_mean(Ap)
    //     subtracted the ALL-ONES component each iteration. All-ones spans the
    //     kernel of the PERIODIC Schur operator, but the CLAMPED kernel is
    //     nu_x (x) nu_z (per-axis L2 complements of the derivative space), so
    //     on a walled box the mean-removal perturbed the operator inside the
    //     recurrence: the recursive residual reported 1e-12 while the TRUE
    //     residual stalled near 6e-2 relative, i.e. CG confidently returned a
    //     wrong pressure (O(1e-3) velocity error per step). The weak-divergence
    //     rhs is exactly consistent (K^T nu = 0 kills its kernel component)
    //     and CG from phi = 0 stays in range(L); any round-off kernel drift in
    //     phi is pure gauge (the K^T gradient annihilates it). So: no mean
    //     removal at all. Verified against the fast-diagonalization direct
    //     solver: velocities agree to 8e-14.
    void solve_schur(const Field2D& rhs) {
        for (double& v : phi.a) v = 0.0;
        Field2D r = rhs, pdir(X.n, Z.n), Ap(X.n, Z.n);
        pdir = r;
        double rr = 0; for (double v : r.a) rr += v * v;
        const double rr0 = rr;
        const double tol = std::fmax(1e-24 * rr0, 1e-28);
        double rr_best = rr; int stall = 0;
        const int maxit = 6 * X.n * Z.n;
        for (int it = 0; it < maxit && rr > tol; ++it) {
            schur(pdir, Ap);
            double pAp = 0; for (size_t q = 0; q < Ap.a.size(); ++q) pAp += pdir.a[q] * Ap.a[q];
            if (pAp <= 0) break;
            const double al = rr / pAp;
            double rr2 = 0;
            for (size_t q = 0; q < r.a.size(); ++q) { phi.a[q] += al * pdir.a[q]; r.a[q] -= al * Ap.a[q]; rr2 += r.a[q] * r.a[q]; }
            const double be = rr2 / rr; rr = rr2;
            for (size_t q = 0; q < r.a.size(); ++q) pdir.a[q] = r.a[q] + be * pdir.a[q];
            if (rr < rr_best * (1.0 - 1e-6)) { rr_best = rr; stall = 0; } else if (++stall > 2 * (X.n + Z.n)) break;
        }
    }
    // project (u,w) weak-divergence-free: solve L phi = div u*; u -= grad phi.
    // pin=true re-enforces no-slip (a fractional-step wall-splitting error, small
    // & bounded, re-introduces a little boundary divergence); pin=false leaves the
    // exactly weak-div-free field (used to show the projection is idempotent).
    void project(bool pin = true) {
        weak_div(u, w, dvg);
        solve_schur(dvg);
        apply_xT(X.K, phi, t1); minv_x(X, t1);          // (Mx^-1 Kx^T (x) I) phi
        for (size_t q = 0; q < u.a.size(); ++q) u.a[q] -= t1.a[q];
        apply_zT(Z.K, phi, t1); minv_z(Z, t1);          // (I (x) Mz^-1 Kz^T) phi
        for (size_t q = 0; q < w.a.size(); ++q) w.a[q] -= t1.a[q];
        if (pin) { no_slip(u); no_slip(w); }
    }
    double max_weak_div() {
        weak_div(u, w, dvg);
        double m = 0; for (double v : dvg.a) m = std::fmax(m, std::fabs(v)); return m;
    }

    // energy-conserving skew advection: returns Cu = <N, 1/2[u.grad u + div(u u)]>,
    // Cw likewise for w. Assembled by tensor-product Gauss quadrature.
    void advect_skew(Field2D& Cu, Field2D& Cw) {
        const int px = X.p, pz = Z.p;
        std::vector<double> gx, gw; gauss_legendre(2 * std::max(px, pz), gx, gw);
        std::vector<std::vector<double>> dx, dz;
        for (double& c : Cu.a) c = 0.0;
        for (double& c : Cw.a) c = 0.0;
        for (int sx = px; sx < X.n; ++sx) {
            const double ax = X.knots[sx], bx = X.knots[sx + 1]; if (bx <= ax) continue;
            const double hx = 0.5 * (bx - ax), mx = 0.5 * (ax + bx);
            for (int sz = pz; sz < Z.n; ++sz) {
                const double az = Z.knots[sz], bz = Z.knots[sz + 1]; if (bz <= az) continue;
                const double hz = 0.5 * (bz - az), mz = 0.5 * (az + bz);
                for (size_t qx = 0; qx < gx.size(); ++qx) {
                    ders_basis_funs(sx, mx + hx * gx[qx], px, 1, X.knots, dx);
                    for (size_t qz = 0; qz < gx.size(); ++qz) {
                        ders_basis_funs(sz, mz + hz * gx[qz], pz, 1, Z.knots, dz);
                        const double wq = gw[qx] * gw[qz] * hx * hz;
                        double U = 0, W = 0, Ux = 0, Uz = 0, Wx = 0, Wz = 0;
                        for (int a = 0; a <= px; ++a) {
                            const int gi = sx - px + a;
                            for (int b = 0; b <= pz; ++b) {
                                const int gj = sz - pz + b;
                                const double uu = u.at(gi, gj), ww = w.at(gi, gj);
                                const double Nx = dx[0][a], dNx = dx[1][a], Nz = dz[0][b], dNz = dz[1][b];
                                U += uu * Nx * Nz; W += ww * Nx * Nz;
                                Ux += uu * dNx * Nz; Uz += uu * Nx * dNz;
                                Wx += ww * dNx * Nz; Wz += ww * Nx * dNz;
                            }
                        }
                        const double adU = U * Ux + W * Uz, adW = U * Wx + W * Wz;  // u.grad u_i
                        const double fUx = U * U, fUz = W * U, fWx = U * W, fWz = W * W; // flux u u_i
                        for (int a = 0; a <= px; ++a) {
                            const int gi = sx - px + a;
                            for (int b = 0; b <= pz; ++b) {
                                const int gj = sz - pz + b;
                                const double Na = dx[0][a] * dz[0][b];
                                const double dNax = dx[1][a] * dz[0][b], dNaz = dx[0][a] * dz[1][b];
                                Cu.at(gi, gj) += wq * 0.5 * (Na * adU - dNax * fUx - dNaz * fUz);
                                Cw.at(gi, gj) += wq * 0.5 * (Na * adW - dNax * fWx - dNaz * fWz);
                            }
                        }
                    }
                }
            }
        }
    }

    // viscous weak Laplacian apply (Sx(x)Mz + Mx(x)Sz) -> stiffness*field.
    // Uses its OWN scratch (ts1, ts2): every call site passes the shared t1
    // as out, and the previous use of t1/t2 as internal scratch silently
    // replaced the (Sx(x)Mz) term by an unweighted (Mx(x)I) mass term,
    // dropping all x-diffusion and injecting an O(nu*N) artificial damping
    // (found 2026-07-13 via the Taylor-Green no-slip decay ladder).
    void stiff(const Field2D& f, Field2D& out) {
        apply_z(Z.M, f, ts1); apply_x(X.S, ts1, ts2);   // (Sx (x) Mz) f
        apply_x(X.M, f, ts1); apply_z(Z.S, ts1, out);   // (Mx (x) Sz) f
        for (size_t q = 0; q < out.a.size(); ++q) out.a[q] += ts2.a[q];
    }
    // d/dt (u,w) = M^-1( -C_adv - nu*Stiff )   then project.  (M = Mx(x)Mz)
    void rhs(Field2D& du, Field2D& dw) {
        if (advect) advect_skew(du, dw); else { for (double& v : du.a) v = 0; for (double& v : dw.a) v = 0; }
        stiff(u, t1); for (size_t q = 0; q < du.a.size(); ++q) du.a[q] = -du.a[q] - nu * t1.a[q];
        stiff(w, t1); for (size_t q = 0; q < dw.a.size(); ++q) dw.a[q] = -dw.a[q] - nu * t1.a[q];
        // constrained-Galerkin rates: interior-restricted mass solve. The
        // previous full-space solve followed by wall pinning contaminated the
        // first ~p interior coefficients through the wall-row mass coupling,
        // an O(h) spatial rate defect (2026-07-13 Taylor-Green forensics).
        mass_solve2d_interior(X, Z, du); mass_solve2d_interior(X, Z, dw);
        no_slip(du); no_slip(dw);   // idempotent now
    }
    void step(double dt) {                  // SSP-RK3 with projection each stage
        u0 = u; w0 = w;
        rhs(ku, kw);
        for (size_t q = 0; q < u.a.size(); ++q) { u.a[q] = u0.a[q] + dt * ku.a[q]; w.a[q] = w0.a[q] + dt * kw.a[q]; }
        no_slip(u); no_slip(w); project();
        rhs(ku, kw);
        for (size_t q = 0; q < u.a.size(); ++q) { u.a[q] = 0.75 * u0.a[q] + 0.25 * (u.a[q] + dt * ku.a[q]); w.a[q] = 0.75 * w0.a[q] + 0.25 * (w.a[q] + dt * kw.a[q]); }
        no_slip(u); no_slip(w); project();
        rhs(ku, kw);
        for (size_t q = 0; q < u.a.size(); ++q) { u.a[q] = (1.0/3.0) * u0.a[q] + (2.0/3.0) * (u.a[q] + dt * ku.a[q]); w.a[q] = (1.0/3.0) * w0.a[q] + (2.0/3.0) * (w.a[q] + dt * kw.a[q]); }
        no_slip(u); no_slip(w); project();
    }
    double kinetic_energy() {                // 1/2 (u^T M u + w^T M w),  M = Mx(x)Mz
        apply_z(Z.M, u, t1); apply_x(X.M, t1, t2);
        double e = 0; for (size_t q = 0; q < u.a.size(); ++q) e += u.a[q] * t2.a[q];
        apply_z(Z.M, w, t1); apply_x(X.M, t1, t2);
        for (size_t q = 0; q < w.a.size(); ++q) e += w.a[q] * t2.a[q];
        return 0.5 * e;
    }
};

#endif  // WALL_NS2D_H
