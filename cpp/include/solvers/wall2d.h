// wall2d.h
//
// A 2D WALLED incompressible flow in coefficient space, built on the energy-
// conserving wall advection (the burgers1d.h kernel, extended to a plane). The
// vorticity-streamfunction form on a clamped(x) x clamped(z) box with free-slip
// walls (omega = 0, psi = 0 on the walls):
//
//   omega_t + u . grad omega = nu lap omega,   lap psi = -omega,   u = (psi_z, -psi_x).
//
// Why this is the wall-NS frontier resolved: the advection is the full 2D
// isogeometric GALERKIN form (Gauss-exact quadrature), and u = curl(psi_h) of a
// spline streamfunction makes div u_h = 0 EXACTLY. So the discrete enstrophy
// 1/2 \int omega^2 is conserved to round-off by the advection (the interior
// injects nothing; omega = u.n = 0 walls kill the boundary flux), independent of
// the operators' non-diagonal energy norm -- exactly what the nodal/collocated
// wall could not do. The Poisson lap psi = -omega is the well-conditioned
// Galerkin stiffness solved by CG. This is the 2D model of the 3D wall-NS loop.

#ifndef WALL2D_H
#define WALL2D_H

#include <vector>
#include <cmath>
#include "bspline.h"
#include "linalg.h"

// 2D clamped x clamped coefficient field, idx = ix*nz + iz.
struct Field2D {
    int nx, nz;
    std::vector<double> a;
    Field2D() : nx(0), nz(0) {}
    Field2D(int nx_, int nz_) : nx(nx_), nz(nz_), a((size_t)nx_ * nz_, 0.0) {}
    double& at(int i, int j) { return a[(size_t)i * nz + j]; }
    double  at(int i, int j) const { return a[(size_t)i * nz + j]; }
};

// One axis: CLAMPED (wall, default) or PERIODIC. Sample matrix B, mass M, weak
// derivative K = int N_i N_j', stiffness S; clamped via Greville/open knots, the
// periodic (channel) axis via the circulant cardinal-B-spline forms. Mfac is the
// banded M^{-1} (clamped); Minv is the dense M^{-1} (periodic, circulant M).
struct Axis {
    int n, p; double L;
    bool per = false;
    std::vector<double> knots, grev;
    DenseMatrix B, Minv;
    Mat M, K, S;
    BandSolver Mfac;
    BandSolver Mfac_i;   // interior-restricted mass LU: M[1:n-2, 1:n-2]
    Axis() : n(0), p(0), L(0) {}
    Axis(int n_, int p_, double L_, bool periodic = false)
        : n(n_), p(p_), L(L_), per(periodic), B(n_) {
        if (per) { build_periodic(); return; }
        knots = open_uniform_knots(n, p, L);
        grev = greville_points(knots, p, n);
        std::vector<std::vector<double>> d;
        for (int r = 0; r < n; ++r) {
            int s = find_span(n - 1, p, grev[r], knots);
            ders_basis_funs(s, grev[r], p, 0, knots, d);
            for (int a = 0; a <= p; ++a) B(r, s - p + a) = d[0][a];
        }
        CoeffForms cf = assemble_forms_clamped(n, p, knots);
        M = cf.M; K = cf.K; S = cf.S;
        DenseMatrix Md(n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) Md(i, j) = M(i, j);
        Mfac.set_from_dense(Md, p, p); Mfac.factor();
        // interior-restricted mass LU for constrained-Galerkin stage rates:
        // the full-space inverse followed by wall-row zeroing contaminates the
        // first ~p interior coefficients through the wall-row mass coupling
        // (an O(h) rate defect found 2026-07-13 via the Taylor-Green ladder)
        if (n > 2) {
            DenseMatrix Mi(n - 2);
            for (int i = 1; i < n - 1; ++i)
                for (int j = 1; j < n - 1; ++j) Mi(i - 1, j - 1) = M(i, j);
            Mfac_i.set_from_dense(Mi, p, p); Mfac_i.factor();
        }
    }
    // wrap an integer offset into [-n/2, n/2) for the circulant evaluation
    int wrap(int d) const { d %= n; if (d > n / 2) d -= n; if (d < -n / 2) d += n; return d; }
    void build_periodic() {
        const double h = L / n;
        grev.resize(n); for (int i = 0; i < n; ++i) grev[i] = i * h;
        CoeffForms cf = assemble_forms_periodic(n, p, h);   // circulant M, K, S
        M = cf.M; K = cf.K; S = cf.S;
        // periodic sample matrix B(i,j) = N_j(x_i) = beta_p(i-j) (circulant)
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            double v = beta_value(p, (double)wrap(i - j));
            if (std::fabs(v) > 1e-15) B(i, j) = v;
        }
        // dense inverse of the circulant M (n small): solve M X = I
        Minv = DenseMatrix(n);
        DenseMatrix Md(n); for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) Md(i, j) = M(i, j);
        std::vector<double> e(n), x;
        for (int c = 0; c < n; ++c) { std::fill(e.begin(), e.end(), 0.0); e[c] = 1.0;
            dense_lu_solve(Md, e, x); for (int i = 0; i < n; ++i) Minv(i, c) = x[i]; }
    }
    std::vector<double> interp(const std::vector<double>& fn) const {
        std::vector<double> c; dense_lu_solve(B, fn, c); return c;
    }
    // solve M x = line in place (banded for clamped, dense circulant inverse for periodic)
    void msolve(std::vector<double>& line) const {
        if (!per) { Mfac.solve_inplace(line); return; }
        std::vector<double> out(n);
        for (int i = 0; i < n; ++i) { double s = 0; for (int j = 0; j < n; ++j) s += Minv(i, j) * line[j]; out[i] = s; }
        line.swap(out);
    }
};

// 2D Galerkin advection  C_ij = \int N_iN_j (u_x phi_x + u_z phi_z),  u = curl(psi).
inline void advect2d(const Axis& X, const Axis& Z, const Field2D& psi,
                     const Field2D& phi, Field2D& C) {
    const int px = X.p, pz = Z.p;
    std::vector<double> gx, gw; gauss_legendre(2 * std::max(px, pz), gx, gw);
    std::vector<std::vector<double>> dx, dz;
    for (double& c : C.a) c = 0.0;
    for (int sx = px; sx < X.n; ++sx) {
        const double ax = X.knots[sx], bx = X.knots[sx + 1]; if (bx <= ax) continue;
        const double hx = 0.5 * (bx - ax), mx = 0.5 * (ax + bx);
        for (int sz = pz; sz < Z.n; ++sz) {
            const double az = Z.knots[sz], bz = Z.knots[sz + 1]; if (bz <= az) continue;
            const double hz = 0.5 * (bz - az), mz = 0.5 * (az + bz);
            for (size_t qx = 0; qx < gx.size(); ++qx) {
                const double xq = mx + hx * gx[qx];
                ders_basis_funs(sx, xq, px, 1, X.knots, dx);
                for (size_t qz = 0; qz < gx.size(); ++qz) {
                    const double zq = mz + hz * gx[qz];
                    ders_basis_funs(sz, zq, pz, 1, Z.knots, dz);
                    const double wq = gw[qx] * gw[qz] * hx * hz;
                    double ux = 0, uz = 0, phx = 0, phz = 0;
                    for (int a = 0; a <= px; ++a) {
                        const int gi = sx - px + a;
                        for (int b = 0; b <= pz; ++b) {
                            const int gj = sz - pz + b;
                            const double ps = psi.at(gi, gj), ph = phi.at(gi, gj);
                            ux += ps * dx[0][a] * dz[1][b];      // psi_z
                            uz -= ps * dx[1][a] * dz[0][b];      // -psi_x
                            phx += ph * dx[1][a] * dz[0][b];     // phi_x
                            phz += ph * dx[0][a] * dz[1][b];     // phi_z
                        }
                    }
                    const double adv = ux * phx + uz * phz;
                    for (int a = 0; a <= px; ++a) {
                        const int gi = sx - px + a;
                        const double Na = dx[0][a];
                        for (int b = 0; b <= pz; ++b)
                            C.at(gi, sz - pz + b) += wq * Na * dz[0][b] * adv;
                    }
                }
            }
        }
    }
}

// separable mass solve  f <- (Mx (x) Mz)^{-1} f
inline void mass_solve2d(const Axis& X, const Axis& Z, Field2D& f) {
    std::vector<double> line;
    for (int j = 0; j < Z.n; ++j) {
        line.resize(X.n);
        for (int i = 0; i < X.n; ++i) line[i] = f.at(i, j);
        X.Mfac.solve_inplace(line);
        for (int i = 0; i < X.n; ++i) f.at(i, j) = line[i];
    }
    for (int i = 0; i < X.n; ++i) {
        line.resize(Z.n);
        for (int j = 0; j < Z.n; ++j) line[j] = f.at(i, j);
        Z.Mfac.solve_inplace(line);
        for (int j = 0; j < Z.n; ++j) f.at(i, j) = line[j];
    }
}

// Interior-restricted mass solve for constrained-Galerkin stage rates:
// solves M_II r_I = load_I on the interior tensor block (interior of a
// tensor grid is a product set, so M_II = Mx[1:-1,1:-1] (x) Mz[1:-1,1:-1])
// and zeroes the wall frame. Replaces "full-space solve then pin", whose
// wall-row mass coupling injects an O(h) near-wall rate defect.
inline void mass_solve2d_interior(const Axis& X, const Axis& Z, Field2D& f) {
    const int nx = X.n, nz = Z.n;
    std::vector<double> line(std::max(nx, nz));
    for (int j = 1; j < nz - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) line[i - 1] = f.at(i, j);
        X.Mfac_i.solve_inplace(line);
        for (int i = 1; i < nx - 1; ++i) f.at(i, j) = line[i - 1];
    }
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < nz - 1; ++j) line[j - 1] = f.at(i, j);
        Z.Mfac_i.solve_inplace(line);
        for (int j = 1; j < nz - 1; ++j) f.at(i, j) = line[j - 1];
    }
    // wall frame rates are owned by the (time-independent) trace data
    for (int i = 0; i < nx; ++i) { f.at(i, 0) = 0.0; f.at(i, nz - 1) = 0.0; }
    for (int j = 0; j < nz; ++j) { f.at(0, j) = 0.0; f.at(nx - 1, j) = 0.0; }
}

inline double enstrophy2d(const Axis& X, const Axis& Z, const Field2D& w) {
    double e = 0.0;
    for (int i = 0; i < X.n; ++i) for (int j = 0; j < Z.n; ++j) {
        double Mw = 0.0;
        for (int k = 0; k < X.n; ++k) for (int l = 0; l < Z.n; ++l)
            Mw += X.M(i, k) * Z.M(j, l) * w.at(k, l);
        e += w.at(i, j) * Mw;
    }
    return 0.5 * e;
}

// apply a 1D Mat along x (axis 0) / z (axis 1) of a 2D field
inline void apply_x(const Mat& A, const Field2D& in, Field2D& out) {
    for (int j = 0; j < in.nz; ++j) for (int i = 0; i < in.nx; ++i) {
        double s = 0; for (int k = 0; k < in.nx; ++k) s += A(i, k) * in.at(k, j);
        out.at(i, j) = s;
    }
}
inline void apply_z(const Mat& A, const Field2D& in, Field2D& out) {
    for (int i = 0; i < in.nx; ++i) for (int j = 0; j < in.nz; ++j) {
        double s = 0; for (int l = 0; l < in.nz; ++l) s += A(j, l) * in.at(i, l);
        out.at(i, j) = s;
    }
}
// transpose applies (use A(k,i) instead of A(i,k))
inline void apply_xT(const Mat& A, const Field2D& in, Field2D& out) {
    for (int j = 0; j < in.nz; ++j) for (int i = 0; i < in.nx; ++i) {
        double s = 0; for (int k = 0; k < in.nx; ++k) s += A(k, i) * in.at(k, j);
        out.at(i, j) = s;
    }
}
inline void apply_zT(const Mat& A, const Field2D& in, Field2D& out) {
    for (int i = 0; i < in.nx; ++i) for (int j = 0; j < in.nz; ++j) {
        double s = 0; for (int l = 0; l < in.nz; ++l) s += A(l, j) * in.at(i, l);
        out.at(i, j) = s;
    }
}
// apply M^{-1} along an axis (per line band solve)
inline void minv_x(const Axis& X, Field2D& f) {
    std::vector<double> line(X.n);
    for (int j = 0; j < f.nz; ++j) {
        for (int i = 0; i < X.n; ++i) line[i] = f.at(i, j);
        X.Mfac.solve_inplace(line);
        for (int i = 0; i < X.n; ++i) f.at(i, j) = line[i];
    }
}
inline void minv_z(const Axis& Z, Field2D& f) {
    std::vector<double> line(Z.n);
    for (int i = 0; i < f.nx; ++i) {
        for (int j = 0; j < Z.n; ++j) line[j] = f.at(i, j);
        Z.Mfac.solve_inplace(line);
        for (int j = 0; j < Z.n; ++j) f.at(i, j) = line[j];
    }
}

// The Galerkin Laplacian stiffness K2D = Sx (x) Mz + Mx (x) Sz, applied to a field.
struct WallVortex2D {
    Axis X, Z;
    double nu;
    Field2D omega, psi, C, t1, t2, w0, kk;
    Field2D ts1, ts2;   // stiff()-private scratch: callers pass t1/t2 as out
    WallVortex2D(int nx, int nz, int p, double Lx, double Lz, double nu_)
        : X(nx, p, Lx), Z(nz, p, Lz), nu(nu_),
          omega(nx, nz), psi(nx, nz), C(nx, nz), t1(nx, nz), t2(nx, nz),
          w0(nx, nz), kk(nx, nz), ts1(nx, nz), ts2(nx, nz) {}

    bool is_bndry(int i, int j) const {
        return i == 0 || i == X.n - 1 || j == 0 || j == Z.n - 1;
    }
    void zero_bndry(Field2D& f) const {
        for (int i = 0; i < X.n; ++i) for (int j = 0; j < Z.n; ++j)
            if (is_bndry(i, j)) f.at(i, j) = 0.0;
    }
    // K2D f -> out  (stiffness; Sx Mz + Mx Sz).
    // Own scratch (ts1, ts2): call sites pass t1/t2 as out, and aliasing the
    // shared scratch corrupted the assembled operator (same defect class as
    // WallNS2D::stiff, found 2026-07-13 via the Taylor-Green decay ladder).
    void stiff(const Field2D& f, Field2D& out) {
        apply_z(Z.M, f, ts1); apply_x(X.S, ts1, out);        // Sx (Mz f)
        apply_z(Z.S, f, ts1); apply_x(X.M, ts1, ts2);        // Mx (Sz f)
        for (size_t q = 0; q < out.a.size(); ++q) out.a[q] += ts2.a[q];
    }
    void mass2d(const Field2D& f, Field2D& out) {
        apply_z(Z.M, f, t1); apply_x(X.M, t1, out);
    }

    // Solve K2D psi = M2D omega with psi = 0 on the walls (CG on the interior).
    void poisson() {
        Field2D b(X.n, Z.n); mass2d(omega, b); zero_bndry(b);
        for (double& v : psi.a) v = 0.0;
        Field2D r = b, pdir = r, Ap(X.n, Z.n);
        double rr = 0; for (size_t q = 0; q < r.a.size(); ++q) rr += r.a[q] * r.a[q];
        const double tol = 1e-20 * rr + 1e-30;
        for (int it = 0; it < 4 * X.n * Z.n && rr > tol; ++it) {
            stiff(pdir, Ap); zero_bndry(Ap);
            double pAp = 0; for (size_t q = 0; q < Ap.a.size(); ++q) pAp += pdir.a[q] * Ap.a[q];
            const double alpha = rr / pAp;
            double rr_new = 0;
            for (size_t q = 0; q < r.a.size(); ++q) {
                psi.a[q] += alpha * pdir.a[q];
                r.a[q] -= alpha * Ap.a[q];
                rr_new += r.a[q] * r.a[q];
            }
            const double beta = rr_new / rr; rr = rr_new;
            for (size_t q = 0; q < r.a.size(); ++q) pdir.a[q] = r.a[q] + beta * pdir.a[q];
        }
    }

    // d(omega)/dt = M2D^{-1}( -C(omega) - nu K2D omega ),  free-slip omega=0 walls.
    void rhs(const Field2D& w, Field2D& dw) {
        omega = w; zero_bndry(omega);
        poisson();
        advect2d(X, Z, psi, omega, C);
        if (nu != 0.0) { stiff(omega, t2); }
        for (size_t q = 0; q < dw.a.size(); ++q)
            dw.a[q] = -C.a[q] - (nu != 0.0 ? nu * t2.a[q] : 0.0);
        mass_solve2d(X, Z, dw);
        zero_bndry(dw);
    }
    void step(double dt) {
        w0 = omega;
        rhs(omega, kk); for (size_t q = 0; q < omega.a.size(); ++q) omega.a[q] = w0.a[q] + dt * kk.a[q]; zero_bndry(omega);
        rhs(omega, kk); for (size_t q = 0; q < omega.a.size(); ++q) omega.a[q] = 0.75 * w0.a[q] + 0.25 * (omega.a[q] + dt * kk.a[q]); zero_bndry(omega);
        rhs(omega, kk); for (size_t q = 0; q < omega.a.size(); ++q) omega.a[q] = (1.0/3.0) * w0.a[q] + (2.0/3.0) * (omega.a[q] + dt * kk.a[q]); zero_bndry(omega);
    }
    double enstrophy() { return enstrophy2d(X, Z, omega); }
    double kinetic_energy() { poisson(); return 0.5 * dot_grad_psi(); }   // 1/2 \int |grad psi|^2
    double dot_grad_psi() {
        // \int |u|^2 = \int |grad psi|^2 = psi^T K2D psi
        stiff(psi, t1);
        double s = 0; for (size_t q = 0; q < psi.a.size(); ++q) s += psi.a[q] * t1.a[q];
        return s;
    }
};

#endif  // WALL2D_H
