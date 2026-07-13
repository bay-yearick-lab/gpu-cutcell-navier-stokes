// wall_ns3d.h
//
// PRIMITIVE-variable incompressible Navier-Stokes in a 3D WALLED box, in
// coefficient space -- the 3D wall-NS loop, the direct extension of WallNS2D.
// Velocity (u,v,w) and pressure phi are equal-order degree-p tensor-product
// spline coefficients on a clamped x clamped x clamped box; no-slip walls.
//
// Couples the two de-risked + composed pieces:
//   * energy-conserving skew GALERKIN advection 1/2[u.grad u_i + div(u u_i)]
//     (tensor-product Gauss quadrature; conserves KE inviscidly to the div-free
//     error),
//   * the consistent inf-sup-stable SCHUR-complement projection
//     L = D Mv^{-1} D^T  (idempotent => weak div u = 0; equal-order, no
//     checkerboard for the smooth B-spline pairing).
//
// SPEED NOTE: the advection assembly is the plain (p+1)^3 Gauss loop -- correct
// but O(N^3 * (2p)^3 * (p+1)^3); production would use sum-factorization. Built
// for correctness of the 3D wall loop; verify at modest N.

#ifndef WALL_NS3D_H
#define WALL_NS3D_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "wall2d.h"      // Axis, bspline + linalg
#include "fastdiag3.h"   // fast-diagonalization direct solver

// 3D coefficient field, idx = (i*ny + j)*nz + k.
struct Field3 {
    int nx, ny, nz;
    std::vector<double> a;
    Field3() : nx(0), ny(0), nz(0) {}
    Field3(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_),
        a((size_t)nx_ * ny_ * nz_, 0.0) {}
    size_t idx(int i, int j, int k) const { return ((size_t)i * ny + j) * nz + k; }
    double& at(int i, int j, int k) { return a[idx(i, j, k)]; }
    double  at(int i, int j, int k) const { return a[idx(i, j, k)]; }
};

// apply 1D operator A (transpose tr) along `axis` of a Field3. Templated so the
// dense eigenvector matrices (fast diagonalization) work alongside the Mat forms.
template <class MAT>
inline void line_apply3(int axis, const MAT& A, bool tr, const Field3& in, Field3& out) {
    const int nx = in.nx, ny = in.ny, nz = in.nz;
    const int n = (axis == 0) ? nx : (axis == 1) ? ny : nz;
    std::vector<double> li(n), lo(n);
    auto domul = [&](void) {
        for (int i = 0; i < n; ++i) {
            double s = 0;
            if (tr) for (int l = 0; l < n; ++l) s += A(l, i) * li[l];
            else    for (int l = 0; l < n; ++l) s += A(i, l) * li[l];
            lo[i] = s;
        }
    };
    if (axis == 0) {
        for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) li[i] = in.at(i, j, k);
            domul();
            for (int i = 0; i < nx; ++i) out.at(i, j, k) = lo[i];
        }
    } else if (axis == 1) {
        for (int i = 0; i < nx; ++i) for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) li[j] = in.at(i, j, k);
            domul();
            for (int j = 0; j < ny; ++j) out.at(i, j, k) = lo[j];
        }
    } else {
        for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) li[k] = in.at(i, j, k);
            domul();
            for (int k = 0; k < nz; ++k) out.at(i, j, k) = lo[k];
        }
    }
}
// Interior-restricted mass solve along a CLAMPED `axis`: solves the interior
// block M[1:n-2, 1:n-2] per line (constrained-Galerkin stage rates; the
// full-space solve followed by wall pinning injects an O(h) near-wall rate
// defect through the wall-row mass coupling, 2026-07-13 Taylor-Green
// forensics). Periodic axes have no walls and fall back to the full solve.
// Wall-slab lines are left untouched; the caller zeroes the wall frame.
inline void minv3_interior(int axis, const Axis& A, Field3& f) {
    const int n = (axis == 0) ? f.nx : (axis == 1) ? f.ny : f.nz;
    std::vector<double> line(n);
    if (axis == 0) {
        for (int j = 0; j < f.ny; ++j) for (int k = 0; k < f.nz; ++k) {
            if (A.per) {
                for (int i = 0; i < n; ++i) line[i] = f.at(i, j, k);
                A.msolve(line);
                for (int i = 0; i < n; ++i) f.at(i, j, k) = line[i];
            } else {
                for (int i = 1; i < n - 1; ++i) line[i - 1] = f.at(i, j, k);
                A.Mfac_i.solve_inplace(line);
                for (int i = 1; i < n - 1; ++i) f.at(i, j, k) = line[i - 1];
            }
        }
    } else if (axis == 1) {
        for (int i = 0; i < f.nx; ++i) for (int k = 0; k < f.nz; ++k) {
            if (A.per) {
                for (int j = 0; j < n; ++j) line[j] = f.at(i, j, k);
                A.msolve(line);
                for (int j = 0; j < n; ++j) f.at(i, j, k) = line[j];
            } else {
                for (int j = 1; j < n - 1; ++j) line[j - 1] = f.at(i, j, k);
                A.Mfac_i.solve_inplace(line);
                for (int j = 1; j < n - 1; ++j) f.at(i, j, k) = line[j - 1];
            }
        }
    } else {
        for (int i = 0; i < f.nx; ++i) for (int j = 0; j < f.ny; ++j) {
            if (A.per) {
                for (int k = 0; k < n; ++k) line[k] = f.at(i, j, k);
                A.msolve(line);
                for (int k = 0; k < n; ++k) f.at(i, j, k) = line[k];
            } else {
                for (int k = 1; k < n - 1; ++k) line[k - 1] = f.at(i, j, k);
                A.Mfac_i.solve_inplace(line);
                for (int k = 1; k < n - 1; ++k) f.at(i, j, k) = line[k - 1];
            }
        }
    }
}

// solve M along `axis` in place (per line band solve)
inline void minv3(int axis, const Axis& Ax, Field3& f) {
    const int n = (axis == 0) ? f.nx : (axis == 1) ? f.ny : f.nz;
    std::vector<double> line(n);
    if (axis == 0) {
        for (int j = 0; j < f.ny; ++j) for (int k = 0; k < f.nz; ++k) {
            for (int i = 0; i < n; ++i) line[i] = f.at(i, j, k);
            Ax.msolve(line);
            for (int i = 0; i < n; ++i) f.at(i, j, k) = line[i];
        }
    } else if (axis == 1) {
        for (int i = 0; i < f.nx; ++i) for (int k = 0; k < f.nz; ++k) {
            for (int j = 0; j < n; ++j) line[j] = f.at(i, j, k);
            Ax.msolve(line);
            for (int j = 0; j < n; ++j) f.at(i, j, k) = line[j];
        }
    } else {
        for (int i = 0; i < f.nx; ++i) for (int j = 0; j < f.ny; ++j) {
            for (int k = 0; k < n; ++k) line[k] = f.at(i, j, k);
            Ax.msolve(line);
            for (int k = 0; k < n; ++k) f.at(i, j, k) = line[k];
        }
    }
}

// Per-axis quadrature tables for sum-factorization, ELEMENT-indexed with a
// local->global map `glob` so the same advection kernel handles CLAMPED (wall,
// elements = interior spans, glob = e+a) and PERIODIC (channel, elements = the n
// uniform intervals, glob = (e + off0 + a) mod n with off0 = -(p-1)/2; ODD p so
// the local basis set is constant per element). Bv/Bd are the basis VALUE and
// DERIVATIVE matrices at the Gauss points.
struct AxisQuad {
    int ng = 0, np = 0, nelem = 0;
    std::vector<std::vector<double>> Bv, Bd;   // [elem] -> (ng*np) row-major [g][a]
    std::vector<std::vector<double>> jw;       // [elem] -> (ng)
    std::vector<std::vector<int>> glob;        // [elem] -> (np) global indices
    void build(const Axis& A) {
        ng = 2 * A.p; np = A.p + 1;
        std::vector<double> gx, gw; gauss_legendre(ng, gx, gw);
        if (A.per) {
            nelem = A.n; const double h = A.L / A.n; const int off0 = -(A.p - 1) / 2;
            Bv.assign(nelem, {}); Bd.assign(nelem, {}); jw.assign(nelem, {}); glob.assign(nelem, {});
            for (int e = 0; e < nelem; ++e) {
                Bv[e].assign(ng * np, 0.0); Bd[e].assign(ng * np, 0.0); jw[e].assign(ng, 0.0); glob[e].assign(np, 0);
                for (int a = 0; a < np; ++a) glob[e][a] = ((e + off0 + a) % A.n + A.n) % A.n;
                for (int q = 0; q < ng; ++q) {
                    const double xi = 0.5 + 0.5 * gx[q];        // element-local coord in [0,1]
                    for (int a = 0; a < np; ++a) {
                        Bv[e][q * np + a] = beta_value(A.p, xi - off0 - a);
                        Bd[e][q * np + a] = beta_deriv1(A.p, xi - off0 - a) / h;
                    }
                    jw[e][q] = gw[q] * (0.5 * h);
                }
            }
        } else {
            nelem = A.n - A.p;
            Bv.assign(nelem, {}); Bd.assign(nelem, {}); jw.assign(nelem, {}); glob.assign(nelem, {});
            std::vector<std::vector<double>> ders;
            for (int e = 0; e < nelem; ++e) {
                const int s = A.p + e;
                Bv[e].assign(ng * np, 0.0); Bd[e].assign(ng * np, 0.0); jw[e].assign(ng, 0.0); glob[e].assign(np, 0);
                for (int a = 0; a < np; ++a) glob[e][a] = s - A.p + a;
                const double a0 = A.knots[s], b0 = A.knots[s + 1];
                if (b0 <= a0) continue;
                const double half = 0.5 * (b0 - a0), mid = 0.5 * (a0 + b0);
                for (int q = 0; q < ng; ++q) {
                    ders_basis_funs(s, mid + half * gx[q], A.p, 1, A.knots, ders);
                    for (int a = 0; a < np; ++a) { Bv[e][q * np + a] = ders[0][a]; Bd[e][q * np + a] = ders[1][a]; }
                    jw[e][q] = gw[q] * half;
                }
            }
        }
    }
};
// sum-factorized eval: out[ng^3] = (Mx (x) My (x) Mz) in[np^3], Ma are (ng x np).
inline void sf_eval(const double* Mx, const double* My, const double* Mz, int ng, int np,
                    const double* in, double* out, double* s1, double* s2) {
    for (int gx = 0; gx < ng; ++gx) for (int b = 0; b < np; ++b) for (int c = 0; c < np; ++c) {
        double s = 0; for (int a = 0; a < np; ++a) s += Mx[gx * np + a] * in[(a * np + b) * np + c];
        s1[(gx * np + b) * np + c] = s;
    }
    for (int gx = 0; gx < ng; ++gx) for (int gy = 0; gy < ng; ++gy) for (int c = 0; c < np; ++c) {
        double s = 0; for (int b = 0; b < np; ++b) s += My[gy * np + b] * s1[(gx * np + b) * np + c];
        s2[(gx * ng + gy) * np + c] = s;
    }
    for (int gx = 0; gx < ng; ++gx) for (int gy = 0; gy < ng; ++gy) for (int gz = 0; gz < ng; ++gz) {
        double s = 0; for (int c = 0; c < np; ++c) s += Mz[gz * np + c] * s2[(gx * ng + gy) * np + c];
        out[(gx * ng + gy) * ng + gz] = s;
    }
}
// transpose (integrate): out[np^3] += (Mx^T (x) My^T (x) Mz^T) in[ng^3]  (ACCUMULATE)
inline void sf_integ(const double* Mx, const double* My, const double* Mz, int ng, int np,
                     const double* in, double* out, double* s1, double* s2) {
    for (int gx = 0; gx < ng; ++gx) for (int gy = 0; gy < ng; ++gy) for (int c = 0; c < np; ++c) {
        double s = 0; for (int gz = 0; gz < ng; ++gz) s += Mz[gz * np + c] * in[(gx * ng + gy) * ng + gz];
        s2[(gx * ng + gy) * np + c] = s;
    }
    for (int gx = 0; gx < ng; ++gx) for (int b = 0; b < np; ++b) for (int c = 0; c < np; ++c) {
        double s = 0; for (int gy = 0; gy < ng; ++gy) s += My[gy * np + b] * s2[(gx * ng + gy) * np + c];
        s1[(gx * np + b) * np + c] = s;
    }
    for (int a = 0; a < np; ++a) for (int b = 0; b < np; ++b) for (int c = 0; c < np; ++c) {
        double s = 0; for (int gx = 0; gx < ng; ++gx) s += Mx[gx * np + a] * s1[(gx * np + b) * np + c];
        out[(a * np + b) * np + c] += s;
    }
}

struct WallNS3D {
    Axis X, Y, Z;
    double nu;
    bool advect = true;
    bool use_sf = true;                 // sum-factorized advection (else slow Gauss loop)
    AxisQuad aq[3];
    Field3 u, v, w, u0, v0, w0, ku, kv, kw, phi, dvg, t1, t2;
    Field3 ts1, ts2;   // stiff()-private scratch: callers may pass t1/t2 as out
    AxisEig peig[3];                  // pressure Schur eigendata per axis (fast Poisson)
    AxisEig veig[3];                  // stiffness eigendata per axis (implicit viscous)

    // px,py,pz: per-axis periodicity (all false = walled box; e.g. px=py=true,
    // pz=false = a CHANNEL: streamwise/spanwise periodic, wall-normal no-slip).
    WallNS3D(int nx, int ny, int nz, int p, double Lx, double Ly, double Lz, double nu_,
             bool px = false, bool py = false, bool pz = false)
        : X(nx, p, Lx, px), Y(ny, p, Ly, py), Z(nz, p, Lz, pz), nu(nu_),
          u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz),
          u0(nx, ny, nz), v0(nx, ny, nz), w0(nx, ny, nz),
          ku(nx, ny, nz), kv(nx, ny, nz), kw(nx, ny, nz),
          phi(nx, ny, nz), dvg(nx, ny, nz), t1(nx, ny, nz), t2(nx, ny, nz),
          ts1(nx, ny, nz), ts2(nx, ny, nz) {
        const Axis* ax[3] = {&X, &Y, &Z};
        for (int d = 0; d < 3; ++d) {                 // build pressure Schur eig per axis
            const Axis& A = *ax[d];
            DenseMatrix Md(A.n), Pa(A.n);
            for (int i = 0; i < A.n; ++i) for (int j = 0; j < A.n; ++j) Md(i, j) = A.M(i, j);
            // Pa = K M^{-1} K^T : solve M Y = K^T, then Pa = K Y
            for (int col = 0; col < A.n; ++col) {
                std::vector<double> kt(A.n);
                for (int r = 0; r < A.n; ++r) kt[r] = A.K(col, r);   // K^T column = K row
                std::vector<double> y; dense_lu_solve(Md, kt, y);
                for (int i = 0; i < A.n; ++i) { double s = 0; for (int l = 0; l < A.n; ++l) s += A.K(i, l) * y[l]; Pa(i, col) = s; }
            }
            peig[d].build(Pa, Md);
            DenseMatrix Sd(A.n);                          // stiffness for implicit viscous
            for (int i = 0; i < A.n; ++i) for (int j = 0; j < A.n; ++j) Sd(i, j) = A.S(i, j);
            veig[d].build(Sd, Md);
        }
        aq[0].build(X); aq[1].build(Y); aq[2].build(Z);   // sum-factorization tables
    }

    // Fast-diagonalization solve of ( a M(x)M(x)M + b L_A ) phi = rhs, using the
    // per-axis eigendata `eg`. a=0,b=1 with eg=peig is the pressure Poisson.
    void fast_solve(const AxisEig eg[3], double a, double b, const Field3& rhs, Field3& out) {
        line_apply3(0, eg[0].Phi, true, rhs, t1);     // Phi^T along each axis (apply A(l,i))
        line_apply3(1, eg[1].Phi, true, t1, t2);
        line_apply3(2, eg[2].Phi, true, t2, t1);
        for (int i = 0; i < X.n; ++i) for (int j = 0; j < Y.n; ++j) for (int k = 0; k < Z.n; ++k) {
            const double d = a + b * (eg[0].lam[i] + eg[1].lam[j] + eg[2].lam[k]);
            double& val = t1.at(i, j, k);
            val = (std::fabs(d) < 1e-12) ? 0.0 : val / d;            // pin the null (const-pressure) mode
        }
        line_apply3(0, eg[0].Phi, false, t1, t2);     // Phi along each axis
        line_apply3(1, eg[1].Phi, false, t2, t1);
        line_apply3(2, eg[2].Phi, false, t1, out);
    }

    bool is_bndry(int i, int j, int k) const {       // walls only on CLAMPED axes
        return (!X.per && (i == 0 || i == X.n - 1)) ||
               (!Y.per && (j == 0 || j == Y.n - 1)) ||
               (!Z.per && (k == 0 || k == Z.n - 1));
    }
    void no_slip(Field3& f) const {
        for (int i = 0; i < X.n; ++i) for (int j = 0; j < Y.n; ++j) for (int k = 0; k < Z.n; ++k)
            if (is_bndry(i, j, k)) f.at(i, j, k) = 0.0;
    }
    void axpy(Field3& d, double s, const Field3& a) { for (size_t q = 0; q < d.a.size(); ++q) d.a[q] += s * a.a[q]; }

    // weak divergence  d = (Kx My Mz)u + (Mx Ky Mz)v + (Mx My Kz)w
    void weak_div(const Field3& uu, const Field3& vv, const Field3& ww, Field3& d) {
        line_apply3(1, Y.M, 0, uu, t1); line_apply3(2, Z.M, 0, t1, t2); line_apply3(0, X.K, 0, t2, d);
        line_apply3(0, X.M, 0, vv, t1); line_apply3(2, Z.M, 0, t1, t2); line_apply3(1, Y.K, 0, t2, t1);
        axpy(d, 1.0, t1);
        line_apply3(0, X.M, 0, ww, t1); line_apply3(1, Y.M, 0, t1, t2); line_apply3(2, Z.K, 0, t2, t1);
        axpy(d, 1.0, t1);
    }
    void project(bool pin = true) {
        weak_div(u, v, w, dvg);
        fast_solve(peig, 0.0, 1.0, dvg, phi);         // direct fast-diagonalization solve
        line_apply3(0, X.K, 1, phi, t1); minv3(0, X, t1); axpy(u, -1.0, t1);
        line_apply3(1, Y.K, 1, phi, t1); minv3(1, Y, t1); axpy(v, -1.0, t1);
        line_apply3(2, Z.K, 1, phi, t1); minv3(2, Z, t1); axpy(w, -1.0, t1);
        if (pin) { no_slip(u); no_slip(v); no_slip(w); }
    }
    double max_weak_div() {
        weak_div(u, v, w, dvg);
        double m = 0; for (double x : dvg.a) m = std::fmax(m, std::fabs(x)); return m;
    }

    // energy-conserving skew advection -> Cu,Cv,Cw (3D Gauss quadrature).
    // OpenMP over spans with per-thread accumulators (the scatter shares basis
    // functions across adjacent spans, so a plain parallel-for would race).
    void advect_skew(Field3& Cu, Field3& Cv, Field3& Cw) {
        const int p = X.p;
        std::vector<double> gx, gw; gauss_legendre(2 * p, gx, gw);
        std::fill(Cu.a.begin(), Cu.a.end(), 0.0);
        std::fill(Cv.a.begin(), Cv.a.end(), 0.0);
        std::fill(Cw.a.begin(), Cw.a.end(), 0.0);
        const int nx = X.n, ny = Y.n, nz = Z.n;
        #pragma omp parallel
        {
        std::vector<std::vector<double>> dx, dy, dz;
        Field3 Lu(nx, ny, nz), Lv(nx, ny, nz), Lw(nx, ny, nz);
        #pragma omp for collapse(3) schedule(dynamic)
        for (int sx = p; sx < nx; ++sx) {
            for (int sy = p; sy < ny; ++sy) {
                for (int sz = p; sz < nz; ++sz) {
                    const double axl = X.knots[sx], bxl = X.knots[sx + 1]; if (bxl <= axl) continue;
                    const double hx = 0.5 * (bxl - axl), mx = 0.5 * (axl + bxl);
                    const double ayl = Y.knots[sy], byl = Y.knots[sy + 1]; if (byl <= ayl) continue;
                    const double hy = 0.5 * (byl - ayl), my = 0.5 * (ayl + byl);
                    const double azl = Z.knots[sz], bzl = Z.knots[sz + 1]; if (bzl <= azl) continue;
                    const double hz = 0.5 * (bzl - azl), mz = 0.5 * (azl + bzl);
                    for (size_t qx = 0; qx < gx.size(); ++qx) {
                        ders_basis_funs(sx, mx + hx * gx[qx], p, 1, X.knots, dx);
                        for (size_t qy = 0; qy < gx.size(); ++qy) {
                            ders_basis_funs(sy, my + hy * gx[qy], p, 1, Y.knots, dy);
                            for (size_t qz = 0; qz < gx.size(); ++qz) {
                                ders_basis_funs(sz, mz + hz * gx[qz], p, 1, Z.knots, dz);
                                const double wq = gw[qx] * gw[qy] * gw[qz] * hx * hy * hz;
                                double U = 0, V = 0, W = 0;
                                double Ux = 0, Uy = 0, Uz = 0, Vx = 0, Vy = 0, Vz = 0, Wx = 0, Wy = 0, Wz = 0;
                                for (int a = 0; a <= p; ++a) { const int gi = sx - p + a;
                                  for (int b = 0; b <= p; ++b) { const int gj = sy - p + b;
                                    for (int c = 0; c <= p; ++c) { const int gk = sz - p + c;
                                      const double N = dx[0][a] * dy[0][b] * dz[0][c];
                                      const double Nx = dx[1][a] * dy[0][b] * dz[0][c];
                                      const double Ny = dx[0][a] * dy[1][b] * dz[0][c];
                                      const double Nz = dx[0][a] * dy[0][b] * dz[1][c];
                                      const double uu = u.at(gi, gj, gk), vv = v.at(gi, gj, gk), ww = w.at(gi, gj, gk);
                                      U += uu * N; V += vv * N; W += ww * N;
                                      Ux += uu * Nx; Uy += uu * Ny; Uz += uu * Nz;
                                      Vx += vv * Nx; Vy += vv * Ny; Vz += vv * Nz;
                                      Wx += ww * Nx; Wy += ww * Ny; Wz += ww * Nz;
                                    } } }
                                const double adU = U * Ux + V * Uy + W * Uz;
                                const double adV = U * Vx + V * Vy + W * Vz;
                                const double adW = U * Wx + V * Wy + W * Wz;
                                for (int a = 0; a <= p; ++a) { const int gi = sx - p + a;
                                  for (int b = 0; b <= p; ++b) { const int gj = sy - p + b;
                                    for (int c = 0; c <= p; ++c) { const int gk = sz - p + c;
                                      const double N = dx[0][a] * dy[0][b] * dz[0][c];
                                      const double Nx = dx[1][a] * dy[0][b] * dz[0][c];
                                      const double Ny = dx[0][a] * dy[1][b] * dz[0][c];
                                      const double Nz = dx[0][a] * dy[0][b] * dz[1][c];
                                      // skew: 1/2[N (u.grad u_i) - grad N . (u u_i)]
                                      Lu.at(gi, gj, gk) += wq * 0.5 * (N * adU - (Nx * U * U + Ny * V * U + Nz * W * U));
                                      Lv.at(gi, gj, gk) += wq * 0.5 * (N * adV - (Nx * U * V + Ny * V * V + Nz * W * V));
                                      Lw.at(gi, gj, gk) += wq * 0.5 * (N * adW - (Nx * U * W + Ny * V * W + Nz * W * W));
                                    } } }
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (size_t q = 0; q < Cu.a.size(); ++q) { Cu.a[q] += Lu.a[q]; Cv.a[q] += Lv.a[q]; Cw.a[q] += Lw.a[q]; }
        }
        }  // omp parallel
    }

    // SUM-FACTORIZED energy-conserving skew advection: O(p^4)/element vs O(p^6).
    void advect_skew_sf(Field3& Cu, Field3& Cv, Field3& Cw) {
        const int ng = aq[0].ng, np = aq[0].np;
        std::fill(Cu.a.begin(), Cu.a.end(), 0.0);
        std::fill(Cv.a.begin(), Cv.a.end(), 0.0);
        std::fill(Cw.a.begin(), Cw.a.end(), 0.0);
        const int nx = X.n, ny = Y.n, nz = Z.n;
        const int np3 = np * np * np, ng3 = ng * ng * ng;
        #pragma omp parallel
        {
        std::vector<double> s1(ng * np * np), s2(ng * ng * np);
        std::vector<double> Ue(np3), Ve(np3), We(np3), Ce(np3);
        std::vector<double> Ug(ng3), Vg(ng3), Wg(ng3);
        std::vector<double> Uxg(ng3), Uyg(ng3), Uzg(ng3), Vxg(ng3), Vyg(ng3), Vzg(ng3), Wxg(ng3), Wyg(ng3), Wzg(ng3);
        std::vector<double> val(ng3), fx(ng3), fy(ng3), fz(ng3);
        Field3 Lu(nx, ny, nz), Lv(nx, ny, nz), Lw(nx, ny, nz);
        #pragma omp for collapse(3) schedule(dynamic)
        for (int ex = 0; ex < aq[0].nelem; ++ex) for (int ey = 0; ey < aq[1].nelem; ++ey) for (int ez = 0; ez < aq[2].nelem; ++ez) {
            const int* gX = aq[0].glob[ex].data(); const int* gY = aq[1].glob[ey].data(); const int* gZ = aq[2].glob[ez].data();
            for (int a = 0; a < np; ++a) for (int b = 0; b < np; ++b) for (int c = 0; c < np; ++c) {
                const int e = (a * np + b) * np + c;
                Ue[e] = u.at(gX[a], gY[b], gZ[c]);
                Ve[e] = v.at(gX[a], gY[b], gZ[c]);
                We[e] = w.at(gX[a], gY[b], gZ[c]);
            }
            const double *Bx = aq[0].Bv[ex].data(), *Dx = aq[0].Bd[ex].data();
            const double *By = aq[1].Bv[ey].data(), *Dy = aq[1].Bd[ey].data();
            const double *Bz = aq[2].Bv[ez].data(), *Dz = aq[2].Bd[ez].data();
            sf_eval(Bx, By, Bz, ng, np, Ue.data(), Ug.data(), s1.data(), s2.data());
            sf_eval(Dx, By, Bz, ng, np, Ue.data(), Uxg.data(), s1.data(), s2.data());
            sf_eval(Bx, Dy, Bz, ng, np, Ue.data(), Uyg.data(), s1.data(), s2.data());
            sf_eval(Bx, By, Dz, ng, np, Ue.data(), Uzg.data(), s1.data(), s2.data());
            sf_eval(Bx, By, Bz, ng, np, Ve.data(), Vg.data(), s1.data(), s2.data());
            sf_eval(Dx, By, Bz, ng, np, Ve.data(), Vxg.data(), s1.data(), s2.data());
            sf_eval(Bx, Dy, Bz, ng, np, Ve.data(), Vyg.data(), s1.data(), s2.data());
            sf_eval(Bx, By, Dz, ng, np, Ve.data(), Vzg.data(), s1.data(), s2.data());
            sf_eval(Bx, By, Bz, ng, np, We.data(), Wg.data(), s1.data(), s2.data());
            sf_eval(Dx, By, Bz, ng, np, We.data(), Wxg.data(), s1.data(), s2.data());
            sf_eval(Bx, Dy, Bz, ng, np, We.data(), Wyg.data(), s1.data(), s2.data());
            sf_eval(Bx, By, Dz, ng, np, We.data(), Wzg.data(), s1.data(), s2.data());
            const double *jx = aq[0].jw[ex].data(), *jy = aq[1].jw[ey].data(), *jz = aq[2].jw[ez].data();
            for (int comp = 0; comp < 3; ++comp) {
                for (int gx = 0; gx < ng; ++gx) for (int gy = 0; gy < ng; ++gy) for (int gz = 0; gz < ng; ++gz) {
                    const int g = (gx * ng + gy) * ng + gz;
                    const double jw = jx[gx] * jy[gy] * jz[gz];
                    const double U = Ug[g], V = Vg[g], W = Wg[g];
                    double ad, ui;
                    if (comp == 0) { ad = U * Uxg[g] + V * Uyg[g] + W * Uzg[g]; ui = U; }
                    else if (comp == 1) { ad = U * Vxg[g] + V * Vyg[g] + W * Vzg[g]; ui = V; }
                    else { ad = U * Wxg[g] + V * Wyg[g] + W * Wzg[g]; ui = W; }
                    val[g] = 0.5 * jw * ad;
                    fx[g] = -0.5 * jw * U * ui; fy[g] = -0.5 * jw * V * ui; fz[g] = -0.5 * jw * W * ui;
                }
                std::fill(Ce.begin(), Ce.end(), 0.0);
                sf_integ(Bx, By, Bz, ng, np, val.data(), Ce.data(), s1.data(), s2.data());
                sf_integ(Dx, By, Bz, ng, np, fx.data(), Ce.data(), s1.data(), s2.data());
                sf_integ(Bx, Dy, Bz, ng, np, fy.data(), Ce.data(), s1.data(), s2.data());
                sf_integ(Bx, By, Dz, ng, np, fz.data(), Ce.data(), s1.data(), s2.data());
                Field3& Lc = (comp == 0) ? Lu : (comp == 1) ? Lv : Lw;
                for (int a = 0; a < np; ++a) for (int b = 0; b < np; ++b) for (int c = 0; c < np; ++c)
                    Lc.at(gX[a], gY[b], gZ[c]) += Ce[(a * np + b) * np + c];
            }
        }
        #pragma omp critical
        {
            for (size_t q = 0; q < Cu.a.size(); ++q) { Cu.a[q] += Lu.a[q]; Cv.a[q] += Lv.a[q]; Cw.a[q] += Lw.a[q]; }
        }
        }  // omp parallel
    }
    // (Sx My Mz + Mx Sy Mz + Mx My Sz) f -> out.
    // Own scratch (ts1, ts2): rhs() passes the shared t1 as out, and aliasing
    // the internal scratch degenerated the operator to 2*(Mx My Sz) -- no x/y
    // diffusion, doubled z diffusion (same defect class as WallNS2D::stiff,
    // found 2026-07-13 via the Taylor-Green no-slip decay ladder).
    void stiff(const Field3& f, Field3& out) {
        line_apply3(1, Y.M, 0, f, ts1); line_apply3(2, Z.M, 0, ts1, ts2); line_apply3(0, X.S, 0, ts2, out);
        line_apply3(0, X.M, 0, f, ts1); line_apply3(2, Z.M, 0, ts1, ts2); line_apply3(1, Y.S, 0, ts2, ts1); axpy(out, 1.0, ts1);
        line_apply3(0, X.M, 0, f, ts1); line_apply3(1, Y.M, 0, ts1, ts2); line_apply3(2, Z.S, 0, ts2, ts1); axpy(out, 1.0, ts1);
    }
    void mass_solve(Field3& f) { minv3(0, X, f); minv3(1, Y, f); minv3(2, Z, f); }
    // constrained-Galerkin stage rates: interior-restricted on clamped axes
    // (full-space solve + wall pinning injects an O(h) near-wall rate defect)
    void mass_solve_interior(Field3& f) {
        minv3_interior(0, X, f); minv3_interior(1, Y, f); minv3_interior(2, Z, f);
        no_slip(f);   // wall-frame rates owned by the (time-independent) trace data
    }
    void rhs(Field3& du, Field3& dv, Field3& dw) {
        if (advect) { if (use_sf) advect_skew_sf(du, dv, dw); else advect_skew(du, dv, dw); }
        else { for (double& x : du.a) x = 0; for (double& x : dv.a) x = 0; for (double& x : dw.a) x = 0; }
        stiff(u, t1); for (size_t q = 0; q < du.a.size(); ++q) du.a[q] = -du.a[q] - nu * t1.a[q];
        stiff(v, t1); for (size_t q = 0; q < dv.a.size(); ++q) dv.a[q] = -dv.a[q] - nu * t1.a[q];
        stiff(w, t1); for (size_t q = 0; q < dw.a.size(); ++q) dw.a[q] = -dw.a[q] - nu * t1.a[q];
        mass_solve_interior(du); mass_solve_interior(dv); mass_solve_interior(dw);
        no_slip(du); no_slip(dv); no_slip(dw);   // idempotent now
    }
    void combine(double c0, double cc, double ck, double dt) {
        for (size_t q = 0; q < u.a.size(); ++q) {
            u.a[q] = c0 * u0.a[q] + cc * u.a[q] + ck * dt * ku.a[q];
            v.a[q] = c0 * v0.a[q] + cc * v.a[q] + ck * dt * kv.a[q];
            w.a[q] = c0 * w0.a[q] + cc * w.a[q] + ck * dt * kw.a[q];
        }
    }
    void step(double dt) {            // explicit viscous SSP-RK3 (high-Re; viscous-CFL)
        u0 = u; v0 = v; w0 = w;
        rhs(ku, kv, kw); combine(1, 0, 1, dt); no_slip(u); no_slip(v); no_slip(w); project();
        rhs(ku, kv, kw); combine(0.75, 0.25, 0.25, dt); no_slip(u); no_slip(v); no_slip(w); project();
        rhs(ku, kv, kw); combine(1.0/3.0, 2.0/3.0, 2.0/3.0, dt); no_slip(u); no_slip(v); no_slip(w); project();
    }

    void apply_mass(const Field3& f, Field3& out) {       // out = (Mx My Mz) f
        line_apply3(0, X.M, false, f, t1); line_apply3(1, Y.M, false, t1, t2); line_apply3(2, Z.M, false, t2, out);
    }
    void adv_rhs(Field3& du, Field3& dv, Field3& dw) {    // advection only (no viscous)
        if (use_sf) advect_skew_sf(du, dv, dw); else advect_skew(du, dv, dw);
        mass_solve_interior(du); mass_solve_interior(dv); mass_solve_interior(dw);
        for (size_t q = 0; q < du.a.size(); ++q) { du.a[q] = -du.a[q]; dv.a[q] = -dv.a[q]; dw.a[q] = -dw.a[q]; }
        no_slip(du); no_slip(dv); no_slip(dw);
    }
    // IMEX step: SSP-RK3 explicit ENERGY-CONSERVING advection (+project), then
    // IMPLICIT viscous backward Euler via fast diagonalization (no viscous-CFL,
    // so dt is advective only -- the right scheme for higher Re). Lie split.
    void step_imex(double dt) {
        u0 = u; v0 = v; w0 = w;
        adv_rhs(ku, kv, kw); combine(1, 0, 1, dt); no_slip(u); no_slip(v); no_slip(w); project();
        adv_rhs(ku, kv, kw); combine(0.75, 0.25, 0.25, dt); no_slip(u); no_slip(v); no_slip(w); project();
        adv_rhs(ku, kv, kw); combine(1.0/3.0, 2.0/3.0, 2.0/3.0, dt); no_slip(u); no_slip(v); no_slip(w); project();
        // implicit viscous: (M + nu dt S) u = M u*  (per component)
        apply_mass(u, ku); fast_solve(veig, 1.0, nu * dt, ku, u);
        apply_mass(v, kv); fast_solve(veig, 1.0, nu * dt, kv, v);
        apply_mass(w, kw); fast_solve(veig, 1.0, nu * dt, kw, w);
        no_slip(u); no_slip(v); no_slip(w); project();
    }
    double mass_inner(const Field3& f) {              // f^T (Mx My Mz) f
        line_apply3(1, Y.M, 0, f, t1); line_apply3(2, Z.M, 0, t1, t2); line_apply3(0, X.M, 0, t2, t1);
        double s = 0; for (size_t q = 0; q < f.a.size(); ++q) s += f.a[q] * t1.a[q]; return s;
    }
    double kinetic_energy() { return 0.5 * (mass_inner(u) + mass_inner(v) + mass_inner(w)); }

    // coefficient derivative along axis (G = M^-1 K): B(G f) = d/dx_axis (B f)
    void deriv_coeff(int axis, const Field3& f, Field3& out) {
        const Axis& A = (axis == 0) ? X : (axis == 1) ? Y : Z;
        line_apply3(axis, A.K, false, f, out);
        minv3(axis, A, out);
    }
    void to_nodal(const Field3& c, Field3& nod) {           // nodal = (Bx By Bz) c
        line_apply3(0, X.B, false, c, t1);
        line_apply3(1, Y.B, false, t1, t2);
        line_apply3(2, Z.B, false, t2, nod);
    }
    // |vorticity| at the nodes, written x-fastest (VTK ImageData order)
    void vorticity_mag(std::vector<double>& mag) {
        Field3 ox(X.n, Y.n, Z.n), oy(X.n, Y.n, Z.n), oz(X.n, Y.n, Z.n), g(X.n, Y.n, Z.n), nd(X.n, Y.n, Z.n);
        deriv_coeff(1, w, ox); deriv_coeff(2, v, g); axpy(ox, -1.0, g);   // wy - vz
        deriv_coeff(2, u, oy); deriv_coeff(0, w, g); axpy(oy, -1.0, g);   // uz - wx
        deriv_coeff(0, v, oz); deriv_coeff(1, u, g); axpy(oz, -1.0, g);   // vx - uy
        std::vector<double> mx(ox.a.size()), my(ox.a.size()), mz(ox.a.size());
        to_nodal(ox, nd); mx = nd.a; to_nodal(oy, nd); my = nd.a; to_nodal(oz, nd); mz = nd.a;
        mag.assign((size_t)X.n * Y.n * Z.n, 0.0);
        for (int i = 0; i < X.n; ++i) for (int j = 0; j < Y.n; ++j) for (int k = 0; k < Z.n; ++k) {
            const size_t s = ox.idx(i, j, k);
            const double m = std::sqrt(mx[s] * mx[s] + my[s] * my[s] + mz[s] * mz[s]);
            mag[((size_t)k * Y.n + j) * X.n + i] = m;        // VTK: x fastest
        }
    }
};

#endif  // WALL_NS3D_H
