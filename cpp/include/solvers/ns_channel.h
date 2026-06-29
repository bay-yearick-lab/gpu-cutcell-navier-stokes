#ifndef NS_CHANNEL_H
#define NS_CHANNEL_H
//
// ns_channel.h -- coefficient-space building blocks for the wall-bounded
// (channel: x,y periodic, z clamped) incompressible solver. Velocity and
// pressure are stored as spline COEFFICIENTS; u = B alpha. The pieces here are
// the coefficient-space gradient/divergence and the discrete pressure projection
// (exact discrete-divergence-free); the implicit viscous solve lives in
// poisson.h (GalerkinChannelPoisson, Neumann or no-slip Dirichlet walls).

#include <vector>
#include <cmath>
#include <complex>
#include "poisson.h"   // FFTPlan, cdbl, GalerkinChannelPoisson, is_pow2
#include "bspline.h"   // assemble_forms_*, knots
#include "config.h"

// ---- per-axis line iteration on a Field3D (dims nx,ny,nz) ------------------
inline void nc_line_geom(int nx, int ny, int /*nz*/, int axis, int L,
                         size_t& base, int& stride, int& n, int NX, int NY, int NZ) {
    if (axis == 0)      { base = (size_t)L * nx;                        stride = 1;        n = NX; }
    else if (axis == 1) { base = (size_t)(L / nx) * nx * ny + (L % nx); stride = nx;       n = NY; }
    else                { base = (size_t)(L / nx) * nx + (L % nx);      stride = nx * ny;  n = NZ; }
}
inline int nc_axis_nlines(int nx, int ny, int nz, int axis) {
    return axis == 0 ? ny * nz : (axis == 1 ? nx * nz : nx * ny);
}

// Coefficient-space first derivative G_a = M_a^{-1} K_a applied along one axis.
// Periodic: out = M^{-1}(K x) per line (K circulant matvec + cyclic solve, both
// exactly circulant so the FFT diagonalizes G_a^2). Clamped: out = Gz x per line
// (Gz = M_z^{-1} K_z dense; M_z is well conditioned so the inverse is harmless).
struct AxisG {
    bool periodic = true;
    Mat K;                 // periodic: K (circulant)
    CyclicSolver Msolve;   // periodic: M^{-1}
    Mat Gz;                // clamped: dense M^{-1} K
    int n = 0, bw = 0;

    void apply_line(const std::vector<double>& in, std::vector<double>& out) const {
        if (periodic) {
            std::vector<double> t(n, 0.0);
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                for (int o = -bw; o <= bw; ++o) { int j = ((i + o) % n + n) % n; s += K(i, j) * in[j]; }
                t[i] = s;
            }
            out = Msolve.solve(t);
        } else {
            out.assign(n, 0.0);
            for (int i = 0; i < n; ++i) {
                const double* r = &Gz.a[(size_t)i * n];
                double s = 0.0; for (int j = 0; j < n; ++j) s += r[j] * in[j];
                out[i] = s;
            }
        }
    }
};

inline void nc_apply_axis(const Field3D& in, Field3D& out, int axis, const AxisG& G) {
    const int nl = nc_axis_nlines(in.nx, in.ny, in.nz, axis);
    #pragma omp parallel
    {
        std::vector<double> line, res;
        #pragma omp for schedule(static)
        for (int L = 0; L < nl; ++L) {
            size_t base; int stride, n;
            nc_line_geom(in.nx, in.ny, in.nz, axis, L, base, stride, n, in.nx, in.ny, in.nz);
            line.assign(n, 0.0);
            for (int i = 0; i < n; ++i) line[i] = in.a[base + (size_t)i * stride];
            G.apply_line(line, res);
            for (int i = 0; i < n; ++i) out.a[base + (size_t)i * stride] = res[i];
        }
    }
}

// out = Op * in along an axis (Op is n x n dense, applied per line). Used for the
// tensor mass apply M alpha (the viscous-solve right-hand side).
inline void nc_axis_mat(const Field3D& in, Field3D& out, int axis, const Mat& Op) {
    const int nl = nc_axis_nlines(in.nx, in.ny, in.nz, axis);
    #pragma omp parallel
    {
        std::vector<double> line, res;
        #pragma omp for schedule(static)
        for (int L = 0; L < nl; ++L) {
            size_t base; int stride, n;
            nc_line_geom(in.nx, in.ny, in.nz, axis, L, base, stride, n, in.nx, in.ny, in.nz);
            line.assign(n, 0.0); res.assign(n, 0.0);
            for (int i = 0; i < n; ++i) line[i] = in.a[base + (size_t)i * stride];
            for (int i = 0; i < n; ++i) {
                const double* row = &Op.a[(size_t)i * n];
                double s = 0.0; for (int j = 0; j < n; ++j) s += row[j] * line[j];
                res[i] = s;
            }
            for (int i = 0; i < n; ++i) out.a[base + (size_t)i * stride] = res[i];
        }
    }
}

// Apply a per-axis FastDeriv op (reduceB = B^{-1}, expandB = B) along an axis.
template <class Op>
inline void nc_axis_fastderiv(const Field3D& in, Field3D& out, int axis, Op op) {
    const int nl = nc_axis_nlines(in.nx, in.ny, in.nz, axis);
    #pragma omp parallel
    {
        std::vector<double> line, res;
        #pragma omp for schedule(static)
        for (int L = 0; L < nl; ++L) {
            size_t base; int stride, n;
            nc_line_geom(in.nx, in.ny, in.nz, axis, L, base, stride, n, in.nx, in.ny, in.nz);
            line.assign(n, 0.0); res.assign(n, 0.0);
            for (int i = 0; i < n; ++i) line[i] = in.a[base + (size_t)i * stride];
            op(line, res);
            for (int i = 0; i < n; ++i) out.a[base + (size_t)i * stride] = res[i];
        }
    }
}

// Coefficient-space discrete projection on a channel. div = sum_a G_a u_a; the
// pressure operator is the consistent composition L = sum_a G_a^2, so
// u - grad(L^{-1} div u) is EXACTLY discrete-divergence-free. Solved by FFT in
// x,y (G_x^2,G_y^2 circulant -> eigenvalues) + a per-mode dense ((lx+ly)I + Gz^2)
// solve in z. The 4 corner (kx,ky) null modes (lx+ly=0) are projected out (the
// collocated pressure null space; BMF staggered has none).
struct ChannelProjector {
    int nx = 0, ny = 0, nz = 0, p = 0;
    AxisG Gx, Gy, Gz;
    Mat Gz2;
    std::vector<double> lx, ly;
    FFTPlan px, py;
    mutable std::vector<cdbl> work;

    bool build(const SimConfig& c, const char*& why) {
        nx = c.ax[0].N; ny = c.ax[1].N; nz = c.ax[2].N; p = c.p;
        if (!c.active(2) || c.periodic(2)) { why = "channel projector needs z clamped"; return false; }
        if (!c.periodic(0) || !c.periodic(1)) { why = "needs x,y periodic"; return false; }
        if (!is_pow2(nx) || !is_pow2(ny)) { why = "x,y need power-of-two N"; return false; }
        CoeffForms cfx = assemble_forms_periodic(nx, p, c.ax[0].L / nx);
        Gx.periodic = true; Gx.n = nx; Gx.bw = p + 1; Gx.K = cfx.K;
        Gx.Msolve.set_from_dense(to_dense(cfx.M), p + 1);
        if (ny == nx && c.ax[1].L == c.ax[0].L) { Gy = Gx; }
        else {
            CoeffForms cfy = assemble_forms_periodic(ny, p, c.ax[1].L / ny);
            Gy.periodic = true; Gy.n = ny; Gy.bw = p + 1; Gy.K = cfy.K;
            Gy.Msolve.set_from_dense(to_dense(cfy.M), p + 1);
        }
        std::vector<double> knots = open_uniform_knots(nz, p, c.ax[2].L);
        CoeffForms cfz = assemble_forms_clamped(nz, p, knots);
        Gz.periodic = false; Gz.n = nz; Gz.Gz = matmul(inverse(cfz.M), cfz.K);
        Gz2 = matmul(Gz.Gz, Gz.Gz);
        lx = eig_axis(Gx); ly = (Gy.n == Gx.n) ? lx : eig_axis(Gy);
        px.init(nx); py.init(ny);
        work.assign((size_t)nx * ny * nz, cdbl(0.0, 0.0));
        return true;
    }

    // lambda[k] = (G^2 cos_k)[0]: the FFT eigenvalue of G_a^2 (real, <= 0).
    static std::vector<double> eig_axis(const AxisG& G) {
        std::vector<double> lam(G.n), cosk(G.n), t1, t2;
        for (int k = 0; k < G.n; ++k) {
            for (int j = 0; j < G.n; ++j) cosk[j] = std::cos(2.0 * M_PI * k * j / G.n);
            G.apply_line(cosk, t1); G.apply_line(t1, t2);
            lam[k] = t2[0];
        }
        return lam;
    }

    void apply_G(const Field3D& in, Field3D& out, int axis) const {
        nc_apply_axis(in, out, axis, axis == 0 ? Gx : (axis == 1 ? Gy : Gz));
    }

    // d = div(u) = G_x u + G_y v + G_z w  (coefficient space).
    void divergence(const Field3D& au, const Field3D& av, const Field3D& aw, Field3D& d) const {
        Field3D t(nx, ny, nz);
        apply_G(au, d, 0);
        apply_G(av, t, 1); for (size_t i = 0; i < d.size(); ++i) d.a[i] += t.a[i];
        apply_G(aw, t, 2); for (size_t i = 0; i < d.size(); ++i) d.a[i] += t.a[i];
    }

    void fxy(int axis, bool inv) const {
        const int n = axis == 0 ? nx : ny, stride = axis == 0 ? 1 : nx;
        const int nl = axis == 0 ? ny * nz : nx * nz;
        const FFTPlan& pl = axis == 0 ? px : py;
        #pragma omp parallel
        { std::vector<cdbl> ln(n);
          #pragma omp for schedule(static)
          for (int L = 0; L < nl; ++L) {
              const size_t base = axis == 0 ? (size_t)L * nx : (size_t)(L / nx) * nx * ny + (L % nx);
              for (int i = 0; i < n; ++i) ln[i] = work[base + (size_t)i * stride];
              pl.run(ln.data(), inv);
              for (int i = 0; i < n; ++i) work[base + (size_t)i * stride] = ln[i];
          } }
    }

    // Solve L phi = rhs (L = sum_a G_a^2). Corner null modes (lx+ly=0) pinned.
    void solve_L(const Field3D& rhs, Field3D& phi) const {
        for (size_t i = 0; i < rhs.size(); ++i) work[i] = rhs.a[i];
        fxy(0, false); fxy(1, false);
        const size_t zs = (size_t)nx * ny;
        #pragma omp parallel
        {
            DenseMatrix A(nz); std::vector<double> re(nz), im(nz), sre, sim;
            #pragma omp for collapse(2) schedule(static)
            for (int ky = 0; ky < ny; ++ky)
                for (int kx = 0; kx < nx; ++kx) {
                    const double sh = lx[kx] + ly[ky];
                    const size_t base = (size_t)ky * nx + kx;
                    if (std::fabs(sh) < 1e-12) {
                        for (int kz = 0; kz < nz; ++kz) work[base + (size_t)kz * zs] = cdbl(0, 0);
                        continue;
                    }
                    for (int i = 0; i < nz; ++i)
                        for (int j = 0; j < nz; ++j) A(i, j) = Gz2(i, j) + (i == j ? sh : 0.0);
                    for (int kz = 0; kz < nz; ++kz) { cdbl v = work[base + (size_t)kz * zs]; re[kz] = v.real(); im[kz] = v.imag(); }
                    dense_lu_solve(A, re, sre); dense_lu_solve(A, im, sim);
                    for (int kz = 0; kz < nz; ++kz) work[base + (size_t)kz * zs] = cdbl(sre[kz], sim[kz]);
                }
        }
        fxy(0, true); fxy(1, true);
        for (size_t i = 0; i < phi.size(); ++i) phi.a[i] = work[i].real();
    }
};

#endif // NS_CHANNEL_H
