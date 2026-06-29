// bm3d.h
//
// A true 3D PDE engine built on the 3d_to_1d reduction: every multidimensional
// B-spline mimetic operator collapses into independent 1D banded/circulant line
// solves along each coordinate axis. A 3D field is a stack of such lines, and the
// lines are independent, so the work parallelizes across all cores with
// OpenMP. Strided directions (y, z) are gathered into a contiguous per-thread
// buffer before the multi-pass solve, which keeps every line in cache.
//
// Nothing dense is ever formed. The per-axis operator is a FastDeriv (bm_apply.h)
// whose factorization is read-only, so all threads share it.

#ifndef BM3D_H
#define BM3D_H

#include <vector>
#include <algorithm>
#include "bm_apply.h"
#include "config.h"

// Flat 3D scalar field, x fastest (unit stride), then y, then z.
struct Field3D {
    int nx = 0, ny = 0, nz = 0;
    std::vector<double> a;
    Field3D() {}
    Field3D(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_),
        a(static_cast<size_t>(nx_) * ny_ * nz_, 0.0) {}
    size_t size() const { return a.size(); }
    size_t idx(int i, int j, int k) const {
        return (static_cast<size_t>(k) * ny + j) * nx + i;
    }
};

enum OpKind { BMC_D1, BMC_D2, BMG_D1, BMG_D2, FD2_D1, FD2_D2, FD4_D1, FD4_D2, FD6_D1, FD6_D2,
              C6_D1, C6_D2, DRP_D1 };

// y += alpha * x over a whole field (parallel).
inline void axpy(Field3D& y, double alpha, const Field3D& x) {
    const size_t N = y.size();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) y.a[i] += alpha * x.a[i];
}
inline void scale(Field3D& y, double s) {
    const size_t N = y.size();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) y.a[i] *= s;
}

// Per-axis derivative engine. One FastDeriv per axis (each may be periodic or
// clamped, any degree, any length). Built either directly or from a SimConfig;
// in the SimConfig path an inactive third axis gives a 2D run.
struct Deriv3D {
    int nx, ny, nz, p;
    FastDeriv fx, fy, fz;
    bool per[3] = {true, true, true};      // axis periodic?
    bool on[3]  = {true, true, true};      // axis active? (z off => 2D)
    Route route_ = Route::BMC;             // which first-derivative route

    Deriv3D(int nx_, int ny_, int nz_, int p_, double Lx, double Ly, double Lz,
            bool periodic = true)
        : nx(nx_), ny(ny_), nz(nz_), p(p_),
          fx(FastDeriv::build(periodic, nx_, p_, Lx)),
          fy(FastDeriv::build(periodic, ny_, p_, Ly)),
          fz(FastDeriv::build(periodic, nz_, p_, Lz)) {
        per[0] = per[1] = per[2] = periodic;
    }

    // Config-driven: per-axis BC/length, 2D when z is inactive, route from config.
    explicit Deriv3D(const SimConfig& c)
        : nx(c.ax[0].N), ny(c.ax[1].N), nz(c.ax[2].on ? c.ax[2].N : 1), p(c.p),
          fx(FastDeriv::build(c.periodic(0), c.ax[0].N, c.p, c.ax[0].L)),
          fy(FastDeriv::build(c.periodic(1), c.ax[1].N, c.p, c.ax[1].L)),
          fz(c.ax[2].on ? FastDeriv::build(c.periodic(2), c.ax[2].N, c.p, c.ax[2].L)
                        : FastDeriv()) {
        for (int a = 0; a < 3; ++a) { on[a] = c.active(a); per[a] = c.active(a) && c.periodic(a); }
        route_ = c.route;
    }

    // Geometry of the lines along a given axis: length n, count nlines, memory
    // stride between consecutive points, and the base offset of line L.
    void axis_geom(int axis, int& n, int& nlines, int& stride) const {
        if (axis == 0)      { n = nx; stride = 1;       nlines = ny * nz; }
        else if (axis == 1) { n = ny; stride = nx;      nlines = nx * nz; }
        else                { n = nz; stride = nx * ny; nlines = nx * ny; }
    }
    size_t line_base(int axis, int L) const {
        if (axis == 0) return static_cast<size_t>(L) * nx;            // contiguous block
        if (axis == 1) return static_cast<size_t>(L / nx) * nx * ny + (L % nx);
        return static_cast<size_t>(L / nx) * nx + (L % nx);          // axis 2
    }
    const FastDeriv& axis_op(int axis) const { return axis == 0 ? fx : (axis == 1 ? fy : fz); }

    // out = (1D operator op) applied along axis to in. Parallel over lines.
    void apply_axis(const Field3D& in, Field3D& out, int axis, OpKind op) const {
        int n, nlines, stride;
        axis_geom(axis, n, nlines, stride);
        const FastDeriv& fd = axis_op(axis);
        #pragma omp parallel
        {
            std::vector<double> line(n), res(n), a(n), b(n);
            #pragma omp for schedule(static)
            for (int L = 0; L < nlines; ++L) {
                const size_t base = line_base(axis, L);
                for (int i = 0; i < n; ++i) line[i] = in.a[base + static_cast<size_t>(i) * stride];
                switch (op) {
                    case BMC_D1: fd.bmc_d1(line, res, a);    break;
                    case BMC_D2: fd.bmc_d2(line, res, a);    break;
                    case BMG_D1: fd.bmg_d1(line, res, a, b); break;
                    case BMG_D2: fd.bmg_d2(line, res, a, b); break;
                    case FD2_D1: fd.fd2_d1(line, res);       break;
                    case FD2_D2: fd.fd2_d2(line, res);       break;
                    case FD4_D1: fd.fd4_d1(line, res);       break;
                    case FD4_D2: fd.fd4_d2(line, res);       break;
                    case FD6_D1: fd.fd6_d1(line, res);       break;
                    case FD6_D2: fd.fd6_d2(line, res);       break;
                    case C6_D1:  fd.compact_d1(line, res);   break;
                    case C6_D2:  fd.compact_d2(line, res);   break;
                    case DRP_D1: fd.drp_d1(line, res);       break;
                }
                for (int i = 0; i < n; ++i) out.a[base + static_cast<size_t>(i) * stride] = res[i];
            }
        }
    }

    // Batched apply over the strided axes y (1) and z (2): W adjacent-i lines are
    // processed together. Their gather/scatter is contiguous (the i index is unit
    // stride), so partial-cache-line waste is eliminated, and the block solve
    // vectorizes across the W lanes. mode 0 = second derivative (Laplacian);
    // mode 1 = advection-diffusion contribution -ca*d1 + nu*d2. Periodic BMC.
    static constexpr int WLANES = 8;
    void batched_yz(const Field3D& in, Field3D& out, int axis, int mode,
                    double ca, double nu, bool add) const {
        const int W = WLANES;
        int n, linestride, crossN;
        if (axis == 1) { n = ny; linestride = nx;      crossN = nz; }
        else           { n = nz; linestride = nx * ny; crossN = ny; }
        const int iblocks = (nx + W - 1) / W;
        const int ngroups = crossN * iblocks;
        const FastDeriv& fd = axis_op(axis);
        #pragma omp parallel
        {
            std::vector<double> blk((size_t)n * W), d1((size_t)n * W),
                                d2((size_t)n * W), sc((size_t)n * W);
            #pragma omp for schedule(static)
            for (int g = 0; g < ngroups; ++g) {
                const int cross = g / iblocks, i0 = (g % iblocks) * W;
                const int Wb = (nx - i0 < W) ? (nx - i0) : W;
                const size_t base = (axis == 1) ? ((size_t)cross * nx * ny + i0)
                                                : ((size_t)cross * nx + i0);
                for (int row = 0; row < n; ++row) {           // contiguous gather
                    const double* src = &in.a[base + (size_t)row * linestride];
                    double* dst = &blk[(size_t)row * Wb];
                    for (int l = 0; l < Wb; ++l) dst[l] = src[l];
                }
                if (mode == 0) {
                    fd.bmc_d2_block(blk.data(), d2.data(), sc.data(), Wb);
                    for (int row = 0; row < n; ++row) {
                        double* o = &out.a[base + (size_t)row * linestride];
                        const double* s = &d2[(size_t)row * Wb];
                        if (add) for (int l = 0; l < Wb; ++l) o[l] += s[l];
                        else     for (int l = 0; l < Wb; ++l) o[l]  = s[l];
                    }
                } else {
                    fd.bmc_d1d2_block(blk.data(), d1.data(), d2.data(), sc.data(), Wb);
                    for (int row = 0; row < n; ++row) {
                        double* o = &out.a[base + (size_t)row * linestride];
                        const double* a1 = &d1[(size_t)row * Wb];
                        const double* a2 = &d2[(size_t)row * Wb];
                        if (add) for (int l = 0; l < Wb; ++l) o[l] += -ca * a1[l] + nu * a2[l];
                        else     for (int l = 0; l < Wb; ++l) o[l]  = -ca * a1[l] + nu * a2[l];
                    }
                }
            }
        }
    }

    // Contiguous x-axis advection-diffusion contribution (one solve, two expands).
    void advdiff_x(const Field3D& in, Field3D& rhs, double cx, double nu, bool add) const {
        const int n = nx, nlines = ny * nz;
        #pragma omp parallel
        {
            std::vector<double> line(n), d1(n), d2(n), a(n);
            #pragma omp for schedule(static)
            for (int L = 0; L < nlines; ++L) {
                const size_t base = (size_t)L * nx;
                for (int i = 0; i < n; ++i) line[i] = in.a[base + i];
                a = line; fx.reduceB(a);
                fx.into_Bp(a, d1);
                fx.into_Bpp(a, d2);
                if (add) for (int i = 0; i < n; ++i) rhs.a[base + i] += -cx * d1[i] + nu * d2[i];
                else     for (int i = 0; i < n; ++i) rhs.a[base + i]  = -cx * d1[i] + nu * d2[i];
            }
        }
    }

    // out = Laplacian(in) = d2x + d2y + d2z. x is contiguous (scalar); y, z use
    // the batched path. No temporary field is needed.
    void laplacian(const Field3D& in, Field3D& out) const {
        apply_axis(in, out, 0, BMC_D2);          // out = d2x
        batched_yz(in, out, 1, 0, 0, 0, true);   // out += d2y
        batched_yz(in, out, 2, 0, 0, 0, true);   // out += d2z
    }

    // Generic same-grid Laplacian for configurable periodic routes. This is slower
    // than the BMC batched path above because y/z use one gathered line at a time,
    // but it exposes BMG_D2 honestly for experiments and correctness checks.
    void laplacian_generic(const Field3D& in, Field3D& out, OpKind d2_op) const {
        Field3D tmp(nx, ny, nz);
        apply_axis(in, out, 0, d2_op);
        apply_axis(in, tmp, 1, d2_op);
        axpy(out, 1.0, tmp);
        apply_axis(in, tmp, 2, d2_op);
        axpy(out, 1.0, tmp);
    }

    // rhs of linear advection-diffusion  du/dt = -c.grad u + nu*lap u.
    void advdiff_rhs(const Field3D& in, Field3D& rhs,
                     double cx, double cy, double cz, double nu) const {
        advdiff_x(in, rhs, cx, nu, false);            // rhs  = x contribution
        batched_yz(in, rhs, 1, 1, cy, nu, true);      // rhs += y contribution
        batched_yz(in, rhs, 2, 1, cz, nu, true);      // rhs += z contribution
    }

    void advdiff_rhs_generic(const Field3D& in, Field3D& rhs,
                             double cx, double cy, double cz, double nu,
                             OpKind d1_op, OpKind d2_op) const {
        Field3D tmp(nx, ny, nz);
        advdiff_rhs_generic(in, rhs, tmp, cx, cy, cz, nu, d1_op, d2_op);
    }

    void advdiff_rhs_generic(const Field3D& in, Field3D& rhs, Field3D& tmp,
                             double cx, double cy, double cz, double nu,
                             OpKind d1_op, OpKind d2_op) const {
        apply_axis(in, rhs, 0, d1_op);
        scale(rhs, -cx);
        apply_axis(in, tmp, 0, d2_op);
        axpy(rhs, nu, tmp);

        apply_axis(in, tmp, 1, d1_op);
        axpy(rhs, -cy, tmp);
        apply_axis(in, tmp, 1, d2_op);
        axpy(rhs, nu, tmp);

        apply_axis(in, tmp, 2, d1_op);
        axpy(rhs, -cz, tmp);
        apply_axis(in, tmp, 2, d2_op);
        axpy(rhs, nu, tmp);
    }

    // -----------------------------------------------------------------------
    // Config-aware vector calculus: honors the configured route (BMC/BMG) and
    // skips an inactive third axis (2D). Correct for periodic and clamped axes.
    // These are the PDE-agnostic building blocks the Navier-Stokes layer composes
    // (grad p, div u, viscous Laplacian).
    // -----------------------------------------------------------------------
    OpKind opkind(int order) const {
        if (route_ == Route::FD2) return order == 1 ? FD2_D1 : FD2_D2;
        if (route_ == Route::FD4) return order == 1 ? FD4_D1 : FD4_D2;
        if (route_ == Route::FD6) return order == 1 ? FD6_D1 : FD6_D2;
        if (route_ == Route::C6)  return order == 1 ? C6_D1  : C6_D2;
        if (route_ == Route::DRP) return order == 1 ? DRP_D1 : FD6_D2;   // DRP advection + E6 diffusion
        if (order == 1) return route_ == Route::BMG ? BMG_D1 : BMC_D1;
        return BMC_D2;                              // same-point curvature (valid on every BC)
    }
    // Batched BMC derivative along a strided axis (y=1, z=2): W adjacent-i lines
    // at once, contiguous gather + lane-vectorized block solve. order 1 or 2.
    void batched_axis_bmc(const Field3D& in, Field3D& out, int axis, int order) const {
        const int W = WLANES;
        int n, linestride, crossN;
        if (axis == 1) { n = ny; linestride = nx;      crossN = nz; }
        else           { n = nz; linestride = nx * ny; crossN = ny; }
        const int iblocks = (nx + W - 1) / W, ngroups = crossN * iblocks;
        const FastDeriv& fd = axis_op(axis);
        #pragma omp parallel
        {
            std::vector<double> blk((size_t)n * W), res((size_t)n * W), sc((size_t)n * W);
            #pragma omp for schedule(static)
            for (int g = 0; g < ngroups; ++g) {
                const int cross = g / iblocks, i0 = (g % iblocks) * W;
                const int Wb = (nx - i0 < W) ? (nx - i0) : W;
                const size_t base = (axis == 1) ? ((size_t)cross * nx * ny + i0)
                                                : ((size_t)cross * nx + i0);
                for (int row = 0; row < n; ++row) {
                    const double* s = &in.a[base + (size_t)row * linestride];
                    double* d = &blk[(size_t)row * Wb];
                    for (int l = 0; l < Wb; ++l) d[l] = s[l];
                }
                if (order == 1) fd.bmc_d1_block(blk.data(), res.data(), sc.data(), Wb);
                else            fd.bmc_d2_block(blk.data(), res.data(), sc.data(), Wb);
                for (int row = 0; row < n; ++row) {
                    double* o = &out.a[base + (size_t)row * linestride];
                    const double* r = &res[(size_t)row * Wb];
                    for (int l = 0; l < Wb; ++l) o[l] = r[l];
                }
            }
        }
    }
    // order-th derivative along one axis; an inactive axis contributes zero.
    // Periodic BMC strided axes take the fast batched path; everything else
    // (x-axis contiguous, clamped walls, BMG route) takes the scalar path.
    void deriv(const Field3D& in, Field3D& out, int axis, int order) const {
        if (!on[axis]) { std::fill(out.a.begin(), out.a.end(), 0.0); return; }
        if (axis >= 1 && per[axis] && route_ == Route::BMC)
            batched_axis_bmc(in, out, axis, order);
        else
            apply_axis(in, out, axis, opkind(order));
    }
    void grad(const Field3D& s, Field3D& gx, Field3D& gy, Field3D& gz) const {
        deriv(s, gx, 0, 1);
        deriv(s, gy, 1, 1);
        deriv(s, gz, 2, 1);                         // writes zero in 2D
    }
    void divergence(const Field3D& vx, const Field3D& vy, const Field3D& vz,
                    Field3D& out, Field3D& tmp) const {
        deriv(vx, out, 0, 1);
        deriv(vy, tmp, 1, 1); axpy(out, 1.0, tmp);
        if (on[2]) { deriv(vz, tmp, 2, 1); axpy(out, 1.0, tmp); }
    }
    void laplacian_any(const Field3D& s, Field3D& out, Field3D& tmp) const {
        deriv(s, out, 0, 2);
        deriv(s, tmp, 1, 2); axpy(out, 1.0, tmp);
        if (on[2]) { deriv(s, tmp, 2, 2); axpy(out, 1.0, tmp); }
    }
};

#endif // BM3D_H
