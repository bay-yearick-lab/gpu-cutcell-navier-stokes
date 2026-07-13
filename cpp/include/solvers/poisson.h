// poisson.h
//
// The separable solver that incompressible Navier-Stokes leans on twice:
//
//   * pressure Poisson    L phi = div(u*)         (a=0, b=1)
//   * implicit viscous    (I - dt*g*nu*L) u = rhs (a=1, b=-dt*g*nu)
//
// Both are (a I + b L) x = rhs with L the mimetic Laplacian -- a Kronecker SUM of
// the 1D operators along each axis (the 3d_to_1d structure). A Kronecker sum is
// diagonalized by the per-axis eigenvectors, so on periodic axes the discrete
// Fourier transform turns the solve into a single divide per mode:
//
//   x_hat[i,j,k] = rhs_hat[i,j,k] / (a + b (lx_i + ly_j + lz_k)).
//
// This is EXACT (not iterative): it inverts the very operator that built the
// right-hand side, so an MMS round trip returns to machine precision. A small
// dependency-free radix-2 FFT does the transforms; periodic axes must be powers
// of two. (Clamped/wall axes use a banded solve per Fourier mode instead -- the
// channel extension, planned.)

#ifndef POISSON_H
#define POISSON_H

#include <complex>
#include <vector>
#include <cmath>
#include "bm3d.h"

using cdbl = std::complex<double>;

inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Radix-2 Cooley-Tukey FFT with PRECOMPUTED twiddles and bit-reversal. The plan
// is built once per axis length and reused for every line, so the per-line cost
// has no trig calls -- just butterflies. (Recomputing cos/sin per line was the
// dominant FFT cost.)
struct FFTPlan {
    int n = 0;
    std::vector<int> rev;            // bit-reversal permutation
    std::vector<cdbl> twf, twi;      // twiddles, flattened over stages (size n-1)

    void init(int n_) {
        n = n_;
        rev.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            int r = 0, x = i;
            for (int b = 1; b < n; b <<= 1) { r = (r << 1) | (x & 1); x >>= 1; }
            rev[i] = r;
        }
        twf.clear(); twi.clear();
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            for (int k = 0; k < half; ++k) {
                const double ang = -2.0 * M_PI * k / len;
                twf.emplace_back(std::cos(ang), std::sin(ang));
                twi.emplace_back(std::cos(ang), -std::sin(ang));
            }
        }
    }

    void run(cdbl* a, bool inverse) const {
        for (int i = 0; i < n; ++i) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
        const cdbl* tw = inverse ? twi.data() : twf.data();
        int base = 0;
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < half; ++k) {
                    const cdbl w = tw[base + k];
                    const cdbl u = a[i + k], v = a[i + k + half] * w;
                    a[i + k] = u + v;
                    a[i + k + half] = u - v;
                }
            base += half;
        }
        if (inverse) { const double s = 1.0 / n; for (int i = 0; i < n; ++i) a[i] *= s; }
    }
};

struct SeparablePoisson {
    int nx = 0, ny = 0, nz = 1;
    bool active_z = true;
    std::vector<double> lx, ly, lz;     // 1D Laplacian eigenvalues per axis
    std::vector<char> kx, ky, kz;       // per-axis kernel flags (relative cut; see build)
    mutable std::vector<cdbl> work;     // reused FFT buffer (no per-solve alloc)
    FFTPlan px, py, pz;                  // per-axis FFT plans (precomputed twiddles)

    // Eigenvalues of the periodic 1D mimetic second derivative. It is a symmetric
    // circulant, so the cosine modes are eigenvectors and lambda_k = (L cos_k)[0].
    static std::vector<double> eig_axis(int N, double L, int p, Route route = Route::BMC) {
        FastDeriv f = FastDeriv::build(true, N, p, L);
        std::vector<double> eig(N), cosk(N), out(N), scratch(N);
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) cosk[j] = std::cos(2.0 * M_PI * k * j / N);
            if      (route == Route::FD2) f.fd2_d2(cosk, out);
            else if (route == Route::FD4) f.fd4_d2(cosk, out);
            else if (route == Route::FD6) f.fd6_d2(cosk, out);
            else if (route == Route::C6)  f.compact_d2(cosk, out);
            else if (route == Route::DRP) f.fd6_d2(cosk, out);          // DRP diffusion = E6 d2
            else                          f.bmc_d2(cosk, out, scratch);
            eig[k] = out[0];
        }
        return eig;
    }

    // Eigenvalues of the COMPOSED operator div.grad = D1*D1 along one axis, where
    // D1 is the route's first derivative. The pressure Poisson MUST use this (not
    // the direct second derivative) so that the discrete projection is exactly
    // discrete-divergence-free: div(u - grad phi) = div u - (div.grad) phi = 0.
    // D1 is skew (D1 cos_k = -mu_k sin_k), so (D1 D1) cos_k = -mu_k^2 cos_k.
    static std::vector<double> eig_axis_divgrad(int N, double L, int p, Route route) {
        FastDeriv f = FastDeriv::build(true, N, p, L);
        std::vector<double> eig(N), cosk(N), t1(N), out(N), a(N), b(N);
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) cosk[j] = std::cos(2.0 * M_PI * k * j / N);
            if (route == Route::BMG)      { f.bmg_d1(cosk, t1, a, b); f.bmg_d1(t1, out, a, b); }
            else if (route == Route::FD2) { f.fd2_d1(cosk, t1);       f.fd2_d1(t1, out); }
            else if (route == Route::FD4) { f.fd4_d1(cosk, t1);       f.fd4_d1(t1, out); }
            else if (route == Route::FD6) { f.fd6_d1(cosk, t1);       f.fd6_d1(t1, out); }
            else if (route == Route::C6)  { f.compact_d1(cosk, t1);   f.compact_d1(t1, out); }
            else if (route == Route::DRP) { f.drp_d1(cosk, t1);       f.drp_d1(t1, out); }
            else                          { f.bmc_d1(cosk, t1, a);    f.bmc_d1(t1, out, a); }
            eig[k] = out[0];
        }
        return eig;
    }

    bool build(const SimConfig& c, const char*& why, bool divgrad = false) {
        nx = c.ax[0].N; ny = c.ax[1].N; active_z = c.active(2); nz = active_z ? c.ax[2].N : 1;
        for (int a = 0; a < (active_z ? 3 : 2); ++a) {
            if (!c.periodic(a)) { why = "SeparablePoisson currently requires periodic axes"; return false; }
            if (!is_pow2(c.ax[a].N)) { why = "periodic FFT solve needs power-of-two N"; return false; }
        }
        auto E = [&](int a) {
            return divgrad ? eig_axis_divgrad(c.ax[a].N, c.ax[a].L, c.p, c.route)
                           : eig_axis(c.ax[a].N, c.ax[a].L, c.p, c.route);
        };
        lx = E(0);
        ly = E(1);
        lz = active_z ? E(2) : std::vector<double>(1, 0.0);
        // Per-axis kernel flags by a RELATIVE threshold. The kernel of the composed divgrad operator
        // is the tensor set {0, N/2}^d, but the eigenvalues here are computed
        // NUMERICALLY, so the kernel "zeros" carry round-off scaling with the
        // operator norm (~N^2/L^2). The previous ABSOLUTE |denom| < 1e-12
        // guard in solve() stopped covering them at production sizes (per-axis
        // |eig[N/2]| up to ~7e-12 at N<=512, p<=11; 3-axis sums cross 1e-12
        // even when each axis is below it), so the solve divided by round-off
        // eigenvalues and amplified checkerboard content into phi. The
        // smallest NONZERO |eig| is O((2pi/L)^2), far above kernel round-off,
        // so a relative cut is unambiguous. `<=` so an inactive axis (single
        // exact-0 eigenvalue) is flagged as kernel.
        auto kflag = [](const std::vector<double>& e) {
            double m = 0.0; for (double v : e) m = std::fmax(m, std::fabs(v));
            std::vector<char> f(e.size());
            for (size_t k = 0; k < e.size(); ++k) f[k] = std::fabs(e[k]) <= 1e-10 * m;
            return f;
        };
        kx = kflag(lx); ky = kflag(ly); kz = kflag(lz);
        px.init(nx); py.init(ny); if (active_z) pz.init(nz);
        work.assign((size_t)nx * ny * nz, cdbl(0.0, 0.0));
        return true;
    }

    // FFT all lines along one axis (parallel over lines, per-thread buffer).
    void fft_axis(std::vector<cdbl>& d, int axis, bool inverse) const {
        if (axis == 2 && !active_z) return;          // 2D: z transform is identity
        int n, nlines, stride;
        if (axis == 0)      { n = nx; stride = 1;       nlines = ny * nz; }
        else if (axis == 1) { n = ny; stride = nx;      nlines = nx * nz; }
        else                { n = nz; stride = nx * ny; nlines = nx * ny; }
        const FFTPlan& plan = (axis == 0) ? px : (axis == 1 ? py : pz);
        #pragma omp parallel
        {
            std::vector<cdbl> line(n);
            #pragma omp for schedule(static)
            for (int Ln = 0; Ln < nlines; ++Ln) {
                size_t base;
                if (axis == 0) base = (size_t)Ln * nx;
                else if (axis == 1) base = (size_t)(Ln / nx) * nx * ny + (Ln % nx);
                else base = (size_t)(Ln / nx) * nx + (Ln % nx);
                for (int i = 0; i < n; ++i) line[i] = d[base + (size_t)i * stride];
                plan.run(line.data(), inverse);
                for (int i = 0; i < n; ++i) d[base + (size_t)i * stride] = line[i];
            }
        }
    }

    // Solve (a I + b L) x = rhs into x. Kernel modes of L (the {0, N/2}^d
    // tensor set of the composed divgrad operator) are identified by the
    // per-axis INDEX flags from build(), not by the magnitude of the assembled
    // denominator: a pure Poisson solve (a = 0) projects them out exactly, and
    // a Helmholtz shift (a != 0) divides them by a alone, dropping the
    // round-off eigenvalue content instead of dividing by it.
    void solve(const Field3D& rhs, Field3D& x, double a, double b) const {
        const size_t N = rhs.size();
        std::vector<cdbl>& d = work;                  // reused buffer, no allocation
        for (size_t i = 0; i < N; ++i) d[i] = rhs.a[i];
        fft_axis(d, 0, false); fft_axis(d, 1, false); fft_axis(d, 2, false);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const size_t id = ((size_t)k * ny + j) * nx + i;
                    if (kx[i] && ky[j] && kz[k]) {
                        d[id] = (a == 0.0) ? cdbl(0.0, 0.0) : d[id] / a;
                    } else {
                        d[id] = d[id] / (a + b * (lx[i] + ly[j] + lz[k]));
                    }
                }
        fft_axis(d, 0, true); fft_axis(d, 1, true); fft_axis(d, 2, true);
        for (size_t i = 0; i < N; ++i) x.a[i] = d[i].real();
    }
};

// ---------------------------------------------------------------------------
// WallPoisson: channel geometry -- x,y periodic (power-of-two, FFT) and z a
// clamped WALL. After the x,y FFT, each (kx,ky) mode is an independent 1D problem
// in z, (a_eff I + b L_z) phi = rhs with a_eff = a + b(lx_kx + ly_ky) and L_z the
// clamped second derivative B'' B^{-1}. Writing phi = B alpha turns it into the
// BANDED system (a_eff B + b B'') alpha = rhs, solved without forming any inverse.
// This is the wall-normal kernel the implicit-viscous step and the pressure solve
// need on bounded domains. Exact -> machine-precision MMS round trip.
// ---------------------------------------------------------------------------
struct WallPoisson {
    int nx = 0, ny = 0, nz = 0, p = 0;
    std::vector<double> lx, ly;          // periodic d2 eigenvalues (x, y)
    std::vector<char> kxf, kyf;          // per-axis kernel flags (relative cut)
    DenseMatrix Bz, Bzz;                 // clamped value / second-derivative (banded)
    FFTPlan px, py;
    mutable std::vector<cdbl> work;
    bool ok = false;

    bool build(const SimConfig& c, const char*& why) {
        nx = c.ax[0].N; ny = c.ax[1].N; nz = c.ax[2].N; p = c.p;
        if (!c.active(2)) { why = "WallPoisson is a 3D channel (z active)"; return false; }
        if (!c.periodic(0) || !c.periodic(1)) { why = "WallPoisson needs x,y periodic"; return false; }
        if (c.periodic(2)) { why = "WallPoisson needs z clamped (a wall)"; return false; }
        if (!is_pow2(nx) || !is_pow2(ny)) { why = "x,y need power-of-two N"; return false; }
        lx = SeparablePoisson::eig_axis(nx, c.ax[0].L, p);
        ly = SeparablePoisson::eig_axis(ny, c.ax[1].L, p);
        // relative kernel flags (2026-07-13): the k=0 eigenvalue is a
        // numerical round-off zero scaling with the operator norm, so the
        // absolute test on a_eff in solve() went blind as N grows
        auto kf = [](const std::vector<double>& e) {
            double m = 0.0; for (double v : e) m = std::fmax(m, std::fabs(v));
            std::vector<char> f(e.size());
            for (size_t k = 0; k < e.size(); ++k) f[k] = std::fabs(e[k]) <= 1e-10 * m;
            return f;
        };
        kxf = kf(lx); kyf = kf(ly);
        SplineOperators op = build_clamped(nz, p, c.ax[2].L);
        Bz = op.B; Bzz = op.Bpp;
        px.init(nx); py.init(ny);
        work.assign((size_t)nx * ny * nz, cdbl(0.0, 0.0));
        ok = true; return true;
    }

    // FFT every x- or y-line of the 3D complex buffer.
    void fft_xy(int axis, bool inverse) const {
        const int n = (axis == 0) ? nx : ny;
        const int stride = (axis == 0) ? 1 : nx;
        const int nlines = (axis == 0) ? ny * nz : nx * nz;
        const FFTPlan& plan = (axis == 0) ? px : py;
        #pragma omp parallel
        {
            std::vector<cdbl> line(n);
            #pragma omp for schedule(static)
            for (int Ln = 0; Ln < nlines; ++Ln) {
                const size_t base = (axis == 0) ? (size_t)Ln * nx
                                                : (size_t)(Ln / nx) * nx * ny + (Ln % nx);
                for (int i = 0; i < n; ++i) line[i] = work[base + (size_t)i * stride];
                plan.run(line.data(), inverse);
                for (int i = 0; i < n; ++i) work[base + (size_t)i * stride] = line[i];
            }
        }
    }

    // Solve (a I + b L) x = rhs on the channel.
    void solve(const Field3D& rhs, Field3D& x, double a, double b) const {
        const size_t N = rhs.size();
        for (size_t i = 0; i < N; ++i) work[i] = rhs.a[i];
        fft_xy(0, false); fft_xy(1, false);
        const size_t zstride = (size_t)nx * ny;
        #pragma omp parallel
        {
            DenseMatrix M(nz);                            // banded but solved with pivoting
            std::vector<double> re(nz), im(nz), are, aim;
            #pragma omp for collapse(2) schedule(static)
            for (int ky = 0; ky < ny; ++ky)
                for (int kx = 0; kx < nx; ++kx) {
                    // relative kernel flags, not an absolute test on a_eff
                    // (the old |a_eff| < 1e-12 missed the round-off zeros at
                    // large N and divided by them, 2026-07-13). At the null
                    // (kx,ky) with a = 0 the collocation z-system b Bzz alpha
                    // = rhs carries NO Neumann rows, so its (0,0) mode is
                    // underdetermined as assembled; the line is gauged to zero
                    // as before. Use GalerkinChannelPoisson for a well-posed
                    // horizontally-averaged Neumann profile.
                    const bool nullxy = kxf[kx] && kyf[ky];
                    const double a_eff = nullxy ? a : a + b * (lx[kx] + ly[ky]);
                    const size_t base = (size_t)ky * nx + kx;
                    if (nullxy && a == 0.0) {              // pure-Poisson null mode
                        for (int kz = 0; kz < nz; ++kz) work[base + (size_t)kz * zstride] = cdbl(0.0, 0.0);
                        continue;
                    }
                    // M = a_eff B + b B''  (half-bandwidth p; off-band entries stay 0)
                    for (int i = 0; i < nz; ++i)
                        for (int j = (i - p > 0 ? i - p : 0); j <= (i + p < nz - 1 ? i + p : nz - 1); ++j)
                            M(i, j) = a_eff * Bz(i, j) + b * Bzz(i, j);
                    for (int kz = 0; kz < nz; ++kz) {
                        const cdbl v = work[base + (size_t)kz * zstride];
                        re[kz] = v.real(); im[kz] = v.imag();
                    }
                    dense_lu_solve(M, re, are);            // alpha = M^{-1} rhs (pivoted)
                    dense_lu_solve(M, im, aim);
                    for (int i = 0; i < nz; ++i) {         // phi = B alpha (banded matvec)
                        const int jlo = (i - p > 0 ? i - p : 0), jhi = (i + p < nz - 1 ? i + p : nz - 1);
                        double pr = 0.0, pi = 0.0;
                        for (int j = jlo; j <= jhi; ++j) { pr += Bz(i, j) * are[j]; pi += Bz(i, j) * aim[j]; }
                        work[base + (size_t)i * zstride] = cdbl(pr, pi);
                    }
                }
        }
        fft_xy(0, true); fft_xy(1, true);
        for (size_t i = 0; i < N; ++i) x.a[i] = work[i].real();
    }
};

// ---------------------------------------------------------------------------
// GalerkinChannelPoisson: the WELL-CONDITIONED channel solver. Works in spline
// COEFFICIENT space with the mass/stiffness forms (M, S) instead of the
// collocation matrix B, so it stays at machine precision for every degree p
// (the dense-inverse-of-B approach degrades; this does not). Physical operator
// (a I - b d^2); Galerkin form (a M + b S). x,y periodic (FFT diagonalizes the
// circulant M,S with symbols), z a clamped wall (banded SPD solve per mode):
//
//   per (kx,ky):  (a_eff M_z + b S_z) alpha = rhs_hat_z / (m_x m_y),
//                 a_eff = a + b (sigma_x[kx] + sigma_y[ky]),  sigma = s_sym/m_sym.
//
// Banded M_z, S_z are SPD, so the per-mode solve needs no pivoting and is fast.
// ---------------------------------------------------------------------------
struct GalerkinChannelPoisson {
    int nx = 0, ny = 0, nz = 0, p = 0;
    std::vector<double> mx, my, sx, sy;   // periodic mass/stiffness symbols (x, y)
    std::vector<char> kxf, kyf;           // per-axis stiffness-kernel flags (relative cut)
    std::vector<double> wz;               // int N_j dz (column sums of Mz), for mean removal
    Mat Mz, Sz;                           // clamped mass / stiffness (banded)
    FFTPlan px, py;
    mutable std::vector<cdbl> work;
    bool ok = false;

    // Eigenvalue of a symmetric circulant operator at mode k (cosine eigenvector).
    static double sym(const Mat& C, int N, int k) {
        std::vector<double> cosk(N);
        for (int j = 0; j < N; ++j) cosk[j] = std::cos(2.0 * M_PI * k * j / N);
        return matvec(C, cosk)[0];
    }

    bool build(const SimConfig& c, const char*& why) {
        nx = c.ax[0].N; ny = c.ax[1].N; nz = c.ax[2].N; p = c.p;
        if (!c.active(2)) { why = "GalerkinChannelPoisson is a 3D channel (z active)"; return false; }
        if (!c.periodic(0) || !c.periodic(1)) { why = "needs x,y periodic"; return false; }
        if (c.periodic(2)) { why = "needs z clamped (a wall)"; return false; }
        if (!is_pow2(nx) || !is_pow2(ny)) { why = "x,y need power-of-two N"; return false; }
        CoeffForms cfx = assemble_forms_periodic(nx, p, c.ax[0].L / nx);
        CoeffForms cfy = assemble_forms_periodic(ny, p, c.ax[1].L / ny);
        mx.resize(nx); sx.resize(nx); my.resize(ny); sy.resize(ny);
        for (int k = 0; k < nx; ++k) { mx[k] = sym(cfx.M, nx, k); sx[k] = sym(cfx.S, nx, k); }
        for (int k = 0; k < ny; ++k) { my[k] = sym(cfy.M, ny, k); sy[k] = sym(cfy.S, ny, k); }
        std::vector<double> knots = open_uniform_knots(nz, p, c.ax[2].L);
        CoeffForms cfz = assemble_forms_clamped(nz, p, knots);
        Mz = cfz.M; Sz = cfz.S;
        // Per-axis stiffness-kernel flags by a RELATIVE cut (2026-07-13): the
        // k=0 stiffness symbol is assembled numerically, so its "zero" carries
        // round-off scaling with the operator norm and a fixed absolute test
        // on a_eff goes blind as N grows (same defect class as the
        // SeparablePoisson guard above).
        auto kf = [](const std::vector<double>& s) {
            double m = 0.0; for (double v : s) m = std::fmax(m, std::fabs(v));
            std::vector<char> f(s.size());
            for (size_t k = 0; k < s.size(); ++k) f[k] = std::fabs(s[k]) <= 1e-10 * m;
            return f;
        };
        kxf = kf(sx); kyf = kf(sy);
        wz.assign(nz, 0.0);                  // int N_j = column sums of the mass form
        for (int j = 0; j < nz; ++j) { double s = 0.0; for (int i = 0; i < nz; ++i) s += Mz(i, j); wz[j] = s; }
        px.init(nx); py.init(ny);
        work.assign((size_t)nx * ny * nz, cdbl(0.0, 0.0));
        ok = true; return true;
    }

    void fft_xy(int axis, bool inverse) const {
        const int n = (axis == 0) ? nx : ny, stride = (axis == 0) ? 1 : nx;
        const int nlines = (axis == 0) ? ny * nz : nx * nz;
        const FFTPlan& plan = (axis == 0) ? px : py;
        #pragma omp parallel
        {
            std::vector<cdbl> line(n);
            #pragma omp for schedule(static)
            for (int Ln = 0; Ln < nlines; ++Ln) {
                const size_t base = (axis == 0) ? (size_t)Ln * nx
                                                : (size_t)(Ln / nx) * nx * ny + (Ln % nx);
                for (int i = 0; i < n; ++i) line[i] = work[base + (size_t)i * stride];
                plan.run(line.data(), inverse);
                for (int i = 0; i < n; ++i) work[base + (size_t)i * stride] = line[i];
            }
        }
    }

    // Solve (a M + b S) alpha = rhs (coefficient space). rhs and out are spline
    // coefficient fields. dirichlet=false: natural (Neumann) BC at the z walls
    // (pressure / scalar); pure-Poisson (a=0) projects out the constant null mode.
    // dirichlet=true: homogeneous Dirichlet (no-slip velocity) -- the boundary
    // z-coefficients are pinned to 0 (for clamped splines u(wall)=alpha_boundary).
    void solve(const Field3D& rhs, Field3D& out, double a, double b, bool dirichlet = false) const {
        const size_t N = rhs.size();
        for (size_t i = 0; i < N; ++i) work[i] = rhs.a[i];
        fft_xy(0, false); fft_xy(1, false);
        const size_t zstride = (size_t)nx * ny;
        #pragma omp parallel
        {
            BandSolver bs; bs.n = nz; bs.kl = p; bs.ku = p; bs.w = 2 * p + 1;
            bs.band.assign((size_t)nz * bs.w, 0.0);
            std::vector<double> re(nz), im(nz);
            #pragma omp for collapse(2) schedule(static)
            for (int ky = 0; ky < ny; ++ky)
                for (int kx = 0; kx < nx; ++kx) {
                    // (kx,ky) in the periodic stiffness kernel: sigma_x, sigma_y
                    // are round-off zeros, so drop them instead of adding them
                    // (relative flags from build; an absolute test on a_eff was
                    // blind at large N).
                    const bool nullxy = kxf[kx] && kyf[ky];
                    const double a_eff = nullxy ? a
                                       : a + b * (sx[kx] / mx[kx] + sy[ky] / my[ky]);
                    const size_t base = (size_t)ky * nx + kx;
                    // Neumann pure Poisson at the null (kx,ky): the z-system is
                    // b Sz alpha = rhs, SPD-singular with the 1-dim constant
                    // kernel. The previous code ZEROED the whole z-line, i.e.
                    // discarded the nz-dimensional horizontally-averaged
                    // profile (e.g. p_bar(z)) instead of solving the
                    // kernel-restricted system (2026-07-13). Now: pin
                    // alpha_0 = 0 (row-0 identity keeps the band and the
                    // remaining principal block SPD), solve, then return the
                    // zero-mean representative (the constant shift is pure
                    // gauge for a Neumann Poisson).
                    const bool neumann_null = !dirichlet && nullxy && a == 0.0;
                    const double inv = 1.0 / (mx[kx] * my[ky]);
                    for (int i = 0; i < nz; ++i)
                        for (int j = (i - p > 0 ? i - p : 0); j <= (i + p < nz - 1 ? i + p : nz - 1); ++j)
                            bs.at(i, j) = a_eff * Mz(i, j) + b * Sz(i, j);    // SPD, no pivot needed
                    if (dirichlet) {                                 // pin boundary coeffs to 0 (no-slip)
                        for (int j = 0; j <= p && j < nz; ++j) bs.at(0, j) = (j == 0 ? 1.0 : 0.0);
                        for (int j = nz - 1 - p; j <= nz - 1; ++j) if (j >= 0) bs.at(nz - 1, j) = (j == nz - 1 ? 1.0 : 0.0);
                    }
                    if (neumann_null) {                              // pin alpha_0 (kernel gauge)
                        for (int j = 0; j <= p && j < nz; ++j) bs.at(0, j) = (j == 0 ? 1.0 : 0.0);
                    }
                    bs.factor();
                    for (int kz = 0; kz < nz; ++kz) {
                        const cdbl v = work[base + (size_t)kz * zstride] * inv;
                        re[kz] = v.real(); im[kz] = v.imag();
                    }
                    if (dirichlet) { re[0] = im[0] = re[nz - 1] = im[nz - 1] = 0.0; }
                    if (neumann_null) { re[0] = im[0] = 0.0; }
                    bs.solve_inplace(re); bs.solve_inplace(im);
                    if (neumann_null) {                              // zero-mean representative
                        double sw = 0.0, sr = 0.0, si = 0.0;
                        for (int j = 0; j < nz; ++j) { sw += wz[j]; sr += wz[j] * re[j]; si += wz[j] * im[j]; }
                        const double cr = sr / sw, ci = si / sw;
                        for (int j = 0; j < nz; ++j) { re[j] -= cr; im[j] -= ci; }
                    }
                    for (int kz = 0; kz < nz; ++kz) work[base + (size_t)kz * zstride] = cdbl(re[kz], im[kz]);
                }
        }
        fft_xy(0, true); fft_xy(1, true);
        for (size_t i = 0; i < N; ++i) out.a[i] = work[i].real();
    }
};

#endif // POISSON_H
