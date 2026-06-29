// bmf3d.h
//
// 3D B-spline Mimetic Flux (BMF) operators for WALL-BOUNDED domains -- the
// staggered scheme. The 1D flux complex (bm_operators.h) gives, per axis, the
// staggered gradient G : primal (N) -> derivative grid (N-1), the metric-adjoint
// divergence Dv : derivative grid -> primal, and the scheme-level second
// derivative D2 = Dv G (primal -> primal). The 3D operators are the directional
// collapse of these.
//
//   * laplacian  = sum_a Dv_a G_a   (primal -> primal): the wall-appropriate,
//                  well-conditioned Laplacian (no Nyquist null mode; natural
//                  spline metric). On a periodic axis the complex collapses to BMC.
//   * grad       : primal scalar -> staggered vector (each component on the
//                  derivative grid of its axis): the MAC velocity/pressure pair.
//   * divergence : staggered vector -> primal scalar.
//
// 1D operators are applied DENSE per line (correctness first); a banded
// reduce-solve-expand BMF apply is the speed follow-up. Staggered grad/div are
// implemented for clamped (wall) axes -- the case the scheme is built for.

#ifndef BMF3D_H
#define BMF3D_H

#include <vector>
#include "bm3d.h"
#include "bm_operators.h"
#include "poisson.h"

// ---------------------------------------------------------------------------
// BMFBanded: matrix-free banded apply of the 1D BMF flux complex (clamped axis),
// the speed/conditioning replacement for the DENSE operators. Every step is a
// banded solve of B, Bhat, or M (never their DENSE inverse) or a banded matvec:
//
//   G  = Bhat * Delta * B^{-1}                       (primal N -> derivative N-1)
//   Dv = H0^{-1} (Bd - G^T H1),  H0^{-1} = B M^{-1} B^T,  H1 = Bhat^{-T} Mhat Bhat^{-1}
//
// so no dense N x N inverse is ever formed (the source of the conditioning /
// eigenvalue inflation of the dense build). O(N*bw) per apply. The endpoint
// reader rows r_w = Bhat^{-T} e_w are precomputed once.
// ---------------------------------------------------------------------------
// Actual half-bandwidth of a (square) Mat -- max |i-j| with a nonzero entry. Used
// so the banded factors capture the full matrix (no truncation, any degree).
inline int bandwidth(const Mat& A) {
    int bw = 0;
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j)
            if (std::fabs(A(i, j)) > 1e-13 && std::abs(i - j) > bw) bw = std::abs(i - j);
    return bw;
}

// Banded matvec y = A x using only the band |i-j| <= bw around the diagonal
// (handles rectangular A: the BMF factors all have entries clustered at j ~ i).
inline std::vector<double> banded_matvec(const Mat& A, const std::vector<double>& x, int bw) {
    std::vector<double> y(A.r, 0.0);
    for (int i = 0; i < A.r; ++i) {
        const int j0 = i - bw > 0 ? i - bw : 0, j1 = i + bw < A.c - 1 ? i + bw : A.c - 1;
        double s = 0.0;
        for (int j = j0; j <= j1; ++j) s += A(i, j) * x[j];
        y[i] = s;
    }
    return y;
}

struct BMFBanded {
    int N = 0, p = 0, m = 0;
    int bB = 0, bH = 0, bMh = 0, bD = 0;       // band half-widths of B, Bhat, Mhat, Delta
    Mat Bm, BhatM, MhatM, DeltaM, BmT, BhatMT, DeltaMT;
    BandSolver Bs, BTs, Bhs, BhTs, Ms;
    std::vector<double> r0, r1;

    void build(int N_, int p_, double L) {
        N = N_; p = p_; m = N - 1;
        std::vector<double> knots = open_uniform_knots(N, p, L);
        SplineOperators op = build_clamped(N, p, L);
        CoeffForms cf = assemble_forms_clamped(N, p, knots);
        DerivativeSpace ds = build_derivative_space(N, p, knots);
        Bm = to_mat(op.B); BhatM = ds.Bhat; MhatM = ds.forms.M; DeltaM = ds.Delta;
        BmT = transpose(Bm); BhatMT = transpose(BhatM); DeltaMT = transpose(DeltaM);
        bB = bandwidth(Bm); bH = bandwidth(BhatM); bMh = bandwidth(MhatM); bD = bandwidth(DeltaM);
        const int bM = bandwidth(cf.M);
        Bs.set_from_dense(to_dense(Bm), bB, bB);     Bs.factor();
        BTs.set_from_dense(to_dense(BmT), bB, bB);   BTs.factor();
        Bhs.set_from_dense(to_dense(BhatM), bH, bH); Bhs.factor();
        BhTs.set_from_dense(to_dense(BhatMT), bH, bH); BhTs.factor();
        Ms.set_from_dense(to_dense(cf.M), bM, bM);   Ms.factor();
        std::vector<std::vector<double>> dbuf;
        std::vector<double> e0(m, 0.0), e1(m, 0.0);
        int s0 = find_span(m - 1, ds.p1, 0.0, ds.knots);
        ders_basis_funs(s0, 0.0, ds.p1, 0, ds.knots, dbuf);
        for (int j = 0; j <= ds.p1; ++j) e0[s0 - ds.p1 + j] = dbuf[0][j];
        int s1 = find_span(m - 1, ds.p1, L, ds.knots);
        ders_basis_funs(s1, L, ds.p1, 0, ds.knots, dbuf);
        for (int j = 0; j <= ds.p1; ++j) e1[s1 - ds.p1 + j] = dbuf[0][j];
        r0 = BhTs.solve(e0); r1 = BhTs.solve(e1);
    }

    std::vector<double> G(const std::vector<double>& u) const {       // primal -> derivative grid
        std::vector<double> beta = banded_matvec(DeltaM, Bs.solve(u), bD);
        return banded_matvec(BhatM, beta, bH);
    }
    std::vector<double> Dv(const std::vector<double>& g) const {      // derivative grid -> primal
        std::vector<double> H1g = BhTs.solve(banded_matvec(MhatM, Bhs.solve(g), bMh));
        std::vector<double> GtH1g = BTs.solve(banded_matvec(DeltaMT, banded_matvec(BhatMT, H1g, bH), bD));
        double d0 = 0.0, d1 = 0.0;
        for (int j = 0; j < m; ++j) { d0 += r0[j] * g[j]; d1 += r1[j] * g[j]; }
        std::vector<double> inner(N);
        for (int i = 0; i < N; ++i) inner[i] = -GtH1g[i];
        inner[0] -= d0; inner[N - 1] += d1;                          // + Bd g
        return banded_matvec(Bm, Ms.solve(banded_matvec(BmT, inner, bB)), bB);  // H0^{-1} inner
    }
};

struct BMFDeriv3D {
    int nx, ny, nz;
    bool on3[3], per3[3];
    Mat D2a[3];                        // per-axis Laplacian Dv.G (BMF wall) / B''B^-1 (BMC periodic)
    Mat Ga[3];                         // per-axis staggered gradient (primal -> derivative grid)
    Mat Dva[3];                        // per-axis staggered divergence (derivative grid -> primal)
    int sn[3];                         // staggered length along each axis (N-1 wall, N periodic)

    explicit BMFDeriv3D(const SimConfig& c)
        : nx(c.ax[0].N), ny(c.ax[1].N), nz(c.active(2) ? c.ax[2].N : 1) {
        for (int a = 0; a < 3; ++a) { on3[a] = c.active(a); per3[a] = c.periodic(a); sn[a] = 1; }
        for (int a = 0; a < 3; ++a) {
            if (!on3[a]) continue;
            if (per3[a]) {
                OperatorSet os = build_bmc_periodic(c.ax[a].N, c.p, c.ax[a].L);
                D2a[a] = os.D2; Ga[a] = os.D1; Dva[a] = os.D1;     // collapses to collocation
                sn[a] = c.ax[a].N;
            } else {
                OperatorSet os = build_bmf_bounded(c.ax[a].N, c.p, c.ax[a].L);
                D2a[a] = os.D2; Ga[a] = os.D1; Dva[a] = os.Dv;     // staggered: N -> N-1, N-1 -> N
                sn[a] = c.ax[a].N - 1;
            }
        }
    }
    int dim(int a) const { return a == 0 ? nx : (a == 1 ? ny : nz); }

    // Line base/stride/len for a field with dims (fx,fy,fz) along an axis, at the
    // cross-index (c1,c2). Cross dims are the two non-axis dims, shared by any two
    // fields that differ only along this axis.
    static void lgeo(int fx, int fy, int fz, int axis, int c1, int c2,
                     size_t& base, int& stride, int& len) {
        if (axis == 0)      { base = ((size_t)c2 * fy + c1) * fx; stride = 1;            len = fx; }
        else if (axis == 1) { base = (size_t)c2 * fx * fy + c1;   stride = fx;           len = fy; }
        else                { base = (size_t)c2 * fx + c1;        stride = (size_t)fx * fy; len = fz; }
    }

    // out = Op * in along an axis, Op rectangular (out_len x in_len); in and out
    // differ only along this axis. Parallel over the cross plane.
    void apply_rect(const Field3D& in, Field3D& out, int axis, const Mat& Op) const {
        int n1, n2;                    // cross-plane sizes (shared by in and out)
        if (axis == 0)      { n1 = in.ny; n2 = in.nz; }
        else if (axis == 1) { n1 = in.nx; n2 = in.nz; }
        else                { n1 = in.nx; n2 = in.ny; }
        const int ilen = (axis == 0) ? in.nx : (axis == 1 ? in.ny : in.nz);
        const int olen = (axis == 0) ? out.nx : (axis == 1 ? out.ny : out.nz);
        #pragma omp parallel
        {
            std::vector<double> line(ilen), res(olen);
            #pragma omp for collapse(2) schedule(static)
            for (int c2 = 0; c2 < n2; ++c2)
                for (int c1 = 0; c1 < n1; ++c1) {
                    size_t ib, ob; int is, os_, il, ol;
                    lgeo(in.nx, in.ny, in.nz, axis, c1, c2, ib, is, il);
                    lgeo(out.nx, out.ny, out.nz, axis, c1, c2, ob, os_, ol);
                    for (int i = 0; i < il; ++i) line[i] = in.a[ib + (size_t)i * is];
                    for (int i = 0; i < ol; ++i) {
                        const double* row = &Op.a[(size_t)i * il];
                        double s = 0.0;
                        for (int j = 0; j < il; ++j) s += row[j] * line[j];
                        res[i] = s;
                    }
                    for (int i = 0; i < ol; ++i) out.a[ob + (size_t)i * os_] = res[i];
                }
        }
    }

    // BMF Laplacian = sum_a Dv_a G_a  (primal -> primal), one temp.
    void laplacian(const Field3D& in, Field3D& out, Field3D& tmp) const {
        apply_rect(in, out, 0, D2a[0]);
        apply_rect(in, tmp, 1, D2a[1]); axpy(out, 1.0, tmp);
        if (on3[2]) { apply_rect(in, tmp, 2, D2a[2]); axpy(out, 1.0, tmp); }
    }

    // Staggered field for the gradient component along axis a (a-dim is sn[a]).
    Field3D make_stag(int a) const {
        return Field3D(a == 0 ? sn[0] : nx, a == 1 ? sn[1] : ny, a == 2 ? sn[2] : nz);
    }
    // grad: primal scalar -> staggered components (gx on x-derivative grid, ...).
    void grad(const Field3D& phi, Field3D& gx, Field3D& gy, Field3D& gz) const {
        apply_rect(phi, gx, 0, Ga[0]);
        apply_rect(phi, gy, 1, Ga[1]);
        if (on3[2]) apply_rect(phi, gz, 2, Ga[2]);
    }
    // divergence: staggered components -> primal scalar, one temp (primal-shaped).
    void divergence(const Field3D& gx, const Field3D& gy, const Field3D& gz,
                    Field3D& out, Field3D& tmp) const {
        apply_rect(gx, out, 0, Dva[0]);
        apply_rect(gy, tmp, 1, Dva[1]); axpy(out, 1.0, tmp);
        if (on3[2]) { apply_rect(gz, tmp, 2, Dva[2]); axpy(out, 1.0, tmp); }
    }
};

// ---------------------------------------------------------------------------
// BMFChannelPoisson: solve (a I + b L) x = rhs for the BMF Laplacian on a
// channel -- x,y periodic (FFT), z a clamped wall. After the x,y FFT each (kx,ky)
// mode is the 1D problem (a_eff I + b L_z) phi_z = rhs_z with a_eff = a + b(lx+ly)
// and L_z = Dv_z G_z the BMF wall Laplacian (dense, well-conditioned -- the fix
// for the collocation conditioning that WallPoisson exposed). Solved per mode
// with a pivoted dense LU. Exact -> machine-precision MMS round trip at any p.
// ---------------------------------------------------------------------------
struct BMFChannelPoisson {
    int nx = 0, ny = 0, nz = 0, p = 0;
    std::vector<double> lx, ly;          // periodic BMC d2 eigenvalues (x, y)
    Mat Lz;                              // BMF z-Laplacian Dv_z G_z (nz x nz dense)
    FastDeriv fx, fy;                    // exactly-circulant d2 for x,y (matches lx,ly + FFT)
    FFTPlan px, py;
    mutable std::vector<cdbl> work;
    bool ok = false;

    // divgrad=false: pressure operator = d2x + d2y + Dv_z G_z (the scheme Laplacian,
    // for round-trip checks). divgrad=true: pressure operator = div.grad =
    // D1^2x + D1^2y + Dv_z G_z, matching the staggered velocity div/grad so the discrete
    // projection is consistent (the periodic Nyquist null mode is then pinned).
    bool build(const SimConfig& c, const char*& why, bool divgrad = false) {
        nx = c.ax[0].N; ny = c.ax[1].N; nz = c.ax[2].N; p = c.p;
        if (!c.active(2)) { why = "BMFChannelPoisson is a 3D channel (z active)"; return false; }
        if (!c.periodic(0) || !c.periodic(1)) { why = "needs x,y periodic"; return false; }
        if (c.periodic(2)) { why = "needs z clamped (a wall)"; return false; }
        if (!is_pow2(nx) || !is_pow2(ny)) { why = "x,y need power-of-two N"; return false; }
        fx = FastDeriv::build(true, nx, p, c.ax[0].L);
        fy = FastDeriv::build(true, ny, p, c.ax[1].L);
        lx = divgrad ? SeparablePoisson::eig_axis_divgrad(nx, c.ax[0].L, p, c.route)
                     : SeparablePoisson::eig_axis(nx, c.ax[0].L, p);
        ly = divgrad ? SeparablePoisson::eig_axis_divgrad(ny, c.ax[1].L, p, c.route)
                     : SeparablePoisson::eig_axis(ny, c.ax[1].L, p);
        Lz = build_bmf_bounded(nz, p, c.ax[2].L).D2;         // Dv_z G_z
        px.init(nx); py.init(ny);
        work.assign((size_t)nx * ny * nz, cdbl(0.0, 0.0));
        ok = true; return true;
    }

    // The operator the solver inverts: out = a*in + b*L*in, with L = d2x + d2y + Lz.
    // d2x, d2y are the exactly-circulant FastDeriv second derivatives (so the FFT
    // diagonalizes them exactly with lx, ly); Lz is the dense BMF wall Laplacian.
    void apply(const Field3D& in, Field3D& out, double a, double b) const {
        const size_t N = in.size();
        for (size_t i = 0; i < N; ++i) out.a[i] = a * in.a[i];
        // + b d2x (contiguous x-lines) and + b d2y (strided y-lines)
        for (int axis = 0; axis < 2; ++axis) {
            const FastDeriv& fd = (axis == 0) ? fx : fy;
            const int n = (axis == 0) ? nx : ny, stride = (axis == 0) ? 1 : nx;
            const int nlines = (axis == 0) ? ny * nz : nx * nz;
            #pragma omp parallel
            {
                std::vector<double> line(n), res(n), sc(n);
                #pragma omp for schedule(static)
                for (int Ln = 0; Ln < nlines; ++Ln) {
                    const size_t base = (axis == 0) ? (size_t)Ln * nx
                                                    : (size_t)(Ln / nx) * nx * ny + (Ln % nx);
                    for (int i = 0; i < n; ++i) line[i] = in.a[base + (size_t)i * stride];
                    fd.bmc_d2(line, res, sc);
                    for (int i = 0; i < n; ++i) out.a[base + (size_t)i * stride] += b * res[i];
                }
            }
        }
        // + b Lz along z (dense)
        const size_t zstride = (size_t)nx * ny;
        #pragma omp parallel
        {
            std::vector<double> line(nz), res(nz);
            #pragma omp for collapse(2) schedule(static)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const size_t base = (size_t)j * nx + i;
                    for (int kz = 0; kz < nz; ++kz) line[kz] = in.a[base + (size_t)kz * zstride];
                    for (int r = 0; r < nz; ++r) { const double* row = &Lz.a[(size_t)r * nz];
                        double s = 0.0; for (int cc = 0; cc < nz; ++cc) s += row[cc] * line[cc]; res[r] = s; }
                    for (int kz = 0; kz < nz; ++kz) out.a[base + (size_t)kz * zstride] += b * res[kz];
                }
        }
    }

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

    void solve(const Field3D& rhs, Field3D& x, double a, double b) const {
        const size_t N = rhs.size();
        for (size_t i = 0; i < N; ++i) work[i] = rhs.a[i];
        fft_xy(0, false); fft_xy(1, false);
        const size_t zstride = (size_t)nx * ny;
        #pragma omp parallel
        {
            DenseMatrix M(nz);
            std::vector<double> re(nz), im(nz), sre, sim;
            #pragma omp for collapse(2) schedule(static)
            for (int ky = 0; ky < ny; ++ky)
                for (int kx = 0; kx < nx; ++kx) {
                    const double a_eff = a + b * (lx[kx] + ly[ky]);
                    const size_t base = (size_t)ky * nx + kx;
                    if (std::fabs(a_eff) < 1e-12) {        // pure-Poisson null mode (constant)
                        for (int kz = 0; kz < nz; ++kz) work[base + (size_t)kz * zstride] = cdbl(0.0, 0.0);
                        continue;
                    }
                    for (int i = 0; i < nz; ++i)
                        for (int j = 0; j < nz; ++j) M(i, j) = b * Lz(i, j) + (i == j ? a_eff : 0.0);
                    for (int kz = 0; kz < nz; ++kz) {
                        const cdbl v = work[base + (size_t)kz * zstride];
                        re[kz] = v.real(); im[kz] = v.imag();
                    }
                    dense_lu_solve(M, re, sre);
                    dense_lu_solve(M, im, sim);
                    for (int kz = 0; kz < nz; ++kz) work[base + (size_t)kz * zstride] = cdbl(sre[kz], sim[kz]);
                }
        }
        fft_xy(0, true); fft_xy(1, true);
        for (size_t i = 0; i < N; ++i) x.a[i] = work[i].real();
    }
};

#endif // BMF3D_H
