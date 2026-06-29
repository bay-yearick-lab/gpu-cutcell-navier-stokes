// mms_bmf.cpp
//
// MMS for the 3D B-spline Mimetic Flux Laplacian (bmf3d.h) on WALL-bounded
// domains -- the staggered scheme. The BMF Laplacian is Dv.G summed over axes
// (divergence of the gradient routed through the shortened derivative grid). It
// is the wall-appropriate, well-conditioned Laplacian (the collocation B''B^{-1}
// that the channel solver exposed degrades at high p; this is the fix).
//
// A separable manufactured field phi = fx(x) fy(y) fz(z) is sampled on the operator
// nodes (Greville on a wall, uniform on a periodic axis) and the discrete BMF
// Laplacian is compared with the exact phi_xx + phi_yy + phi_zz, refined to read
// the order. Configs: a fully walled box and a channel (periodic x,y + wall z).

#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include "bmf3d.h"
#include "bspline.h"

static const double PI = 3.14159265358979323846;

// Per-axis factor f and its second derivative at the operator's nodes.
struct AxisField { std::vector<double> x, f, fpp; };
static AxisField make_axis(int N, double L, bool periodic, int p, int idx) {
    AxisField a;
    if (periodic) for (int i = 0; i < N; ++i) a.x.push_back(i * L / N);
    else { std::vector<double> kn = open_uniform_knots(N, p, L); a.x = greville_points(kn, p, N); }
    const double w  = periodic ? (2.0 * PI * (idx + 2) / L) : (1.3 + 0.4 * idx);
    const double ph = periodic ? 0.0 : (0.3 + 0.2 * idx);
    for (int i = 0; i < N; ++i) {
        a.f.push_back(std::sin(w * a.x[i] + ph));
        a.fpp.push_back(-w * w * std::sin(w * a.x[i] + ph));
    }
    return a;
}

static double laplacian_error(const SimConfig& c) {
    const int nx = c.ax[0].N, ny = c.ax[1].N, nz = c.active(2) ? c.ax[2].N : 1;
    AxisField fx = make_axis(nx, c.ax[0].L, c.periodic(0), c.p, 0);
    AxisField fy = make_axis(ny, c.ax[1].L, c.periodic(1), c.p, 1);
    AxisField fz;
    if (c.active(2)) fz = make_axis(nz, c.ax[2].L, c.periodic(2), c.p, 2);
    else { fz.f = {1.0}; fz.fpp = {0.0}; }

    Field3D phi(nx, ny, nz), lap(nx, ny, nz), tmp(nx, ny, nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                phi.a[phi.idx(i, j, k)] = fx.f[i] * fy.f[j] * fz.f[k];

    BMFDeriv3D D(c);
    D.laplacian(phi, lap, tmp);

    double err = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double ex = fx.fpp[i] * fy.f[j] * fz.f[k]
                                + fx.f[i] * fy.fpp[j] * fz.f[k]
                                + fx.f[i] * fy.f[j] * fz.fpp[k];
                err = std::fmax(err, std::fabs(lap.a[phi.idx(i, j, k)] - ex));
            }
    return err;
}

// Cross-check the separate staggered grad and div: div(grad phi) must equal the
// composite BMF Laplacian Dv.G to round-off (verifies the rectangular apply).
static double graddiv_consistency(const SimConfig& c) {
    const int nx = c.ax[0].N, ny = c.ax[1].N, nz = c.active(2) ? c.ax[2].N : 1;
    AxisField fx = make_axis(nx, c.ax[0].L, c.periodic(0), c.p, 0);
    AxisField fy = make_axis(ny, c.ax[1].L, c.periodic(1), c.p, 1);
    AxisField fz; if (c.active(2)) fz = make_axis(nz, c.ax[2].L, c.periodic(2), c.p, 2);
    else { fz.f = {1.0}; }

    Field3D phi(nx, ny, nz), lap1(nx, ny, nz), lap2(nx, ny, nz), tmp(nx, ny, nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                phi.a[phi.idx(i, j, k)] = fx.f[i] * fy.f[j] * fz.f[k];

    BMFDeriv3D D(c);
    D.laplacian(phi, lap1, tmp);                          // composite Dv.G
    Field3D gx = D.make_stag(0), gy = D.make_stag(1), gz = D.make_stag(2);
    D.grad(phi, gx, gy, gz);                              // primal -> staggered
    D.divergence(gx, gy, gz, lap2, tmp);                 // staggered -> primal
    double e = 0.0;
    for (size_t i = 0; i < lap1.size(); ++i) e = std::fmax(e, std::fabs(lap1.a[i] - lap2.a[i]));
    return e;
}

static void study(const char* label, SimConfig base, int N1, int N2) {
    auto scale = [](SimConfig c, int N) {
        for (int a = 0; a < 3; ++a) if (c.ax[a].on) c.ax[a].N = N;
        return c;
    };
    const char* why = nullptr;
    SimConfig c1 = scale(base, N1), c2 = scale(base, N2);
    if (!c1.validate(why) || !c2.validate(why)) { std::printf("  %-26s invalid: %s\n", label, why); return; }
    const double e1 = laplacian_error(c1), e2 = laplacian_error(c2);
    const double ord = std::log(e1 / e2) / std::log((double)N2 / N1);
    std::printf("  %-26s  err(N=%d) %.3e   order %.1f\n", label, N2, e2, ord);
}

int main() {
    std::printf("=================================================================\n");
    std::printf(" MMS for the 3D BMF (staggered flux) Laplacian on wall boundaries\n");
    std::printf(" L = sum_a Dv_a G_a ;  expect order ~ p-1 (wall-row cap)\n");
    std::printf("=================================================================\n\n");
    const double TwoPi = 2.0 * PI;

    for (int p : {3, 4, 5}) {
        SimConfig c; c.p = p;
        for (int a = 0; a < 3; ++a) c.ax[a] = {32, 2.0, BC::Clamped, true};
        char lbl[40]; std::snprintf(lbl, sizeof(lbl), "walled box p=%d", p);
        study(lbl, c, 24, 36);
    }
    std::printf("\n");
    for (int p : {3, 4, 5}) {
        SimConfig c; c.p = p;
        c.ax[0] = {32, TwoPi, BC::Periodic, true};
        c.ax[1] = {32, TwoPi, BC::Periodic, true};
        c.ax[2] = {32, 2.0,   BC::Clamped,  true};
        char lbl[40]; std::snprintf(lbl, sizeof(lbl), "channel (z wall) p=%d", p);
        study(lbl, c, 24, 36);
    }
    std::printf("\n staggered grad/div cross-check: ||div(grad phi) - (Dv.G) phi||\n");
    for (int p : {3, 4, 5}) {
        SimConfig c; c.p = p;
        for (int a = 0; a < 3; ++a) c.ax[a] = {28, 2.0, BC::Clamped, true};
        std::printf("   walled box p=%d:  %.2e\n", p, graddiv_consistency(c));
    }

    std::printf("\n BMF channel Poisson/Helmholtz (FFT x,y + per-mode Dv_z G_z solve)\n");
    std::printf("   usable at moderate p; high-p limited by cond of the DENSE-inverse\n");
    std::printf("   BMF operator (matrix-free banded reduce-solve-expand = the fix)\n");
    for (int p : {3, 4, 5, 6}) {
        SimConfig c; c.p = p;
        c.ax[0] = {32, TwoPi, BC::Periodic, true};
        c.ax[1] = {32, TwoPi, BC::Periodic, true};
        c.ax[2] = {32, 2.0,   BC::Clamped,  true};
        const int nx = 32, ny = 32, nz = 32;
        AxisField fx = make_axis(nx, TwoPi, true, p, 0);
        AxisField fy = make_axis(ny, TwoPi, true, p, 1);
        AxisField fz = make_axis(nz, 2.0, false, p, 2);
        Field3D phi(nx, ny, nz), rhs(nx, ny, nz), out(nx, ny, nz);
        for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i)
            phi.a[phi.idx(i, j, k)] = fx.f[i] * fy.f[j] * fz.f[k];
        BMFChannelPoisson ps; const char* why = nullptr;
        if (!ps.build(c, why)) { std::printf("   p=%d setup failed: %s\n", p, why); continue; }
        ps.apply(phi, rhs, 0.0, 1.0);                                    // Poisson a=0,b=1
        ps.solve(rhs, out, 0.0, 1.0);
        double e_p = 0.0; for (size_t i = 0; i < out.size(); ++i) e_p = std::fmax(e_p, std::fabs(out.a[i] - phi.a[i]));
        const double al = 0.37;                                          // Helmholtz
        ps.apply(phi, rhs, 1.0, -al);
        ps.solve(rhs, out, 1.0, -al);
        double e_h = 0.0; for (size_t i = 0; i < out.size(); ++i) e_h = std::fmax(e_h, std::fabs(out.a[i] - phi.a[i]));
        std::printf("   channel p=%d:  Poisson %.2e   Helmholtz %.2e\n", p, e_p, e_h);
    }

    // Matrix-free banded BMF apply vs the dense operators, and the eigenvalue
    // pollution test: apply the wall Laplacian D2 = Dv(G(.)) to the roughest mode
    // and compare the magnitude (dense build vs banded). Dense-inverse formation
    // inflates the high modes; the banded chain should give a much smaller result.
    std::printf("\n Matrix-free banded BMF apply (vs dense) + eigenvalue-pollution test\n");
    std::printf("   p    ||G_band - G_dense||   ||Dv_band - Dv_dense||   max|D2 rough|: dense / banded\n");
    for (int p : {3, 4, 5, 6, 7, 8}) {
        const int N = 32; const double L = 2.0;
        BMFBanded bb; bb.build(N, p, L);
        OperatorSet os = build_bmf_bounded(N, p, L);
        std::vector<double> u(N), g(N - 1);
        for (int i = 0; i < N; ++i) u[i] = std::sin(0.7 * i) + 0.3 * std::cos(0.2 * i);
        for (int i = 0; i < N - 1; ++i) g[i] = std::cos(0.5 * i) + 0.2 * std::sin(0.3 * i);
        std::vector<double> Gb = bb.G(u), Gd = matvec(os.D1, u);
        std::vector<double> Db = bb.Dv(g), Dd = matvec(os.Dv, g);
        const double eG = max_abs_diff(Gb, Gd), eD = max_abs_diff(Db, Dd);
        std::vector<double> rough(N);                       // roughest resolvable mode
        for (int i = 0; i < N; ++i) rough[i] = (i % 2 ? 1.0 : -1.0);
        std::vector<double> d2_band = bb.Dv(bb.G(rough)), d2_dense = matvec(os.D2, rough);
        double md = 0, mb = 0;
        for (double v : d2_dense) md = std::fmax(md, std::fabs(v));
        for (double v : d2_band) mb = std::fmax(mb, std::fabs(v));
        std::printf("  %2d    %.2e               %.2e                 %.2e / %.2e\n", p, eG, eD, md, mb);
    }

    // Speed: banded G+Dv apply (O(N*bw)) vs the dense matvec (O(N^2)), large N.
    {
        const int N = 512, p = 5; const double L = 2.0; const int reps = 400;
        BMFBanded bb; bb.build(N, p, L);
        OperatorSet os = build_bmf_bounded(N, p, L);
        std::vector<double> u(N, 0.0); for (int i = 0; i < N; ++i) u[i] = std::sin(0.01 * i);
        volatile double sink = 0.0;
        clock_t t0 = clock();
        for (int r = 0; r < reps; ++r) { std::vector<double> y = bb.Dv(bb.G(u)); sink += y[N / 2]; }
        const double tb = (double)(clock() - t0) / CLOCKS_PER_SEC;
        t0 = clock();
        for (int r = 0; r < reps; ++r) { std::vector<double> y = matvec(os.D2, u); sink += y[N / 2]; }
        const double td = (double)(clock() - t0) / CLOCKS_PER_SEC;
        std::printf("\n D2 apply at N=%d p=%d (%d reps): banded %.3f s   dense %.3f s   speedup %.1fx\n",
                    N, p, reps, tb, td, td / tb);
        (void)sink;
    }

    std::printf("\nDone.\n");
    return 0;
}
