// mms_poisson.cpp
//
// MMS for the separable solver (poisson.h). Because the FFT diagonalization
// inverts exactly the discrete operator used to build the right-hand side, the
// round trip returns to MACHINE PRECISION (not just an order of convergence):
//
//   pick phi_exact -> rhs = L phi_exact -> solve L phi = rhs -> phi == phi_exact.
//
// Both the pure Poisson solve (a=0, b=1) and the implicit-viscous Helmholtz
// solve (I - alpha L) are checked, in 3D and 2D.

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "poisson.h"

static const double PI = 3.14159265358979323846;

static void fill_mode(Field3D& f, const SimConfig& c, int kx, int ky, int kz) {
    const double hx = c.ax[0].L / c.ax[0].N, hy = c.ax[1].L / c.ax[1].N;
    const double hz = c.active(2) ? c.ax[2].L / c.ax[2].N : 1.0;
    for (int k = 0; k < f.nz; ++k)
        for (int j = 0; j < f.ny; ++j)
            for (int i = 0; i < f.nx; ++i) {
                double v = std::cos(kx * i * hx) * std::cos(ky * j * hy);
                if (c.active(2)) v *= std::cos(kz * k * hz);
                f.a[f.idx(i, j, k)] = v;
            }
}
static double max_diff(const Field3D& a, const Field3D& b) {
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i) e = std::fmax(e, std::fabs(a.a[i] - b.a[i]));
    return e;
}

// Channel geometry: x,y periodic, z a clamped wall (Greville nodes).
static void test_channel(const char* label, SimConfig c, int kx, int ky) {
    const char* why = nullptr;
    if (!c.validate(why)) { std::printf("  %-22s config invalid: %s\n", label, why); return; }
    WallPoisson wp;
    if (!wp.build(c, why)) { std::printf("  %-22s solver setup failed: %s\n", label, why); return; }

    const int nx = c.ax[0].N, ny = c.ax[1].N, nz = c.ax[2].N;
    const double hx = c.ax[0].L / nx, hy = c.ax[1].L / ny;
    std::vector<double> knots = open_uniform_knots(nz, c.p, c.ax[2].L);
    std::vector<double> zn = greville_points(knots, c.p, nz);

    Field3D phi(nx, ny, nz), rhs(nx, ny, nz), lap(nx, ny, nz), tmp(nx, ny, nz), out(nx, ny, nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                phi.a[phi.idx(i, j, k)] = std::cos(kx * i * hx) * std::cos(ky * j * hy)
                                        * std::sin(1.3 * zn[k] + 0.2);
    Deriv3D D(c);
    D.laplacian_any(phi, lap, tmp);

    // pure Poisson (kx,ky >= 1 so no constant x,y mode)
    for (size_t i = 0; i < rhs.size(); ++i) rhs.a[i] = lap.a[i];
    wp.solve(rhs, out, 0.0, 1.0);
    const double e_poisson = max_diff(out, phi);
    // Helmholtz
    const double alpha = 0.37;
    for (size_t i = 0; i < rhs.size(); ++i) rhs.a[i] = phi.a[i] - alpha * lap.a[i];
    wp.solve(rhs, out, 1.0, -alpha);
    const double e_helm = max_diff(out, phi);

    std::printf("  %-22s Poisson %.2e   Helmholtz %.2e\n", label, e_poisson, e_helm);
}

static void test(const char* label, SimConfig c, int kx, int ky, int kz) {
    const char* why = nullptr;
    if (!c.validate(why)) { std::printf("  %-22s config invalid: %s\n", label, why); return; }
    SeparablePoisson ps;
    if (!ps.build(c, why)) { std::printf("  %-22s solver setup failed: %s\n", label, why); return; }

    const int nx = c.ax[0].N, ny = c.ax[1].N, nz = c.active(2) ? c.ax[2].N : 1;
    Field3D phi(nx, ny, nz), rhs(nx, ny, nz), out(nx, ny, nz), tmp(nx, ny, nz);
    fill_mode(phi, c, kx, ky, kz);

    Deriv3D D(c);
    D.laplacian_any(phi, rhs, tmp);              // rhs = L phi_exact (same operator the solver inverts)

    ps.solve(rhs, out, 0.0, 1.0);                // pure Poisson: L out = rhs
    const double e_poisson = max_diff(out, phi);

    const double alpha = 0.37;                    // implicit-viscous Helmholtz
    for (size_t i = 0; i < rhs.size(); ++i) rhs.a[i] = phi.a[i] - alpha * (rhs.a[i]); // (I - alpha L) phi
    ps.solve(rhs, out, 1.0, -alpha);
    const double e_helm = max_diff(out, phi);

    std::printf("  %-22s Poisson %.2e   Helmholtz %.2e\n", label, e_poisson, e_helm);
}

// 1D clamped Helmholtz (I - c*d^2) round-trip vs degree p, two ways:
//   collocation: solve (B - c B'') alpha = rhs  (nodal; conditioning of clamped B)
//   Galerkin:    solve (M + c S)    alpha = rhs  (coefficient space; banded SPD)
// The Galerkin mass/stiffness matrices are far better conditioned than the
// collocation B, so the Galerkin round trip stays at machine precision while the
// collocation one degrades with p. This is the keystone fix for high-p walls.
static void wall_conditioning() {
    std::printf("\n 1D clamped Helmholtz round-trip vs degree (the wall conditioning fix)\n");
    std::printf("   p    collocation (B - cB'')   Galerkin (M + cS)\n");
    const int N = 40; const double L = 2.0, c = 0.05;
    for (int p = 2; p <= 8; ++p) {
        std::vector<double> knots = open_uniform_knots(N, p, L);
        SplineOperators op = build_clamped(N, p, L);
        CoeffForms cf = assemble_forms_clamped(N, p, knots);
        std::vector<double> xe(N);
        for (int i = 0; i < N; ++i) xe[i] = std::sin(0.7 * i) + 0.4 * std::cos(0.2 * i);

        DenseMatrix Kc(N);                       // collocation  B - c B''
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) Kc(i, j) = op.B(i, j) - c * op.Bpp(i, j);
        std::vector<double> rc, rhs_c = Kc.matvec(xe);
        dense_lu_solve(Kc, rhs_c, rc);
        const double ec = max_abs_diff(rc, xe);

        DenseMatrix Kg(N);                        // Galerkin  M + c S  (SPD)
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) Kg(i, j) = cf.M(i, j) + c * cf.S(i, j);
        std::vector<double> rg, rhs_g = Kg.matvec(xe);
        dense_lu_solve(Kg, rhs_g, rg);
        const double eg = max_abs_diff(rg, xe);

        std::printf("  %2d    %.3e                %.3e\n", p, ec, eg);
    }
}

// Galerkin channel solver MMS: separable manufactured coefficients
// alpha = ax (x) ay (x) az, so rhs = (aM + bS) alpha is a sum of products of 1D
// mass/stiffness matvecs. Round trip must be machine precision at every p.
static double galerkin_channel_err(int p, double a, double b, bool dir = false) {
    const int nx = 32, ny = 32, nz = 32;
    const double Lp = 2.0 * PI, Lz = 2.0;
    SimConfig c; c.p = p;
    c.ax[0] = {nx, Lp, BC::Periodic, true};
    c.ax[1] = {ny, Lp, BC::Periodic, true};
    c.ax[2] = {nz, Lz, BC::Clamped, true};
    GalerkinChannelPoisson ps; const char* why = nullptr;
    if (!ps.build(c, why)) { std::printf("   setup failed: %s\n", why); return -1; }

    CoeffForms cfx = assemble_forms_periodic(nx, p, Lp / nx);
    CoeffForms cfy = assemble_forms_periodic(ny, p, Lp / ny);
    std::vector<double> knots = open_uniform_knots(nz, p, Lz);
    CoeffForms cfz = assemble_forms_clamped(nz, p, knots);

    std::vector<double> ax(nx), ay(ny), az(nz);
    for (int i = 0; i < nx; ++i) ax[i] = std::sin(0.7 * i) + 0.3 * std::cos(0.2 * i);
    for (int j = 0; j < ny; ++j) ay[j] = std::cos(0.5 * j) + 0.2 * std::sin(0.3 * j);
    for (int k = 0; k < nz; ++k) az[k] = std::sin(0.4 * k + 0.1);
    if (dir) { az[0] = az[nz - 1] = 0.0; }            // no-slip: zero boundary z-coefficients
    if (a == 0.0) {
        // Pure Neumann Poisson: the operator kernel is the GLOBAL constant,
        // so the exact solution is fixed by the zero-mean gauge. Give az an
        // M_z-weighted zero mean; the (kx,ky)=(0,0) horizontally-averaged
        // profile then carries O(1) signal in the solver's own gauge and MUST
        // round-trip. (Until 2026-07-13 the solver ZEROED that whole z-line,
        // and this test hid it by killing the constant x,y modes of the
        // manufactured solution; this construction now gates the
        // kernel-restricted null-mode solve.)
        std::vector<double> wz(nz, 0.0);
        for (int j = 0; j < nz; ++j) for (int i = 0; i < nz; ++i) wz[j] += cfz.M(i, j);
        double sw = 0.0, sa = 0.0;
        for (int k = 0; k < nz; ++k) { sw += wz[k]; sa += wz[k] * az[k]; }
        for (double& v : az) v -= sa / sw;
    }
    std::vector<double> Max = matvec(cfx.M, ax), Sax = matvec(cfx.S, ax);
    std::vector<double> May = matvec(cfy.M, ay), Say = matvec(cfy.S, ay);
    std::vector<double> Maz = matvec(cfz.M, az), Saz = matvec(cfz.S, az);

    Field3D rhs(nx, ny, nz), exact(nx, ny, nz), out(nx, ny, nz);
    for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        exact.a[exact.idx(i, j, k)] = ax[i] * ay[j] * az[k];
        rhs.a[rhs.idx(i, j, k)] = a * (Max[i] * May[j] * Maz[k])
            + b * (Sax[i] * May[j] * Maz[k] + Max[i] * Say[j] * Maz[k] + Max[i] * May[j] * Saz[k]);
    }
    ps.solve(rhs, out, a, b, dir);
    return max_diff(out, exact);
}

int main() {
    std::printf("=================================================================\n");
    std::printf(" MMS for the separable Poisson / Helmholtz solver (round trip)\n");
    std::printf(" exact inverse -> expect machine precision\n");
    std::printf("=================================================================\n\n");
    const double TwoPi = 2.0 * PI;

    {   // 3D triperiodic, N = 32 (power of two)
        SimConfig c; c.p = 4;
        for (int a = 0; a < 3; ++a) c.ax[a] = {32, TwoPi, BC::Periodic, true};
        test("3D periodic 32^3", c, 2, 3, 1);
    }
    {   // 3D triperiodic, higher degree
        SimConfig c; c.p = 6;
        for (int a = 0; a < 3; ++a) c.ax[a] = {32, TwoPi, BC::Periodic, true};
        test("3D periodic p=6", c, 3, 1, 2);
    }
    {   // 2D periodic, N = 64
        SimConfig c; c.p = 4;
        c.ax[0] = {64, TwoPi, BC::Periodic, true};
        c.ax[1] = {64, TwoPi, BC::Periodic, true};
        c.ax[2] = {1, 1.0, BC::Periodic, false};
        test("2D periodic 64^2", c, 3, 5, 0);
    }
    {   // robustness: a non-power-of-two periodic axis must be refused
        SimConfig c; c.p = 4;
        for (int a = 0; a < 3; ++a) c.ax[a] = {48, TwoPi, BC::Periodic, true};
        const char* why = nullptr; SeparablePoisson ps;
        std::printf("  %-22s %s\n", "N=48 (not pow2)",
                    ps.build(c, why) ? "accepted (unexpected!)" : (std::string("refused: ") + why).c_str());
    }

    std::printf("\n channel: x,y periodic (FFT), z clamped wall (banded solve)\n");
    std::printf("   (error grows with p = conditioning of clamped COLLOCATION B; the\n");
    std::printf("    natural-metric BMG / SBP-flux BMF wall operators are the fix.)\n");
    for (int p : {2, 3, 4, 5, 6}) {
        SimConfig c; c.p = p;
        c.ax[0] = {32, TwoPi, BC::Periodic, true};
        c.ax[1] = {32, TwoPi, BC::Periodic, true};
        c.ax[2] = {32, 2.0, BC::Clamped, true};
        char lbl[32]; std::snprintf(lbl, sizeof(lbl), "channel 32^3 p=%d", p);
        test_channel(lbl, c, 2, 3);
    }

    wall_conditioning();

    std::printf("\n Galerkin channel solver (coefficient space, M & S) -- well-conditioned at high p\n");
    std::printf("   (p=8 ~1e-10 vs collocation/dense-BMF ~1e-4: ~5 orders better)\n");
    std::printf("   p    Poisson(Neumann)  Helmholtz(Neumann)  Helmholtz(Dirichlet/no-slip)\n");
    for (int p : {2, 4, 6, 8}) {
        const double ep = galerkin_channel_err(p, 0.0, 1.0);
        const double eh = galerkin_channel_err(p, 1.0, 0.3);
        const double ed = galerkin_channel_err(p, 1.0, 0.3, true);
        std::printf("  %2d    %.3e         %.3e           %.3e\n", p, ep, eh, ed);
    }

    std::printf("\nDone.\n");
    return 0;
}
