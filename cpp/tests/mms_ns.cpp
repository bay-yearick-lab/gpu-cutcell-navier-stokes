// mms_ns.cpp
//
// Method of Manufactured Solutions for the config-driven 3D vector calculus
// (gradient, divergence, Laplacian) that the incompressible Navier-Stokes layer
// is built from. The point is modularity: the SAME test is run across very
// different SimConfigs -- all-periodic, mixed periodic/clamped walls, 2D, and a
// different route (BMG) -- and every one must converge at its formal order. If
// flipping a config silently broke an operator, this catches it.
//
// Manufactured field is a separable product s = fx(x) fy(y) fz(z); each factor is
// periodic (sin at an integer wavenumber) on a periodic axis and a generic smooth
// sine on a clamped axis. Exact grad/div/Laplacian follow by the product rule.

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "bm3d.h"
#include "bspline.h"      // node coordinates (Greville for clamped)

static const double PI = 3.14159265358979323846;

// Per-axis sampled factor and its first two derivatives at the operator's nodes.
struct AxisField {
    std::vector<double> x, f, fp, fpp;
};
static AxisField make_axis(int N, double L, bool periodic, int p, int idx) {
    AxisField a;
    if (N == 1) { a.x = {0.0}; a.f = {1.0}; a.fp = {0.0}; a.fpp = {0.0}; return a; }
    if (periodic) {
        for (int i = 0; i < N; ++i) a.x.push_back(i * L / N);
    } else {
        std::vector<double> knots = open_uniform_knots(N, p, L);
        a.x = greville_points(knots, p, N);
    }
    // periodic factor: integer wavenumber so it is exactly L-periodic; clamped
    // factor: a generic non-periodic sine with a per-axis phase/frequency.
    const double w   = periodic ? (2.0 * PI * (idx + 2) / L) : (1.3 + 0.4 * idx);
    const double ph  = periodic ? 0.0 : (0.3 + 0.2 * idx);
    for (int i = 0; i < N; ++i) {
        const double xx = a.x[i];
        a.f.push_back(std::sin(w * xx + ph));
        a.fp.push_back(w * std::cos(w * xx + ph));
        a.fpp.push_back(-w * w * std::sin(w * xx + ph));
    }
    return a;
}

struct Err { double grad, div, lap; };
static Err run_once(const SimConfig& c) {
    const int nx = c.ax[0].N, ny = c.ax[1].N, nz = c.active(2) ? c.ax[2].N : 1;
    AxisField fx = make_axis(nx, c.ax[0].L, c.periodic(0), c.p, 0);
    AxisField fy = make_axis(ny, c.ax[1].L, c.periodic(1), c.p, 1);
    AxisField fz = make_axis(nz, c.active(2) ? c.ax[2].L : 1.0,
                             c.active(2) && c.periodic(2), c.p, 2);

    Field3D s(nx, ny, nz), gx(nx, ny, nz), gy(nx, ny, nz), gz(nx, ny, nz),
            dv(nx, ny, nz), lap(nx, ny, nz), tmp(nx, ny, nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                s.a[s.idx(i, j, k)] = fx.f[i] * fy.f[j] * fz.f[k];

    Deriv3D D(c);
    D.grad(s, gx, gy, gz);
    D.divergence(s, s, s, dv, tmp);              // vx = vy = vz = s
    D.laplacian_any(s, lap, tmp);

    Err e{0, 0, 0};
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t id = s.idx(i, j, k);
                const double ex_gx = fx.fp[i] * fy.f[j]  * fz.f[k];
                const double ex_gy = fx.f[i]  * fy.fp[j] * fz.f[k];
                const double ex_gz = fx.f[i]  * fy.f[j]  * fz.fp[k];
                const double ex_dv = ex_gx + ex_gy + ex_gz;
                const double ex_lap = fx.fpp[i] * fy.f[j] * fz.f[k]
                                    + fx.f[i] * fy.fpp[j] * fz.f[k]
                                    + fx.f[i] * fy.f[j] * fz.fpp[k];
                e.grad = std::fmax(e.grad, std::fabs(gx.a[id] - ex_gx));
                e.grad = std::fmax(e.grad, std::fabs(gy.a[id] - ex_gy));
                e.grad = std::fmax(e.grad, std::fabs(gz.a[id] - ex_gz));
                e.div = std::fmax(e.div, std::fabs(dv.a[id] - ex_dv));
                e.lap = std::fmax(e.lap, std::fabs(lap.a[id] - ex_lap));
            }
    return e;
}

// Run one config at two resolutions and report errors + observed order.
static void study(const char* label, SimConfig c, int N1, int N2) {
    const char* why = nullptr;
    auto scale = [](SimConfig cc, int N) {
        for (int a = 0; a < 3; ++a) if (cc.ax[a].on) cc.ax[a].N = N;
        return cc;
    };
    SimConfig c1 = scale(c, N1), c2 = scale(c, N2);
    if (!c1.validate(why) || !c2.validate(why)) { std::printf("  %-26s INVALID: %s\n", label, why); return; }
    Err e1 = run_once(c1), e2 = run_once(c2);
    const double r = std::log((double)N2 / N1);
    auto ord = [&](double a, double b) { return (a > 1e-13 && b > 1e-13) ? std::log(a / b) / r : 0.0; };
    std::printf("  %-26s grad %.2e (ord %.1f)  div %.2e (ord %.1f)  lap %.2e (ord %.1f)\n",
                label, e2.grad, ord(e1.grad, e2.grad), e2.div, ord(e1.div, e2.div),
                e2.lap, ord(e1.lap, e2.lap));
}

int main() {
    std::printf("=================================================================\n");
    std::printf(" MMS for the config-driven grad / div / Laplacian (modularity)\n");
    std::printf("=================================================================\n");
    std::printf(" formal order: p=%d -> bounded ~p, periodic >= p (errors at the finer grid)\n\n", 4);

    const double TwoPi = 2.0 * PI;

    // 1. fully periodic cube, BMC.
    {
        SimConfig c; c.p = 4; c.route = Route::BMC;
        for (int a = 0; a < 3; ++a) { c.ax[a] = {32, TwoPi, BC::Periodic, true}; }
        study("3D periodic BMC", c, 32, 48);
    }
    // 2. periodic x,y + clamped wall in z (a channel), BMC.
    {
        SimConfig c; c.p = 4; c.route = Route::BMC;
        c.ax[0] = {32, TwoPi, BC::Periodic, true};
        c.ax[1] = {32, TwoPi, BC::Periodic, true};
        c.ax[2] = {32, 2.0,   BC::Clamped,  true};
        study("3D channel (z wall) BMC", c, 32, 48);
    }
    // 3. clamped box in all three axes, BMC.
    {
        SimConfig c; c.p = 4; c.route = Route::BMC;
        for (int a = 0; a < 3; ++a) { c.ax[a] = {32, 2.0, BC::Clamped, true}; }
        study("3D walled box BMC", c, 32, 48);
    }
    // 4. 2D periodic (z inactive), BMC.
    {
        SimConfig c; c.p = 4; c.route = Route::BMC;
        c.ax[0] = {32, TwoPi, BC::Periodic, true};
        c.ax[1] = {32, TwoPi, BC::Periodic, true};
        c.ax[2] = {1, 1.0, BC::Periodic, false};       // inactive third axis
        study("2D periodic BMC", c, 48, 64);
    }
    // 5. fully periodic cube, BMG route (switch the first-derivative route).
    {
        SimConfig c; c.p = 4; c.route = Route::BMG;
        for (int a = 0; a < 3; ++a) { c.ax[a] = {32, TwoPi, BC::Periodic, true}; }
        study("3D periodic BMG", c, 32, 48);
    }
    // 6. robustness: a deliberately bad config must be rejected, not crash.
    {
        SimConfig c; c.p = 6;
        c.ax[0] = {5, TwoPi, BC::Periodic, true};      // N too small for p=6
        c.ax[1] = {32, TwoPi, BC::Periodic, true};
        c.ax[2] = {32, TwoPi, BC::Periodic, true};
        const char* why = nullptr;
        std::printf("  %-26s %s\n", "bad config (N=5,p=6)",
                    c.validate(why) ? "accepted (unexpected!)" : (std::string("rejected: ") + why).c_str());
    }

    std::printf("\nDone.\n");
    return 0;
}
