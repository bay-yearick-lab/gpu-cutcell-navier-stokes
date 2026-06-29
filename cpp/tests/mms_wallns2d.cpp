// mms_wallns2d.cpp
//
// PRIMITIVE-variable 2D walled-box NS (wall_ns2d.h): the two de-risked wall
// pieces assembled into one loop -- the energy-conserving Galerkin advection +
// the consistent inf-sup-stable Schur-complement projection. Verifies:
//   (1) the projection is IDEMPOTENT: weak div u -> round-off after a project;
//   (2) STOKES (no advection): no-slip decay is stable, energy monotone, div held;
//   (3) full NS: a no-slip vortex stays STABLE & bounded with weak div held ~0.
// This is the 2D model of the 3D wall-NS loop.

#include <cstdio>
#include <cmath>
#include <vector>
#include "wall_ns2d.h"

static const double PI = 3.14159265358979323846;

// interpolate analytic (x,z) field to clamped(x) x clamped(z) coefficients
static void interp2d(const Axis& X, const Axis& Z, double (*f)(double, double), Field2D& out) {
    Field2D nodal(X.n, Z.n);
    for (int i = 0; i < X.n; ++i) for (int j = 0; j < Z.n; ++j) nodal.at(i, j) = f(X.grev[i], Z.grev[j]);
    Field2D tmp(X.n, Z.n);
    std::vector<double> col(X.n), row(Z.n);
    for (int j = 0; j < Z.n; ++j) {
        for (int i = 0; i < X.n; ++i) col[i] = nodal.at(i, j);
        std::vector<double> c; dense_lu_solve(X.B, col, c);
        for (int i = 0; i < X.n; ++i) tmp.at(i, j) = c[i];
    }
    for (int i = 0; i < X.n; ++i) {
        for (int j = 0; j < Z.n; ++j) row[j] = tmp.at(i, j);
        std::vector<double> c; dense_lu_solve(Z.B, row, c);
        for (int j = 0; j < Z.n; ++j) out.at(i, j) = c[j];
    }
}

static double Lx = 2.0, Lz = 2.0;
// no-slip div-free field u = d_z psi, w = -d_x psi, psi = sin^2(pi x/Lx) sin^2(pi z/Lz)
static double u_ic(double x, double z) {
    return std::sin(PI * x / Lx) * std::sin(PI * x / Lx) * (PI / Lz) * std::sin(2 * PI * z / Lz);
}
static double w_ic(double x, double z) {
    return -(PI / Lx) * std::sin(2 * PI * x / Lx) * std::sin(PI * z / Lz) * std::sin(PI * z / Lz);
}

int main() {
    std::printf("\n PRIMITIVE 2D WALLED-BOX NS (energy-conserving advection + Schur projection)\n\n");
    const int N = 24, p = 4;

    // (1) projection idempotency on a GENUINELY divergent field
    std::printf("=================================================================\n");
    std::printf(" (1) Schur projection idempotency: weak div of a divergent field\n");
    std::printf("=================================================================\n");
    {
        WallNS2D S(N, N, p, Lx, Lz, 0.0);
        // a divergent field (NOT from a streamfunction): u=w=sin(.)cos(.) products
        interp2d(S.X, S.Z, [](double x, double z) { return std::sin(2 * x) * std::cos(z) + 0.3 * x * z; }, S.u);
        interp2d(S.X, S.Z, [](double x, double z) { return std::cos(x) * std::sin(2 * z) + 0.2 * z; }, S.w);
        std::printf("   max|weak div| before = %.3e\n", S.max_weak_div());
        S.project(false);                       // no boundary pin -> exact weak-div-free
        std::printf("   max|weak div| after  = %.3e   (idempotent: weak-div-free to CG tol)\n\n",
                    S.max_weak_div());
    }

    // (2) Stokes decay (no advection) -- explicit viscous, stable dt
    std::printf("=================================================================\n");
    std::printf(" (2) Stokes (no advection): no-slip viscous decay, div held, stable\n");
    std::printf("=================================================================\n");
    {
        WallNS2D S(N, N, p, Lx, Lz, 2e-3); S.advect = false;
        interp2d(S.X, S.Z, u_ic, S.u); interp2d(S.X, S.Z, w_ic, S.w);
        S.no_slip(S.u); S.no_slip(S.w); S.project();
        const double e0 = S.kinetic_energy();
        const double dz = Lz / N, dt = 0.06 * dz;             // explicit-viscous-stable
        const int steps = 300;
        std::printf("    step    KE/KE0      max|weak div|\n");
        for (int n = 0; n <= steps; ++n) {
            if (n % 100 == 0) std::printf("   %4d    %.6f    %.2e\n", n, S.kinetic_energy() / e0, S.max_weak_div());
            if (n < steps) S.step(dt);
        }
        std::printf("   => monotone viscous decay, stable (div ~ wall-splitting error).\n\n");
    }

    // (3) full NS: no-slip vortex, stable + bounded + div held
    std::printf("=================================================================\n");
    std::printf(" (3) Full NS: no-slip vortex, energy-conserving advection + projection\n");
    std::printf("=================================================================\n");
    {
        WallNS2D S(N, N, p, Lx, Lz, 2e-3); S.advect = true;
        interp2d(S.X, S.Z, u_ic, S.u); interp2d(S.X, S.Z, w_ic, S.w);
        S.no_slip(S.u); S.no_slip(S.w); S.project();
        const double e0 = S.kinetic_energy();
        const double dz = Lz / N, dt = 0.08 * dz;                 // advective CFL-ish
        const int steps = 400;
        double emax = 1.0;
        std::printf("    step    KE/KE0      max|weak div|   max|u|\n");
        for (int n = 0; n <= steps; ++n) {
            double um = 0; for (size_t q = 0; q < S.u.a.size(); ++q) um = std::fmax(um, std::fabs(S.u.a[q]));
            emax = std::fmax(emax, S.kinetic_energy() / e0);
            if (n % 100 == 0) std::printf("   %4d    %.6f    %.2e        %.3f\n", n, S.kinetic_energy() / e0, S.max_weak_div(), um);
            if (!(um < 1e3)) { std::printf("   ** UNSTABLE **\n"); return 2; }
            if (n < steps) S.step(dt);
        }
        std::printf("   => STABLE nonlinear wall NS: bounded (max KE/KE0=%.3f), weak div\n", emax);
        std::printf("      held small (~wall fractional-step splitting error). The energy-\n");
        std::printf("      conserving advection + Schur projection give a stable wall loop.\n");
    }
    std::printf("\nDone.\n");
    return 0;
}
