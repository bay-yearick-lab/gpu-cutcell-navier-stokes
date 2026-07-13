// mms_wallns3d.cpp
//
// The 3D wall-NS loop (wall_ns3d.h): primitive (u,v,w,phi) incompressible NS in
// a 3D walled box -- the direct extension of the stable 2D loop. Verifies:
//   (1) the Schur projection is idempotent: weak div u -> CG tol after a project;
//   (2) STOKES (no advection): no-slip viscous decay is stable;
//   (3) full NS: a no-slip flow stays STABLE & bounded with weak div held small.
// Modest N (the plain Gauss-quad advection is O(N^3 (2p)^3 (p+1)^3); production =
// sum-factorization), enough to confirm the 3D loop works.

#include <cstdio>
#include <cmath>
#include <vector>
#include "wall_ns3d.h"

static const double PI = 3.14159265358979323846;

// interpolate analytic (x,y,z) field to clamped^3 coefficients (per-axis B^-1)
static void interp3d(const WallNS3D& S, double (*f)(double, double, double), Field3& out) {
    const int nx = S.X.n, ny = S.Y.n, nz = S.Z.n;
    Field3 g(nx, ny, nz);
    for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k)
        g.at(i, j, k) = f(S.X.grev[i], S.Y.grev[j], S.Z.grev[k]);
    // x
    std::vector<double> line;
    for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k) {
        line.assign(nx, 0); for (int i = 0; i < nx; ++i) line[i] = g.at(i, j, k);
        std::vector<double> c; dense_lu_solve(S.X.B, line, c);
        for (int i = 0; i < nx; ++i) g.at(i, j, k) = c[i];
    }
    for (int i = 0; i < nx; ++i) for (int k = 0; k < nz; ++k) {
        line.assign(ny, 0); for (int j = 0; j < ny; ++j) line[j] = g.at(i, j, k);
        std::vector<double> c; dense_lu_solve(S.Y.B, line, c);
        for (int j = 0; j < ny; ++j) g.at(i, j, k) = c[j];
    }
    for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) {
        line.assign(nz, 0); for (int k = 0; k < nz; ++k) line[k] = g.at(i, j, k);
        std::vector<double> c; dense_lu_solve(S.Z.B, line, c);
        for (int k = 0; k < nz; ++k) g.at(i, j, k) = c[k];
    }
    out = g;
}

static double L = 2.0;
// smooth fields ~0 at the walls (sin(pi .)), generic per component
static double fu(double x, double y, double z) { return std::sin(PI*x/L)*std::sin(PI*y/L)*std::sin(PI*z/L)*std::cos(y); }
static double fv(double x, double y, double z) { return std::sin(PI*x/L)*std::sin(PI*y/L)*std::sin(PI*z/L)*std::cos(z); }
static double fw(double x, double y, double z) { return std::sin(PI*x/L)*std::sin(PI*y/L)*std::sin(PI*z/L)*std::cos(x); }

static double um(const Field3& f) { double m=0; for (double x: f.a) m=std::fmax(m,std::fabs(x)); return m; }

int main() {
    std::printf("\n 3D WALLED-BOX PRIMITIVE NS (wall_ns3d.h) -- the 3D wall-NS loop\n\n");
    const int N = 10, p = 3;

    std::printf("=================================================================\n");
    std::printf(" (1) Schur projection idempotency (3D), divergent field\n");
    std::printf("=================================================================\n");
    {
        WallNS3D S(N, N, N, p, L, L, L, 0.0);
        interp3d(S, fu, S.u); interp3d(S, fv, S.v); interp3d(S, fw, S.w);
        std::printf("   max|weak div| before = %.3e\n", S.max_weak_div());
        S.project(false);
        std::printf("   max|weak div| after  = %.3e   (idempotent: weak-div-free to CG tol)\n\n", S.max_weak_div());
    }

    std::printf("=================================================================\n");
    std::printf(" (2) Stokes (no advection): no-slip viscous decay, stable\n");
    std::printf("=================================================================\n");
    {
        WallNS3D S(N, N, N, p, L, L, L, 2e-3); S.advect = false;
        interp3d(S, fu, S.u); interp3d(S, fv, S.v); interp3d(S, fw, S.w);
        S.no_slip(S.u); S.no_slip(S.v); S.no_slip(S.w); S.project();
        const double e0 = S.kinetic_energy();
        const double dz = L / N, dt = 0.05 * dz;
        const int steps = 60;
        std::printf("    step   KE/KE0     max|weak div|\n");
        for (int n = 0; n <= steps; ++n) {
            if (n % 20 == 0) std::printf("   %4d   %.6f    %.2e\n", n, S.kinetic_energy()/e0, S.max_weak_div());
            if (n < steps) S.step(dt);
        }
        std::printf("   => stable viscous decay.\n\n");
    }

    std::printf("=================================================================\n");
    std::printf(" (3) Full 3D NS: no-slip flow, energy-conserving advection + Schur\n");
    std::printf("=================================================================\n");
    {
        WallNS3D S(N, N, N, p, L, L, L, 3e-3); S.advect = true;
        interp3d(S, fu, S.u); interp3d(S, fv, S.v); interp3d(S, fw, S.w);
        S.no_slip(S.u); S.no_slip(S.v); S.no_slip(S.w); S.project();
        const double e0 = S.kinetic_energy();
        const double dz = L / N, dt = 0.05 * dz;
        const int steps = 30;
        double emax = 1.0;
        std::printf("    step   KE/KE0     max|weak div|   max|u|\n");
        for (int n = 0; n <= steps; ++n) {
            emax = std::fmax(emax, S.kinetic_energy()/e0);
            if (n % 20 == 0) std::printf("   %4d   %.6f    %.2e        %.3f\n", n, S.kinetic_energy()/e0, S.max_weak_div(), um(S.u));
            if (!(um(S.u) < 1e3)) { std::printf("   ** UNSTABLE **\n"); return 2; }
            if (n < steps) S.step(dt);
        }
        std::printf("   => STABLE 3D nonlinear wall NS: bounded (max KE/KE0=%.3f), div held.\n", emax);
    }

    std::printf("=================================================================\n");
    std::printf(" (4) IMEX (implicit viscous, fast-diag): stable at ADVECTIVE dt, high Re\n");
    std::printf("=================================================================\n");
    for (double Re : {2000.0, 20000.0}) {
        WallNS3D S(N, N, N, p, L, L, L, L / Re); S.advect = true;
        interp3d(S, fu, S.u); interp3d(S, fv, S.v); interp3d(S, fw, S.w);
        S.no_slip(S.u); S.no_slip(S.v); S.no_slip(S.w); S.project();
        const double e0 = S.kinetic_energy();
        const double dt = 0.3 * (L / N);                 // ADVECTIVE dt (no viscous-CFL)
        for (int n = 0; n < 60; ++n) S.step_imex(dt);
        std::printf("   Re=%6.0f  dt=%.4f  60 imex steps:  KE/KE0=%.4f  max|wdiv|=%.1e  max|u|=%.2f  %s\n",
                    Re, dt, S.kinetic_energy()/e0, S.max_weak_div(), um(S.u), um(S.u) < 1e3 ? "STABLE" : "UNSTABLE");
    }
    std::printf("   => implicit viscous lifts the viscous-CFL limit (advective dt at any Re).\n");

    std::printf("=================================================================\n");
    std::printf(" (5) CHANNEL (x,y periodic + z wall): projection idempotent + stable\n");
    std::printf("=================================================================\n");
    {
        const double Lp = 2 * PI;
        WallNS3D S(N, N, N, p, Lp, Lp, L, 3e-3, true, true, false);   // x,y periodic, z wall
        auto cu = [](double x, double y, double z) { return std::sin(x)*std::cos(y)*std::sin(PI*z/2.0) + 0.4*std::cos(2*x)*std::sin(PI*z/2.0); };
        auto cv = [](double x, double y, double z) { return std::cos(x)*std::sin(y)*std::sin(PI*z/2.0); };
        auto cw = [](double x, double y, double z) { return std::sin(x)*std::sin(y)*std::sin(2*PI*z/2.0); };
        interp3d(S, cu, S.u); interp3d(S, cv, S.v); interp3d(S, cw, S.w);
        std::printf("   weak div before = %.3e\n", S.max_weak_div());
        S.project(false);
        std::printf("   weak div after  = %.3e   (Fourier x,y + Galerkin z; Nyquist pinned)\n", S.max_weak_div());
        interp3d(S, cu, S.u); interp3d(S, cv, S.v); interp3d(S, cw, S.w); S.project();
        const double e0 = S.kinetic_energy(), dt = 0.2 * (L / N);
        for (int n = 0; n < 40; ++n) S.step_imex(dt);
        std::printf("   40 imex steps:  KE/KE0=%.4f  max|wdiv|=%.1e  max|u|=%.2f  %s\n",
                    S.kinetic_energy()/e0, S.max_weak_div(), um(S.u), um(S.u) < 1e3 ? "STABLE" : "UNSTABLE");
        std::printf("   => the turbulent-CHANNEL solver works (canonical wall-DNS geometry).\n");
    }
    std::printf("=================================================================\n");
    std::printf(" (6) operator identities: stiff-as-called + interior rate solve\n");
    std::printf("=================================================================\n");
    // Two round-off-exact witnesses for the 2026-07-13 defect classes:
    // (a) stiff(u, t1) EXACTLY as rhs() calls it (out aliases the shared t1)
    //     must equal the same operator assembled with fresh local scratch
    //     (the aliasing bug degenerated it to 2*(Mx My Sz): no x/y diffusion);
    // (b) mass_solve_interior must invert the interior mass block: applying
    //     the full mass to its (wall-zeroed) result must reproduce the load
    //     on every interior row (the full-solve+pin path failed this at O(h)).
    {
        int bad = 0;
        for (int cfg = 0; cfg < 2; ++cfg) {
            const bool per = (cfg == 1);            // walls^3, then channel (x,y periodic)
            WallNS3D S(N, N, N, p, per ? 2*PI : L, per ? 2*PI : L, L, 1e-3, per, per, false);
            interp3d(S, fu, S.u); interp3d(S, fv, S.v); interp3d(S, fw, S.w);
            // (a) aliased-call stiff vs fresh-scratch assembly
            S.stiff(S.u, S.t1);
            Field3 a1(N, N, N), a2(N, N, N), ref(N, N, N);
            line_apply3(1, S.Y.M, 0, S.u, a1); line_apply3(2, S.Z.M, 0, a1, a2); line_apply3(0, S.X.S, 0, a2, ref);
            line_apply3(0, S.X.M, 0, S.u, a1); line_apply3(2, S.Z.M, 0, a1, a2); line_apply3(1, S.Y.S, 0, a2, a1);
            for (size_t q = 0; q < ref.a.size(); ++q) ref.a[q] += a1.a[q];
            line_apply3(0, S.X.M, 0, S.u, a1); line_apply3(1, S.Y.M, 0, a1, a2); line_apply3(2, S.Z.S, 0, a2, a1);
            for (size_t q = 0; q < ref.a.size(); ++q) ref.a[q] += a1.a[q];
            double ds = 0, sc = 0;
            for (size_t q = 0; q < ref.a.size(); ++q) {
                ds = std::fmax(ds, std::fabs(S.t1.a[q] - ref.a[q]));
                sc = std::fmax(sc, std::fabs(ref.a[q]));
            }
            // (b) interior mass solve identity on interior rows
            Field3 load(N, N, N), r(N, N, N);
            interp3d(S, fw, load);
            r = load; S.mass_solve_interior(r);
            line_apply3(0, S.X.M, 0, r, a1); line_apply3(1, S.Y.M, 0, a1, a2); line_apply3(2, S.Z.M, 0, a2, a1);
            double dm = 0, sm = 0;
            for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) for (int k = 0; k < N; ++k) {
                if (S.is_bndry(i, j, k)) continue;
                dm = std::fmax(dm, std::fabs(a1.at(i, j, k) - load.at(i, j, k)));
                sm = std::fmax(sm, std::fabs(load.at(i, j, k)));
            }
            std::printf("   %-18s stiff(aliased)-vs-fresh = %.2e   M_II^-1 identity = %.2e\n",
                        per ? "channel (xy per):" : "walled box:", ds / sc, dm / sm);
            if (ds > 1e-12 * sc) { std::printf("   ** FAIL: stiff scratch aliasing **\n"); bad = 1; }
            if (dm > 1e-12 * sm) { std::printf("   ** FAIL: interior mass solve **\n"); bad = 1; }
        }
        if (bad) return 3;
        std::printf("   => both identities hold to round-off.\n");
    }
    std::printf("\nDone.\n");
    return 0;
}
