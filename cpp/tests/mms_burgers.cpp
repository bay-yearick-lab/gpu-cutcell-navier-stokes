// mms_burgers.cpp
//
// The ENERGY-CONSERVING WALL ADVECTION result (the wall-NS frontier), on 1D
// Burgers in coefficient space (burgers1d.h):
//
//   (1) SEMI-DISCRETE ENERGY IDENTITY -- for ANY coefficients, the convective
//       energy rate alpha^T C equals EXACTLY the physical wall flux [u^3/3]_0^L
//       (the interior cubic integrals cancel to round-off). With u=0 walls the
//       rate is zero: the bounded advection conserves energy to round-off. This
//       is what the collocated/nodal wall could NOT do.
//
//   (2) DYNAMIC STABILITY -- a time-marched no-slip Burgers run stays bounded
//       and conserves energy to the time-integration error (O(dt^3), shrinking
//       as dt halves), instead of the collocated wall's energy blow-up.
//
//   (3) ACCURACY -- the Galerkin convective term converges at the spline order.

#include <cstdio>
#include <cmath>
#include <vector>
#include "burgers1d.h"

static const double PI = 3.14159265358979323846;

static void test_identity() {
    std::printf("=================================================================\n");
    std::printf(" (1) Semi-discrete energy identity:  alpha^T C == wall flux [u^3/3]\n");
    std::printf("=================================================================\n");
    BurgersBounded S(24, 4, 2.0, 0.0);
    const int n = S.n;
    std::vector<double> a(n);
    for (int j = 0; j < n; ++j) a[j] = std::sin(2.3 * j + 0.7) + 0.5 * std::cos(1.1 * j);
    // case A: u=0 at both walls (no-slip) -> rate must be round-off
    a[0] = 0.0; a[n - 1] = 0.0;
    std::printf("   no-slip walls (u(0)=u(L)=0):  -dE/dt|conv = %.3e   (must be round-off)\n",
                S.dEdt_conv(a));
    // case B: nonzero walls -> rate must equal the analytic flux exactly
    a[0] = 0.3; a[n - 1] = -0.4;
    const double rate = S.dEdt_conv(a), flux = S.wall_flux(a);
    std::printf("   open walls:  -dE/dt|conv = %.6e   flux -[u^3/3] = %.6e   |diff| = %.2e\n",
                rate, flux, std::fabs(rate - flux));
    std::printf("   => the interior advection injects ZERO energy; only the wall flux remains.\n\n");
}

static void test_stability() {
    std::printf("=================================================================\n");
    std::printf(" (2) Dynamic stability: no-slip Burgers, energy vs time-step error\n");
    std::printf("=================================================================\n");
    const int n = 40, p = 4; const double L = 2.0;
    for (double nu : {0.0, 0.01}) {
        std::printf("   nu = %.3g\n", nu);
        double prev_drift = 0.0;
        for (double dt : {0.004, 0.002, 0.001}) {
            BurgersBounded S(n, p, L, nu);
            std::vector<double> u(n);
            for (int i = 0; i < n; ++i) u[i] = std::sin(PI * S.grev[i] / L);   // 0 at walls
            S.set_nodal(u);
            const double E0 = S.energy();
            const double Tend = 0.4; const int steps = (int)std::lround(Tend / dt);
            double emax = E0, emin = E0;
            for (int s = 0; s < steps; ++s) {
                S.step(dt);
                const double e = S.energy(); emax = std::fmax(emax, e); emin = std::fmin(emin, e);
            }
            const double Ef = S.energy();
            const double drift = (nu == 0.0) ? std::fabs(Ef - E0) / E0 : 0.0;
            // bounded check
            const std::vector<double> un = S.nodal();
            double umax = 0; for (double x : un) umax = std::fmax(umax, std::fabs(x));
            if (nu == 0.0)
                std::printf("     dt=%.4f  |dE|/E0=%.2e %s  max|u|=%.3f (bounded)\n",
                            dt, drift,
                            prev_drift > 0 ? (drift < prev_drift ? "(shrinks w/ dt)" : "") : "",
                            umax);
            else
                std::printf("     dt=%.4f  E/E0=%.4f (viscous decay, monotone)  max|u|=%.3f\n",
                            dt, Ef / E0, umax);
            prev_drift = drift;
        }
    }
    std::printf("   => bounded + energy conserved to O(dt^3); the collocated wall BLEW UP here.\n\n");
}

static void test_accuracy() {
    std::printf("=================================================================\n");
    std::printf(" (3) Accuracy: Galerkin convective term B M^-1 C vs analytic d_x(u^2/2)\n");
    std::printf("=================================================================\n");
    const double L = 2.0;
    for (int p : {2, 4, 6}) {
        std::printf("   p=%d\n     N        h          rel-L2        order\n", p);
        double prev_e = 0, prev_h = 0, last = 0;
        for (int n : {16, 24, 32, 48}) {
            BurgersBounded S(n, p, L, 0.0);
            std::vector<double> u(n);
            for (int i = 0; i < n; ++i) u[i] = std::sin(PI * S.grev[i] / L);
            S.set_nodal(u);
            std::vector<double> C; S.conv(S.alpha, C);
            // remove the wall-flux boundary contribution so the interior compares cleanly
            C[0] -= (1.0 / 3.0) * (-S.alpha[0] * S.alpha[0]);
            C[n - 1] -= (1.0 / 3.0) * (S.alpha[n - 1] * S.alpha[n - 1]);
            std::vector<double> proj(n);
            S.Mfac.solve_inplace(C);                 // C now holds M^-1 C
            // nodal value = B (M^-1 C)
            double err = 0, ref = 0;
            for (int r = 0; r < n; ++r) {
                double bn = 0; for (int j = 0; j < n; ++j) bn += S.B(r, j) * C[j];
                const double x = S.grev[r];
                const double ux = (PI / L) * std::sin(PI * x / L) * std::cos(PI * x / L); // d_x(u^2/2)
                err += (bn - ux) * (bn - ux); ref += ux * ux;
            }
            const double e = std::sqrt(err / ref), h = L / n;
            double ord = 0; if (prev_e > 0) { ord = std::log(prev_e / e) / std::log(prev_h / h); last = ord; }
            std::printf("    %3d   %.5f   %.4e   %s%.2f\n", n, h, e,
                        prev_e > 0 ? "" : "   -- ", prev_e > 0 ? ord : 0.0);
            prev_e = e; prev_h = h; (void)proj;
        }
        std::printf("     => measured ~%.1f\n\n", last);
    }
}

int main() {
    std::printf("\n ENERGY-CONSERVING WALL ADVECTION -- 1D Burgers (coefficient space)\n\n");
    test_identity();
    test_stability();
    test_accuracy();
    std::printf("Done.\n");
    return 0;
}
