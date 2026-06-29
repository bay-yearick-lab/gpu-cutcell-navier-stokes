// ns_channel_stokes.cpp
//
// COEFFICIENT-SPACE no-slip viscous time integration on a channel (the stable
// core of the wall-bounded momentum step) + the projection-coupling finding.
// State = velocity coefficients alpha; u = B alpha. Implicit no-slip viscous step
//   (M + nu dt S) u^{n+1}_a = M u^n_a            [Dirichlet, GalerkinChannelPoisson]
// is validated on the exact decaying shear mode
//   u_x = sin(m pi z/Lz) e^{-nu (m pi/Lz)^2 t}   (no-slip, div-free, zero pressure):
// the energy must follow e^{-2 lambda t} and the discrete divergence stay at
// round-off. This mode needs no pressure, so it isolates the viscous integration.
//
// FINDING (why the projection is NOT in this loop): the collocated coefficient-
// space discrete projection (ChannelProjector, pressure operator L = sum_a G_a^2) is
// exactly divergence-free for a SINGLE projection (verified in ns_channel_project)
// but UNSTABLE when iterated in a time loop. The wall pressure operator has a large
// near-null space (corner (kx,ky) modes + const/linear in z), so cond(L) is huge;
// solving L q = div for a round-off-level divergence amplifies it (||correction||^2
// = div^T L^{-1} div), the corrector injects spurious velocity, and it compounds
// (~10^13 growth in div over one step here). The weak (stiffness) Poisson is well
// conditioned but is not an idempotent projection (L != div.grad), so it is also
// unstable iterated. This is the classic inf-sup / checkerboard problem at a wall.
// The robust fix is the BMF STAGGERED route (bmf3d.h), which has no pressure null
// modes -- that is the path for the full wall-bounded NS time loop.

#include <cstdio>
#include <cmath>
#include <vector>
#include "ns_channel.h"

static const double PI = 3.14159265358979323846;

int main() {
    const int N = 32, p = 4;
    const double Lp = 2.0 * PI, Lz = 2.0, nu = 0.05;
    const int mz = 1;
    SimConfig c; c.p = p; c.nu = nu;
    c.ax[0] = {N, Lp, BC::Periodic, true};
    c.ax[1] = {N, Lp, BC::Periodic, true};
    c.ax[2] = {N, Lz, BC::Clamped, true};

    std::printf("==================================================================\n");
    std::printf(" Coefficient-space no-slip viscous time integration on a channel\n");
    std::printf(" (u = B alpha; implicit Dirichlet viscous; shear-mode decay test)\n");
    std::printf("==================================================================\n\n");
    c.print();

    GalerkinChannelPoisson visc; const char* why = nullptr;
    ChannelProjector proj;
    if (!visc.build(c, why) || !proj.build(c, why)) { std::printf("setup failed: %s\n", why); return 1; }

    FastDeriv fx = FastDeriv::build(true,  N, p, Lp);
    FastDeriv fy = FastDeriv::build(true,  N, p, Lp);
    FastDeriv fz = FastDeriv::build(false, N, p, Lz);
    Mat Mx = assemble_forms_periodic(N, p, Lp / N).M, My = Mx;
    std::vector<double> knots = open_uniform_knots(N, p, Lz);
    Mat Mz = assemble_forms_clamped(N, p, knots).M;
    std::vector<double> xz = greville_points(knots, p, N);

    Field3D u(N, N, N), v(N, N, N), w(N, N, N);   // velocity coefficients
    Field3D us(N, N, N), vs(N, N, N), ws(N, N, N), rhs(N, N, N), d(N, N, N);

    // Initial shear mode u_x = sin(m pi z/Lz); reduce to coefficients (once).
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                u.a[u.idx(i, j, k)] = std::sin(mz * PI * xz[k] / Lz);
    nc_axis_fastderiv(u, u, 0, [&](std::vector<double>& l, std::vector<double>& r){ r = l; fx.reduceB(r); });
    nc_axis_fastderiv(u, u, 1, [&](std::vector<double>& l, std::vector<double>& r){ r = l; fy.reduceB(r); });
    nc_axis_fastderiv(u, u, 2, [&](std::vector<double>& l, std::vector<double>& r){ r = l; fz.reduceB(r); });

    // Tensor mass apply M alpha -> out (the viscous-solve right-hand side).
    auto massapply = [&](const Field3D& a, Field3D& out) {
        nc_axis_mat(a, out, 0, Mx); nc_axis_mat(out, out, 1, My); nc_axis_mat(out, out, 2, Mz);
    };
    // Viscous predictor for one component: solve (M + nu dt S) a* = M a  (no-slip).
    auto viscous = [&](const Field3D& a, double dt, Field3D& astar) {
        massapply(a, rhs);
        visc.solve(rhs, astar, 1.0, nu * dt, /*dirichlet=*/true);
    };
    auto energy = [&]() {
        Field3D un(N, N, N), t(N, N, N); double e = 0.0;
        const Field3D* comp[3] = {&u, &v, &w};
        for (int cc = 0; cc < 3; ++cc) {
            nc_axis_fastderiv(*comp[cc], un, 0, [&](std::vector<double>& l, std::vector<double>& r){ fx.expandB(l, r); });
            nc_axis_fastderiv(un, un, 1, [&](std::vector<double>& l, std::vector<double>& r){ fy.expandB(l, r); });
            nc_axis_fastderiv(un, un, 2, [&](std::vector<double>& l, std::vector<double>& r){ fz.expandB(l, r); });
            for (double val : un.a) e += val * val;
        }
        return e;
    };
    auto maxdiv = [&]() { proj.divergence(u, v, w, d); double e = 0; for (double val : d.a) e = std::fmax(e, std::fabs(val)); return e; };

    const double lambda = nu * (mz * PI / Lz) * (mz * PI / Lz);
    const double dt = 2e-3, T = 1.0;
    const int nsteps = (int)std::lround(T / dt);
    const double e0 = energy();
    std::printf("\n lambda = %.4f   dt = %.4f   steps = %d\n", lambda, dt, nsteps);
    std::printf("   t      energy/E0   exact e^{-2 lam t}   max|div u|\n");
    for (int n = 0; n <= nsteps; ++n) {
        if (n % (nsteps / 5) == 0) {
            const double t = n * dt;
            std::printf("  %.2f    %.6f    %.6f          %.2e\n", t, energy() / e0, std::exp(-2 * lambda * t), maxdiv());
        }
        if (n == nsteps) break;
        // implicit no-slip viscous step, per component (the shear mode needs no pressure;
        // see the header note on why the collocated projection is not iterated here)
        viscous(u, dt, us); u = us;
        viscous(v, dt, vs); v = vs;
        viscous(w, dt, ws); w = ws;
    }

    // Compare u_x = B alpha_u to the exact decayed shear mode.
    Field3D un(N, N, N);
    nc_axis_fastderiv(u, un, 0, [&](std::vector<double>& l, std::vector<double>& r){ fx.expandB(l, r); });
    nc_axis_fastderiv(un, un, 1, [&](std::vector<double>& l, std::vector<double>& r){ fy.expandB(l, r); });
    nc_axis_fastderiv(un, un, 2, [&](std::vector<double>& l, std::vector<double>& r){ fz.expandB(l, r); });
    const double decay = std::exp(-lambda * nsteps * dt);
    double err = 0.0;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                err = std::fmax(err, std::fabs(un.a[un.idx(i, j, k)] - std::sin(mz * PI * xz[k] / Lz) * decay));
    std::printf("\n max|u_x - exact shear mode| = %.3e  (implicit-Euler O(dt) + spatial)\n", err);
    std::printf("\nDone.\n");
    return 0;
}
