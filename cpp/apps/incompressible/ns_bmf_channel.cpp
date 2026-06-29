// ns_bmf_channel.cpp
//
// BMF STAGGERED channel solver -- the robust wall route for the incompressible
// loop. Channel = x,y periodic, z clamped wall. MAC layout: pressure p, u, v on
// the primal grid (N,N,N); the wall-normal w on the staggered (derivative) grid
// (N,N,N-1). This file is the STABLE unsteady-Stokes wall solver: implicit
// viscous + exact-divergence-free projection.
//
// Why this works where the collocated route failed (see ns_channel_stokes):
//  * Consistent pressure operator = div.grad (D1^2x + D1^2y + Dv_z G_z,
//    BMFChannelPoisson divgrad=true): no Nyquist null mode, so the discrete
//    projection is exactly divergence-free and stable in a loop. The (0,0) x,y
//    mode is solved on its z line with only the z-constant pinned (BMFChannelPoisson
//    handles this), so z-structured mean divergence is removed too.
//  * The dense-inverse-formed BMF wall Laplacian has HUGE eigenvalues (~1e8), so
//    viscous MUST be implicit: u,v via BMFChannelPoisson (primal Dv_z G_z), w via
//    an inline staggered Helmholtz (z-operator G_z Dv_z on the N-1 grid).
//
// NONLINEAR ADVECTION -- the remaining piece -- is NOT included here yet. Attempts
// to add explicit skew-symmetric advection (with primal<->staggered z-interpolation
// and 2/3 de-aliasing) drove the divergence to round-off but the ENERGY blew up
// even at large viscosity: the dense-inverse-formed wall first-derivative is not
// skew-adjoint and the grid-transfer interpolations are not adjoint, so the
// staggered advection is not discretely energy-conserving. The fix is the
// matrix-free BANDED BMF operators (no dense inverses) + energy-conserving
// (skew-adjoint / adjoint-interpolation) advection -- the next build. The linear
// wall solver below is correct and stable.

#include <cstdio>
#include <cmath>
#include <vector>
#include "bmf3d.h"
#include "bm_operators.h"

static const double PI = 3.14159265358979323846;

int main() {
    const int N = 32, p = 3;
    const double Lp = 2.0 * PI, Lz = 2.0, nu = 0.05;
    SimConfig c; c.p = p; c.nu = nu;
    c.ax[0] = {N, Lp, BC::Periodic, true};
    c.ax[1] = {N, Lp, BC::Periodic, true};
    c.ax[2] = {N, Lz, BC::Clamped, true};

    std::printf("==================================================================\n");
    std::printf(" BMF staggered channel unsteady Stokes (MAC: w staggered in z)\n");
    std::printf(" implicit viscous + consistent exact-div-free projection (stable)\n");
    std::printf("==================================================================\n\n");
    c.print();

    BMFDeriv3D D(c);
    BMFChannelPoisson proj, visc; const char* why = nullptr;
    if (!proj.build(c, why, /*divgrad=*/true) ||      // pressure = div.grad (consistent)
        !visc.build(c, why, /*divgrad=*/false)) {     // u,v viscous = primal BMF Laplacian
        std::printf("setup failed: %s\n", why); return 1;
    }
    OperatorSet bz = build_bmf_bounded(N, p, Lz);
    Mat Lwz = matmul(bz.D1, bz.Dv);                   // G_z Dv_z, the staggered-w z-Laplacian
    std::vector<double> lxd2 = SeparablePoisson::eig_axis(N, Lp, p);   // d2 eigenvalues (x,y)
    FFTPlan pwx, pwy; pwx.init(N); pwy.init(N);

    Field3D u(N, N, N), v(N, N, N), w = D.make_stag(2);
    Field3D un(N, N, N), vn(N, N, N);
    Field3D pr(N, N, N), dvg(N, N, N), tmp(N, N, N);
    Field3D gx(N, N, N), gy(N, N, N), gz = D.make_stag(2);

    auto maxabs = [](const Field3D& f) { double e = 0; for (double val : f.a) e = std::fmax(e, std::fabs(val)); return e; };
    auto sumsq = [](const Field3D& f) { double e = 0; for (double val : f.a) e += val * val; return e; };
    auto energy = [&]() { return sumsq(u) + sumsq(v) + sumsq(w); };
    auto project = [&]() {                            // Chorin: div.grad p = div u ; u -= grad p
        D.divergence(u, v, w, dvg, tmp);
        proj.solve(dvg, pr, 0.0, 1.0);
        D.grad(pr, gx, gy, gz);
        for (size_t i = 0; i < u.size(); ++i) u.a[i] -= gx.a[i];
        for (size_t i = 0; i < v.size(); ++i) v.a[i] -= gy.a[i];
        for (size_t i = 0; i < w.size(); ++i) w.a[i] -= gz.a[i];
    };

    // Implicit wall-normal viscous: (I - nu dt (d2x + d2y + G_z Dv_z)) w = w^n.
    // FFT x,y; per-mode dense ((1 - nu dt (lx+ly)) I - nu dt Lwz) on the N-1 z line.
    std::vector<cdbl> ww((size_t)N * N * (N - 1));
    auto wfft = [&](int axis, bool inv) {
        const int n = N, stride = (axis == 0) ? 1 : N, nl = N * (N - 1);
        const FFTPlan& pl = (axis == 0) ? pwx : pwy;
        std::vector<cdbl> line(n);
        for (int L = 0; L < nl; ++L) {
            const size_t base = (axis == 0) ? (size_t)L * N : (size_t)(L / N) * N * N + (L % N);
            for (int i = 0; i < n; ++i) line[i] = ww[base + (size_t)i * stride];
            pl.run(line.data(), inv);
            for (int i = 0; i < n; ++i) ww[base + (size_t)i * stride] = line[i];
        }
    };
    auto wvisc = [&](double dt) {
        const int nz = N - 1; const size_t zs = (size_t)N * N;
        for (size_t i = 0; i < w.size(); ++i) ww[i] = w.a[i];
        wfft(0, false); wfft(1, false);
        DenseMatrix A(nz); std::vector<double> re(nz), im(nz), sre, sim;
        for (int ky = 0; ky < N; ++ky) for (int kx = 0; kx < N; ++kx) {
            const double sh = 1.0 - nu * dt * (lxd2[kx] + lxd2[ky]);
            const size_t base = (size_t)ky * N + kx;
            for (int i = 0; i < nz; ++i) for (int j = 0; j < nz; ++j)
                A(i, j) = -nu * dt * Lwz(i, j) + (i == j ? sh : 0.0);
            for (int kz = 0; kz < nz; ++kz) { cdbl val = ww[base + (size_t)kz * zs]; re[kz] = val.real(); im[kz] = val.imag(); }
            dense_lu_solve(A, re, sre); dense_lu_solve(A, im, sim);
            for (int kz = 0; kz < nz; ++kz) ww[base + (size_t)kz * zs] = cdbl(sre[kz], sim[kz]);
        }
        wfft(0, true); wfft(1, true);
        for (size_t i = 0; i < w.size(); ++i) w.a[i] = ww[i].real();
    };

    // Smooth initial field, then project to divergence-free.
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        u.a[u.idx(i, j, k)] =  std::sin(2*i*Lp/N) * std::cos(j*Lp/N);
        v.a[v.idx(i, j, k)] = -std::cos(i*Lp/N) * std::sin(j*Lp/N);
    }
    for (int k = 0; k < N-1; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        w.a[w.idx(i, j, k)] = 0.1 * std::sin(i*Lp/N) * std::sin(PI * (k+0.5)/(N-1));
    }
    project();

    const double dt = 5e-3;  const int nsteps = 400;
    const double e0 = energy();
    std::printf("\n dt = %.1e   steps = %d   nu = %.3f\n", dt, nsteps, nu);
    std::printf("   step    energy/E0     max|div u|\n");
    for (int n = 0; n <= nsteps; ++n) {
        if (n % (nsteps / 8) == 0) {
            D.divergence(u, v, w, dvg, tmp);
            std::printf("  %5d    %.6f      %.3e\n", n, energy() / e0, maxabs(dvg));
        }
        if (n == nsteps) break;
        un = u; visc.solve(un, u, 1.0, -nu * dt);      // implicit viscous u, v (primal BMF Lap)
        vn = v; visc.solve(vn, v, 1.0, -nu * dt);
        wvisc(dt);                                     // implicit viscous w (staggered)
        project();
    }

    const double ef = energy() / e0;
    D.divergence(u, v, w, dvg, tmp);
    const bool stable = std::isfinite(ef) && ef < 2.0 && maxabs(dvg) < 1e-6;
    std::printf("\n  %s\n", stable
        ? "STABLE: bounded run, div ~ round-off (BMF staggered wall Stokes loop is stable)."
        : "UNSTABLE");
    std::printf("\nDone.\n");
    return 0;
}
