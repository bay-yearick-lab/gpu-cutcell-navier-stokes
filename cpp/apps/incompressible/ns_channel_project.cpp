// ns_channel_project.cpp
//
// Verifies the coefficient-space discrete projection on a channel (ChannelProjector
// in ns_channel.h): pressure operator L = sum_a (M_a^-1 K_a)^2 so that
// u_new = u - grad(L^{-1} div u) is EXACTLY discrete-divergence-free. Test field
// is a PURE GRADIENT u = grad(psi) with psi in the removable (non-corner) modes,
// so the projection must drive both u and div(u) to round-off -- the decisive
// consistency check of div / grad / L / solve.

#include <cstdio>
#include <cmath>
#include "ns_channel.h"

static const double PI = 3.14159265358979323846;

int main() {
    const int nx = 16, ny = 16, nz = 16, p = 4;
    const double Lp = 2.0 * PI, Lz = 2.0;
    std::printf("==================================================================\n");
    std::printf(" Coefficient-space discrete projection on a channel (exact div-free)\n");
    std::printf(" L = sum_a (M_a^-1 K_a)^2 ;  div(u - grad phi) -> round-off\n");
    std::printf("==================================================================\n");

    SimConfig c; c.p = p;
    c.ax[0] = {nx, Lp, BC::Periodic, true};
    c.ax[1] = {ny, Lp, BC::Periodic, true};
    c.ax[2] = {nz, Lz, BC::Clamped, true};
    ChannelProjector proj; const char* why = nullptr;
    if (!proj.build(c, why)) { std::printf("setup failed: %s\n", why); return 1; }

    // u = grad(psi), psi in removable (x,y wavenumber 1, no const/Nyquist) modes.
    Field3D psi(nx, ny, nz), au(nx, ny, nz), av(nx, ny, nz), aw(nx, ny, nz);
    Field3D d(nx, ny, nz), phi(nx, ny, nz), t(nx, ny, nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                psi.a[psi.idx(i, j, k)] = std::sin(2 * PI * i / nx) * std::sin(2 * PI * j / ny)
                                        * std::sin(PI * (k + 0.5) / nz);
    proj.apply_G(psi, au, 0);
    proj.apply_G(psi, av, 1);
    proj.apply_G(psi, aw, 2);

    auto maxabs = [](const Field3D& f) { double e = 0; for (double v : f.a) e = std::fmax(e, std::fabs(v)); return e; };
    proj.divergence(au, av, aw, d);
    const double div0 = maxabs(d);

    proj.solve_L(d, phi);                              // L phi = div u
    proj.apply_G(phi, t, 0); for (size_t i = 0; i < au.size(); ++i) au.a[i] -= t.a[i];
    proj.apply_G(phi, t, 1); for (size_t i = 0; i < av.size(); ++i) av.a[i] -= t.a[i];
    proj.apply_G(phi, t, 2); for (size_t i = 0; i < aw.size(); ++i) aw.a[i] -= t.a[i];

    proj.divergence(au, av, aw, d);
    const double div1 = maxabs(d);
    double unew = 0.0;
    for (size_t i = 0; i < au.size(); ++i)
        unew = std::fmax(unew, std::fmax(std::fabs(au.a[i]), std::fmax(std::fabs(av.a[i]), std::fabs(aw.a[i]))));

    std::printf("\n  max|div u|   before projection: %.3e\n", div0);
    std::printf("  max|div u|   after  projection: %.3e\n", div1);
    std::printf("  max|u_new|   (pure gradient -> 0): %.3e\n", unew);
    const bool pass = div1 < 1e-9 && unew < 1e-7;     // div exact; velocity = residual x cond(L)
    std::printf("\n  %s\n", pass ? "PASS: gradient removed; projection is discretely divergence-free."
                                 : "FAIL");
    std::printf("\nDone.\n");
    return 0;
}
