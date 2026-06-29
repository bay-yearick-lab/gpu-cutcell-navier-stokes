// ns.h
//
// Incompressible Navier-Stokes on the collocated B-spline mimetic grid
// (same-grid BMC/BMG route). Fractional-step projection:
//
//   du/dt = -N(u) + nu lap(u) - grad p,    div u = 0
//
// where N(u) is the SKEW-SYMMETRIC convective form (energy-conserving):
//   N_i = 1/2 [ sum_j u_j d_j u_i  +  sum_j d_j (u_j u_i) ].
//
// Time advance is SSP-RK3; each stage advances the momentum without pressure to
// an intermediate u*, then PROJECTS it divergence-free:
//   solve (div.grad) phi = div u* ;   u = u* - grad phi.
// Because the pressure operator is exactly div.grad (poisson.h, divgrad=true) and
// div/grad are the same discrete first-derivative operators, the projected
// velocity is divergence-free to round-off. Pressure p = phi/dt is recoverable.
//
// Everything is the directional collapse: grad, div, Laplacian, and advection are
// axis-wise 1D line solves, OpenMP-parallel (bm3d.h). 2D is z inactive.
//
// NOTE (collocated caveat): div.grad on a collocated grid has a Nyquist null mode
// (the first-derivative symbol vanishes at the grid Nyquist), which the solve
// projects out. For smooth/resolved DNS this is harmless; flows with energy at
// the grid scale want the staggered BMF route (the reason both are planned).

#ifndef NS_H
#define NS_H

#include <vector>
#include "bm3d.h"
#include "poisson.h"

// ---- pointwise field helpers (parallel) -----------------------------------
inline void fld_mul(Field3D& o, const Field3D& a, const Field3D& b) {
    const size_t N = o.size();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) o.a[i] = a.a[i] * b.a[i];
}
inline void fld_fma(Field3D& o, const Field3D& a, const Field3D& b) {   // o += a*b
    const size_t N = o.size();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) o.a[i] += a.a[i] * b.a[i];
}
inline void fld_add(Field3D& o, const Field3D& b) {
    const size_t N = o.size();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) o.a[i] += b.a[i];
}
// u <- c0*u0 + cu*u + ck*k   (the SSP-RK3 stage combination)
inline void rk_combine(Field3D& u, const Field3D& u0, double c0, double cu,
                       const Field3D& k, double ck) {
    const size_t N = u.size();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) u.a[i] = c0 * u0.a[i] + cu * u.a[i] + ck * k.a[i];
}

struct IncompressibleNS {
    SimConfig cfg;
    Deriv3D D;
    SeparablePoisson pois;           // pressure: div.grad
    SeparablePoisson pois_visc;      // implicit viscous: direct Laplacian
    bool ready = false;
    const char* setup_error = nullptr;
    int nx, ny, nz; bool d3; double nu;

    Field3D u, v, w;                 // velocity (w unused/zero in 2D)
    Field3D u0, v0, w0;              // SSP-RK3 snapshot
    Field3D ru, rv, rw;             // momentum rhs (explicit scheme)
    Field3D Nu, Nv, Nw;             // skew advection N(u) at this stage (IMEX)
    Field3D Np_u, Np_v, Np_w;       // N(u) at the previous stage (IMEX)
    Field3D t1, t2, t3, phi, divu;  // work

    explicit IncompressibleNS(const SimConfig& c)
        : cfg(c), D(c),
          nx(c.ax[0].N), ny(c.ax[1].N), nz(c.active(2) ? c.ax[2].N : 1),
          d3(c.active(2)), nu(c.nu),
          u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz),
          u0(nx, ny, nz), v0(nx, ny, nz), w0(nx, ny, nz),
          ru(nx, ny, nz), rv(nx, ny, nz), rw(nx, ny, nz),
          Nu(nx, ny, nz), Nv(nx, ny, nz), Nw(nx, ny, nz),
          Np_u(nx, ny, nz), Np_v(nx, ny, nz), Np_w(nx, ny, nz),
          t1(nx, ny, nz), t2(nx, ny, nz), t3(nx, ny, nz),
          phi(nx, ny, nz), divu(nx, ny, nz) {
        ready = pois.build(c, setup_error, /*divgrad=*/true);          // pressure operator = div.grad
        ready = ready && pois_visc.build(c, setup_error, /*divgrad=*/false);  // viscous: direct Laplacian
    }

    // Skew-symmetric convective term for one advected component q -> N.
    void advect_component(const Field3D& q, Field3D& N) {
        // advective form: N = u q_x + v q_y + w q_z
        D.deriv(q, t1, 0, 1); fld_mul(N, u, t1);
        D.deriv(q, t1, 1, 1); fld_fma(N, v, t1);
        if (d3) { D.deriv(q, t1, 2, 1); fld_fma(N, w, t1); }
        // divergence form: N += (u q)_x + (v q)_y + (w q)_z
        fld_mul(t2, u, q); D.deriv(t2, t1, 0, 1); fld_add(N, t1);
        fld_mul(t2, v, q); D.deriv(t2, t1, 1, 1); fld_add(N, t1);
        if (d3) { fld_mul(t2, w, q); D.deriv(t2, t1, 2, 1); fld_add(N, t1); }
        scale(N, 0.5);                                          // average -> skew-symmetric
    }

    // Momentum rhs (no pressure): R_i = -N_i + nu lap(u_i).
    void rhs() {
        advect_component(u, ru);
        advect_component(v, rv);
        if (d3) advect_component(w, rw);
        D.laplacian_any(u, t3, t1); scale(ru, -1.0); axpy(ru, nu, t3);
        D.laplacian_any(v, t3, t1); scale(rv, -1.0); axpy(rv, nu, t3);
        if (d3) { D.laplacian_any(w, t3, t1); scale(rw, -1.0); axpy(rw, nu, t3); }
    }

    // Project (u,v,w) divergence-free in place: solve div.grad phi = div u; u -= grad phi.
    void project() {
        D.divergence(u, v, w, divu, t1);
        pois.solve(divu, phi, 0.0, 1.0);
        D.deriv(phi, t1, 0, 1); axpy(u, -1.0, t1);
        D.deriv(phi, t1, 1, 1); axpy(v, -1.0, t1);
        if (d3) { D.deriv(phi, t1, 2, 1); axpy(w, -1.0, t1); }
    }

    // One SSP-RK3 step (projecting every stage so the state stays div-free).
    void step(double dt) {
        u0.a = u.a; v0.a = v.a; if (d3) w0.a = w.a;
        // stage 1: u = P(u0 + dt R(u0))
        rhs();
        axpy(u, dt, ru); axpy(v, dt, rv); if (d3) axpy(w, dt, rw);
        project();
        // stage 2: u = P(3/4 u0 + 1/4 u + 1/4 dt R(u))
        rhs();
        rk_combine(u, u0, 0.75, 0.25, ru, 0.25 * dt);
        rk_combine(v, v0, 0.75, 0.25, rv, 0.25 * dt);
        if (d3) rk_combine(w, w0, 0.75, 0.25, rw, 0.25 * dt);
        project();
        // stage 3: u = P(1/3 u0 + 2/3 u + 2/3 dt R(u))
        rhs();
        rk_combine(u, u0, 1.0 / 3.0, 2.0 / 3.0, ru, (2.0 / 3.0) * dt);
        rk_combine(v, v0, 1.0 / 3.0, 2.0 / 3.0, rv, (2.0 / 3.0) * dt);
        if (d3) rk_combine(w, w0, 1.0 / 3.0, 2.0 / 3.0, rw, (2.0 / 3.0) * dt);
        project();
    }

    // ----- IMEX time integration (low-storage IMEX RK3 / Crank-Nicolson) ---
    // Explicit RK3 on the skew-symmetric advection, implicit CN on the viscous
    // term -- the incompressible-DNS standard. Each of the 3 substages solves a
    // Helmholtz problem (I - dt nu beta_k L) and then projects. Removing the
    // viscous term from the explicit stability limit is the whole point: dt is
    // set by advective CFL only, even on fine/wall-resolved grids.

    void advect_all() {                          // N(u) for every component
        advect_component(u, Nu);
        advect_component(v, Nv);
        if (d3) advect_component(w, Nw);
    }

    // Advance one velocity component through a substage:
    //   (I - dt nu ab L) q = q + dt(-g N - z Np + nu ab L q).
    void imex_component(Field3D& q, const Field3D& Nq, const Field3D& Nq_p,
                        double g, double z, double ab, double dt) {
        D.laplacian_any(q, t3, t1);              // t3 = L q (direct Laplacian, matches pois_visc)
        const size_t Ntot = q.size();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < Ntot; ++i)
            t2.a[i] = q.a[i] + dt * (-g * Nq.a[i] - z * Nq_p.a[i] + nu * ab * t3.a[i]);
        pois_visc.solve(t2, q, 1.0, -dt * nu * ab);
    }

    void step_imex(double dt) {
        static const double g[3] = {8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0};
        static const double z[3] = {0.0, -17.0 / 60.0, -5.0 / 12.0};
        for (int k = 0; k < 3; ++k) {
            const double ab = 0.5 * (g[k] + z[k]);          // Crank-Nicolson viscous weight
            advect_all();
            imex_component(u, Nu, Np_u, g[k], z[k], ab, dt);
            imex_component(v, Nv, Np_v, g[k], z[k], ab, dt);
            if (d3) imex_component(w, Nw, Np_w, g[k], z[k], ab, dt);
            project();
            Np_u.a.swap(Nu.a); Np_v.a.swap(Nv.a); if (d3) Np_w.a.swap(Nw.a);
        }
    }

    // Diagnostics.
    double max_divergence() {
        D.divergence(u, v, w, divu, t1);
        double e = 0.0;
        for (double x : divu.a) e = std::fmax(e, std::fabs(x));
        return e;
    }
    double kinetic_energy() const {
        double s = 0.0;
        const size_t N = u.size();
        for (size_t i = 0; i < N; ++i) s += u.a[i] * u.a[i] + v.a[i] * v.a[i] + (d3 ? w.a[i] * w.a[i] : 0.0);
        return 0.5 * s / N;                                    // mean KE density
    }
};

#endif // NS_H
