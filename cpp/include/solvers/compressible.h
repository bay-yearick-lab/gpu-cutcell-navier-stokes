// compressible.h
//
// Compressible Navier-Stokes on the periodic B-spline mimetic grid, in the
// SMOOTH (shock-free) regime: turbulence and aeroacoustics. Conservative
// variables  U = (rho, rho u, rho v, rho w, rho E)  advanced by an explicit
// SSP-RK3 step.
//
//   d_t rho       + d_j(rho u_j)                       = 0
//   d_t(rho u_i)  + d_j(rho u_i u_j) + d_i p           = d_j tau_ij
//   d_t(rho E)    + d_j((rho E + p) u_j)               = d_j(u_i tau_ij - q_j)
//
//   p = (gamma-1)(rho E - 1/2 rho|u|^2),  T = p/(rho R),
//   tau_ij = mu(d_j u_i + d_i u_j - 2/3 delta_ij d_k u_k),  q_j = -kappa d_j T,
//   kappa  = mu c_p / Pr = mu gamma R / ((gamma-1) Pr).
//
// CONVECTION uses the fully-split skew-symmetric compressible form:
// every convective flux d_j(a u_j) is replaced by the discrete average
//   1/2[ d_j(a u_j) + u_j d_j a + a d_j u_j ],
// and the momentum triple product rho u_i u_j by
//   1/2[ d_j(rho u_i u_j) + u_i d_j(rho u_j) + rho u_j d_j u_i ].
// Because the periodic BMC derivative D is EXACTLY skew-symmetric (D^T = -D, so
// discrete integration by parts sum a(Db) = -sum (Da)b holds to round-off), this
// split form conserves mass, momentum and total energy EXACTLY, and kinetic
// energy in the inviscid limit -- the discrete analogue of the continuous energy
// budget, with no added dissipation. That is the whole point of a mimetic core
// for DNS / CAA. (Verified: ns compressible MMS + isentropic-vortex order +
// inviscid TGV conservation.)
//
// Operators are the directional collapse (bm3d.h): every d_j is an axis-wise 1D
// banded/circulant line solve, OpenMP-parallel across lines.

#ifndef COMPRESSIBLE_H
#define COMPRESSIBLE_H

#include <vector>
#include <cmath>
#include "bm3d.h"

struct CompressibleNS {
    SimConfig cfg;
    Deriv3D D;
    int nx, ny, nz; bool d3;
    double gamma, Rgas, mu, Pr;
    double Twall = 1.0;              // isothermal-wall temperature (clamped axes)
    bool ready = false;
    bool with_pressure = true;       // set false to isolate the convective KE rate
    const char* setup_error = nullptr;

    // conservative state
    Field3D rho, mx, my, mz, Egy;
    // primitives (derived each rhs)
    Field3D u, v, w, pr, Tmp;
    // rhs accumulators
    Field3D Rr, Rmx, Rmy, Rmz, Re_;
    // RK3 snapshot
    Field3D rho0, mx0, my0, mz0, E0;
    // velocity gradients g[i][j] = d u_i / d x_j, and scratch
    Field3D g[3][3];
    Field3D s1, s2, s3, divm, theta;
    Field3D bvis, acc_e;             // viscous scratch (only used when mu>0)

    explicit CompressibleNS(const SimConfig& c, double gamma_ = 1.4,
                            double Rgas_ = 1.0, double mu_ = 0.0, double Pr_ = 0.72)
        : cfg(c), D(c),
          nx(c.ax[0].N), ny(c.ax[1].N), nz(c.active(2) ? c.ax[2].N : 1),
          d3(c.active(2)), gamma(gamma_), Rgas(Rgas_), mu(mu_), Pr(Pr_),
          rho(nx, ny, nz), mx(nx, ny, nz), my(nx, ny, nz), mz(nx, ny, nz), Egy(nx, ny, nz),
          u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz), pr(nx, ny, nz), Tmp(nx, ny, nz),
          Rr(nx, ny, nz), Rmx(nx, ny, nz), Rmy(nx, ny, nz), Rmz(nx, ny, nz), Re_(nx, ny, nz),
          rho0(nx, ny, nz), mx0(nx, ny, nz), my0(nx, ny, nz), mz0(nx, ny, nz), E0(nx, ny, nz),
          s1(nx, ny, nz), s2(nx, ny, nz), s3(nx, ny, nz), divm(nx, ny, nz), theta(nx, ny, nz),
          bvis(nx, ny, nz), acc_e(nx, ny, nz) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) g[a][b] = Field3D(nx, ny, nz);
        ready = cfg.validate(setup_error);
    }

    size_t N() const { return rho.size(); }
    void d1(const Field3D& in, Field3D& out, int ax) const { D.deriv(in, out, ax, 1); }
    void d2(const Field3D& in, Field3D& out, int ax) const { D.deriv(in, out, ax, 2); }

    // primitives from conservative state
    void primitives() {
        const size_t M = N();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            const double r = rho.a[i], ir = 1.0 / r;
            const double uu = mx.a[i] * ir, vv = my.a[i] * ir,
                         ww = d3 ? mz.a[i] * ir : 0.0;
            u.a[i] = uu; v.a[i] = vv; w.a[i] = ww;
            const double ke = 0.5 * r * (uu * uu + vv * vv + ww * ww);
            const double p = (gamma - 1.0) * (Egy.a[i] - ke);
            pr.a[i] = p;
            Tmp.a[i] = p * ir / Rgas;
        }
    }

    void rhs() {
        const size_t M = N();
        primitives();

        // velocity gradients g[i][j] = d u_i/dx_j
        const Field3D* U[3] = {&u, &v, &w};
        const int na = d3 ? 3 : 2;
        for (int a = 0; a < (d3 ? 3 : 2); ++a)
            for (int b = 0; b < na; ++b) d1(*U[a], g[a][b], b);
        // theta = div u
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i)
            theta.a[i] = g[0][0].a[i] + g[1][1].a[i] + (d3 ? g[2][2].a[i] : 0.0);
        // divm = d_j (rho u_j) = d_j m_j
        d1(mx, s1, 0); d1(my, s2, 1);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) divm.a[i] = s1.a[i] + s2.a[i];
        if (d3) {
            d1(mz, s1, 2);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < M; ++i) divm.a[i] += s1.a[i];
        }

        // ---- continuity: Rr = -div(rho u) = -div m (plain divergence) --------
        // m_j is a stored conserved variable, so this is a pure linear
        // derivative -- NO aliasing -- AND it is the form the split-form momentum
        // split needs for EXACT discrete kinetic-energy conservation (the split
        // continuity, equal in the continuum, breaks that cancellation).
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) Rr.a[i] = -divm.a[i];

        // ---- momentum (split skew): for each i ----
        // conv_i = 1/2[ sum_j d_j(m_i u_j) + u_i divm + sum_j m_j d_j u_i ]
        Field3D* Rm[3] = {&Rmx, &Rmy, &Rmz};
        const Field3D* Mi[3] = {&mx, &my, &mz};
        for (int i = 0; i < na; ++i) {
            Field3D& R = *Rm[i];
            #pragma omp parallel for schedule(static)
            for (size_t q = 0; q < M; ++q) R.a[q] = 0.5 * U[i]->a[q] * divm.a[q];
            // sum_j d_j(m_i u_j)
            for (int j = 0; j < na; ++j) {
                const Field3D& vel = (j == 0) ? u : (j == 1) ? v : w;
                #pragma omp parallel for schedule(static)
                for (size_t q = 0; q < M; ++q) s1.a[q] = Mi[i]->a[q] * vel.a[q];
                d1(s1, s2, j);
                #pragma omp parallel for schedule(static)
                for (size_t q = 0; q < M; ++q) R.a[q] += 0.5 * s2.a[q];
            }
            // sum_j m_j d_j u_i = m_x g[i][0] + m_y g[i][1] + m_z g[i][2]
            #pragma omp parallel for schedule(static)
            for (size_t q = 0; q < M; ++q) {
                double c = mx.a[q] * g[i][0].a[q] + my.a[q] * g[i][1].a[q];
                if (d3) c += mz.a[q] * g[i][2].a[q];
                R.a[q] += 0.5 * c;
            }
            // pressure gradient: + d_i p ; total convective+pressure with minus sign
            d1(pr, s1, i);
            const double pw = with_pressure ? 1.0 : 0.0;
            #pragma omp parallel for schedule(static)
            for (size_t q = 0; q < M; ++q) R.a[q] = -R.a[q] - pw * s1.a[q];
        }

        // ---- energy (split): Re_ = -1/2[div(a u)+u.grad a+a theta], a=E+p ----
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) { Re_.a[i] = 0.0; s3.a[i] = Egy.a[i] + pr.a[i]; }
        // reuse split_conv_scalar with a = (E+p) stored in a temp; need its own buffer
        // (split_conv_scalar writes into dst with -1/2 factors)
        {
            Field3D& aE = s3;        // a = E + p (already filled)
            // div(a u_j) + u_j d_j a  per axis, plus a theta
            for (int ax = 0; ax < na; ++ax) {
                const Field3D& vel = (ax == 0) ? u : (ax == 1) ? v : w;
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < M; ++i) s1.a[i] = aE.a[i] * vel.a[i];
                d1(s1, s2, ax);
                d1(aE, divm, ax);    // reuse divm as scratch for d_j a (divm no longer needed)
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < M; ++i)
                    Re_.a[i] += -0.5 * (s2.a[i] + vel.a[i] * divm.a[i]);
            }
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < M; ++i) Re_.a[i] += -0.5 * aE.a[i] * theta.a[i];
        }

        // ---- viscous (constant mu, kappa) ----
        if (mu > 0.0) add_viscous();
    }

    // Add div(tau) to momentum rhs and div(u.tau) + kappa lap T to energy rhs.
    void add_viscous() {
        const size_t M = N();
        const int na = d3 ? 3 : 2;
        const double kappa = mu * gamma * Rgas / ((gamma - 1.0) * Pr);
        const double tt = 2.0 / 3.0;
        // tau is symmetric; compute and immediately differentiate per (i): need
        // sum_j d_j tau_ij. Build tau_ij on the fly using g and theta.
        // For energy we also need b_j = sum_i u_i tau_ij, then div(b).
        // Store tau columns transiently in s1 (per (i,j)) -- but we need d_j of
        // each, so loop i, accumulate.
        Field3D* Rm[3] = {&Rmx, &Rmy, &Rmz};
        const Field3D* U[3] = {&u, &v, &w};
        Field3D& bj = bvis;          // b_j for current j (member scratch)
        #pragma omp parallel for schedule(static)
        for (size_t q = 0; q < M; ++q) acc_e.a[q] = 0.0;   // div(b) + heat accumulator
        for (int i = 0; i < na; ++i) {
            // momentum component i: sum_j d_j tau_ij
            Field3D& R = *Rm[i];
            for (int j = 0; j < na; ++j) {
                // tau_ij = mu(g[i][j]+g[j][i] - (2/3)delta_ij theta)
                #pragma omp parallel for schedule(static)
                for (size_t q = 0; q < M; ++q) {
                    double tij = g[i][j].a[q] + g[j][i].a[q];
                    if (i == j) tij -= tt * theta.a[q];
                    s1.a[q] = mu * tij;             // tau_ij
                }
                d1(s1, s2, j);
                #pragma omp parallel for schedule(static)
                for (size_t q = 0; q < M; ++q) R.a[q] += s2.a[q];
                // energy work: b_j += u_i * tau_ij  (accumulate over i for this j)
                // handled in the separate j-loop below to avoid recompute
            }
        }
        // energy viscous: b_j = sum_i u_i tau_ij ; acc_e = sum_j d_j b_j
        for (int j = 0; j < na; ++j) {
            #pragma omp parallel for schedule(static)
            for (size_t q = 0; q < M; ++q) bj.a[q] = 0.0;
            for (int i = 0; i < na; ++i) {
                #pragma omp parallel for schedule(static)
                for (size_t q = 0; q < M; ++q) {
                    double tij = g[i][j].a[q] + g[j][i].a[q];
                    if (i == j) tij -= tt * theta.a[q];
                    bj.a[q] += U[i]->a[q] * (mu * tij);
                }
            }
            d1(bj, s2, j);
            #pragma omp parallel for schedule(static)
            for (size_t q = 0; q < M; ++q) acc_e.a[q] += s2.a[q];
        }
        // heat conduction: + kappa lap T
        for (int ax = 0; ax < na; ++ax) {
            d2(Tmp, s2, ax);
            #pragma omp parallel for schedule(static)
            for (size_t q = 0; q < M; ++q) acc_e.a[q] += kappa * s2.a[q];
        }
        #pragma omp parallel for schedule(static)
        for (size_t q = 0; q < M; ++q) Re_.a[q] += acc_e.a[q];
    }

    // U <- c0*U0 + cu*U + ck*dt*R   over all five conservative fields
    void rk_combine(double c0, double cu, double ck, double dt) {
        const size_t M = N();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < M; ++i) {
            rho.a[i] = c0 * rho0.a[i] + cu * rho.a[i] + ck * dt * Rr.a[i];
            mx.a[i]  = c0 * mx0.a[i]  + cu * mx.a[i]  + ck * dt * Rmx.a[i];
            my.a[i]  = c0 * my0.a[i]  + cu * my.a[i]  + ck * dt * Rmy.a[i];
            if (d3) mz.a[i] = c0 * mz0.a[i] + cu * mz.a[i] + ck * dt * Rmz.a[i];
            Egy.a[i] = c0 * E0.a[i]   + cu * Egy.a[i] + ck * dt * Re_.a[i];
        }
    }

    void snapshot() {
        rho0.a = rho.a; mx0.a = mx.a; my0.a = my.a; E0.a = Egy.a;
        if (d3) mz0.a = mz.a;
    }

    // No-slip isothermal walls on clamped axes: enforce velocity=0 (momentum=0)
    // and internal energy from Twall (E = rho R Twall/(gamma-1)) on each wall
    // plane, as a STRONG (Dirichlet) BC applied after each RK stage. No-op when
    // every axis is periodic. Density at the wall evolves with the scheme.
    void set_wall_node(size_t id) {
        const double e_int = Rgas * Twall / (gamma - 1.0);
        mx.a[id] = 0.0; my.a[id] = 0.0; if (d3) mz.a[id] = 0.0;
        Egy.a[id] = rho.a[id] * e_int;
    }
    void apply_wall_bc() {
        if (cfg.active(0) && !cfg.periodic(0))
            for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) {
                set_wall_node(rho.idx(0, j, k)); set_wall_node(rho.idx(nx - 1, j, k)); }
        if (cfg.active(1) && !cfg.periodic(1))
            for (int k = 0; k < nz; ++k) for (int i = 0; i < nx; ++i) {
                set_wall_node(rho.idx(i, 0, k)); set_wall_node(rho.idx(i, ny - 1, k)); }
        if (d3 && cfg.active(2) && !cfg.periodic(2))
            for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
                set_wall_node(rho.idx(i, j, 0)); set_wall_node(rho.idx(i, j, nz - 1)); }
    }

    // One explicit SSP-RK3 step. apply_wall_bc() is a no-op when
    // all axes are periodic.
    void step(double dt) {
        snapshot();
        rhs(); rk_combine(1.0, 0.0, 1.0, dt);          apply_wall_bc();   // U1
        rhs(); rk_combine(0.75, 0.25, 0.25, dt);       apply_wall_bc();   // U2
        rhs(); rk_combine(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, dt); apply_wall_bc();
    }

    // ---- diagnostics ----
    double max_wavespeed() {
        primitives();
        const size_t M = N();
        double smax = 0.0;
        #pragma omp parallel for reduction(max:smax) schedule(static)
        for (size_t i = 0; i < M; ++i) {
            const double c = std::sqrt(gamma * pr.a[i] / rho.a[i]);
            const double sp = std::sqrt(u.a[i] * u.a[i] + v.a[i] * v.a[i] + w.a[i] * w.a[i]) + c;
            if (sp > smax) smax = sp;
        }
        return smax;
    }
    double total(const Field3D& f) const {
        double s = 0.0; const size_t M = f.size();
        for (size_t i = 0; i < M; ++i) s += f.a[i];
        return s / M;
    }
    double kinetic_energy() {
        primitives();
        const size_t M = N(); double s = 0.0;
        for (size_t i = 0; i < M; ++i)
            s += 0.5 * rho.a[i] * (u.a[i] * u.a[i] + v.a[i] * v.a[i] + w.a[i] * w.a[i]);
        return s / M;
    }
    double min_density() const {
        double m = 1e300; for (double r : rho.a) if (r < m) m = r; return m;
    }
};

#endif  // COMPRESSIBLE_H
