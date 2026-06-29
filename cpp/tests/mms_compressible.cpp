// mms_compressible.cpp
//
// Verification of the compressible Navier-Stokes solver (compressible.h):
//
//   (1) INVISCID CONSERVATION -- the split skew-symmetric form must conserve
//       mass, momentum, and total energy to round-off (linear invariants), and
//       kinetic energy to the time-integration error (semi-discrete invariant).
//
//   (2) ISENTROPIC VORTEX -- an EXACT solution of the compressible Euler
//       equations advects without change of shape; the L2 error vs the exact
//       translated vortex measures the spatial order (= the BMC D1 order
//       2*ceil(p/2)). Formal vs measured reported per degree.
//
//   (3) VISCOUS OPERATOR MMS -- isolating rhs(mu) - rhs(0) at a fixed state
//       gives EXACTLY the viscous term; compared against the analytic stress
//       divergence d_j tau_ij and viscous work for a manufactured velocity, it
//       must converge at the operator order.
//
//   (4) FULL INVISCID RHS MMS -- a GENERIC smooth manufactured state (not the
//       special vortex); the discrete split RHS for all five equations is
//       compared against the analytic continuous Euler RHS. Verifies the whole
//       coupled convective + pressure + energy-flux + EOS system at order p.
//
//   (5) HEAT CONDUCTION MMS -- with u=0 the energy RHS reduces to kappa lap T;
//       compared against analytic kappa lap T it converges at the D2 order. This
//       closes operator coverage (vortex = convection; (3) = viscous stress +
//       work; (5) = heat).

#include <cstdio>
#include <cmath>
#include <vector>
#include "compressible.h"
#include "bspline.h"        // open_uniform_knots, greville_points (clamped node coords)

static const double PI = 3.14159265358979323846;

// One separable trig mode  phi = c0 + A * X(x) Y(y) Z(z),  k=1  (X = sin or cos).
struct Prim {
    double c0, A; bool sx, sy, sz;
    static void trig(bool s, double t, double& X, double& Xp) {
        if (s) { X = std::sin(t); Xp = std::cos(t); }
        else   { X = std::cos(t); Xp = -std::sin(t); }
    }
    void at(double x, double y, double z,
            double& v, double& vx, double& vy, double& vz) const {
        double X, Xp, Y, Yp, Z, Zp;
        trig(sx, x, X, Xp); trig(sy, y, Y, Yp); trig(sz, z, Z, Zp);
        v = c0 + A * X * Y * Z;
        vx = A * Xp * Y * Z; vy = A * X * Yp * Z; vz = A * X * Y * Zp;
    }
};

// A generic smooth manufactured compressible state + its analytic continuous
// INVISCID (Euler) RHS, used for both periodic and channel (wall) MMS.
struct ManState {
    Prim r{1.2, 0.15, false, false, false};   // rho  (cos cos cos)
    Prim u{0.0, 0.30, true,  false, false};   // u    (sin cos cos)
    Prim v{0.0, 0.25, false, true,  false};   // v    (cos sin cos)
    Prim w{0.0, 0.20, false, false, true };   // w    (cos cos sin)
    Prim p{1.0, 0.20, false, false, false};   // p    (cos cos cos)
    double gamma = 1.4;
    void cons(double x, double y, double z, double C[5]) const {
        double rho, rx, ry, rz, U, ux, uy, uz, V, vx, vy, vz, W, wx, wy, wz, P, px, py, pz;
        r.at(x, y, z, rho, rx, ry, rz); u.at(x, y, z, U, ux, uy, uz);
        v.at(x, y, z, V, vx, vy, vz); w.at(x, y, z, W, wx, wy, wz); p.at(x, y, z, P, px, py, pz);
        C[0] = rho; C[1] = rho * U; C[2] = rho * V; C[3] = rho * W;
        C[4] = P / (gamma - 1.0) + 0.5 * rho * (U * U + V * V + W * W);
    }
    void euler_rhs(double x, double y, double z, double R[5]) const {
        const double gm = gamma / (gamma - 1.0);
        double rho, rx, ry, rz, U, ux, uy, uz, V, vx, vy, vz, W, wx, wy, wz, P, px, py, pz;
        r.at(x, y, z, rho, rx, ry, rz); u.at(x, y, z, U, ux, uy, uz);
        v.at(x, y, z, V, vx, vy, vz); w.at(x, y, z, W, wx, wy, wz); p.at(x, y, z, P, px, py, pz);
        R[0] = -((rx * U + rho * ux) + (ry * V + rho * vy) + (rz * W + rho * wz));
        R[1] = -((rx * U * U + rho * 2 * U * ux)
               + (ry * U * V + rho * uy * V + rho * U * vy)
               + (rz * U * W + rho * uz * W + rho * U * wz)) - px;
        R[2] = -((rx * U * V + rho * ux * V + rho * U * vx)
               + (ry * V * V + rho * 2 * V * vy)
               + (rz * V * W + rho * vz * W + rho * V * wz)) - py;
        R[3] = -((rx * U * W + rho * ux * W + rho * U * wx)
               + (ry * V * W + rho * vy * W + rho * V * wy)
               + (rz * W * W + rho * 2 * W * wz)) - pz;
        const double q = U * U + V * V + W * W, Hd = gm * P + 0.5 * rho * q;
        const double Hdx = gm * px + 0.5 * (rx * q + rho * 2 * (U * ux + V * vx + W * wx));
        const double Hdy = gm * py + 0.5 * (ry * q + rho * 2 * (U * uy + V * vy + W * wy));
        const double Hdz = gm * pz + 0.5 * (rz * q + rho * 2 * (U * uz + V * vz + W * wz));
        R[4] = -((Hdx * U + Hd * ux) + (Hdy * V + Hd * vy) + (Hdz * W + Hd * wz));
    }
};

// node coordinates of one axis (uniform if periodic, Greville if clamped)
static std::vector<double> axis_nodes(int N, double L, bool periodic, int p) {
    std::vector<double> x(N);
    if (periodic) { for (int i = 0; i < N; ++i) x[i] = i * L / N; }
    else x = greville_points(open_uniform_knots(N, p, L), p, N);
    return x;
}

static SimConfig periodic_cfg(int N, int p, bool three_d, double L = 2 * PI) {
    SimConfig c; c.p = p; c.route = Route::BMC; c.nu = 0.0;
    c.ax[0] = {N, L, BC::Periodic, true};
    c.ax[1] = {N, L, BC::Periodic, true};
    c.ax[2] = {N, L, BC::Periodic, three_d};
    return c;
}

// ----------------------------------------------------------------------------
// (1) inviscid conservation
// ----------------------------------------------------------------------------
static void test_conservation() {
    std::printf("=================================================================\n");
    std::printf(" (1) Inviscid conservation (split skew-symmetric form)\n");
    std::printf("=================================================================\n");
    const int N = 32, p = 4; const double L = 2 * PI;
    SimConfig c = periodic_cfg(N, p, true, L);
    CompressibleNS S(c, /*gamma=*/1.4);
    const double h = L / N;
    // richer multi-mode IC so the physical pressure-work term does NOT vanish
    // by mode orthogonality (makes the convective-KE check meaningful).
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h, z = k * h;
        const size_t id = S.rho.idx(i, j, k);
        const double rho = 1.0 + 0.2 * std::sin(x) * std::cos(y) * std::cos(z)
                               + 0.1 * std::cos(2 * x) * std::sin(y);
        const double uu = 0.3 * std::sin(2 * x) * std::cos(y) * std::cos(z)
                        + 0.2 * std::cos(x) * std::sin(z);
        const double vv = 0.3 * std::cos(x) * std::sin(2 * y) * std::cos(z);
        const double ww = 0.3 * std::cos(x) * std::cos(y) * std::sin(2 * z);
        const double p_ = 1.0 + 0.2 * std::cos(x) * std::cos(y) * std::sin(z)
                              + 0.1 * std::sin(x) * std::sin(2 * z);
        S.rho.a[id] = rho;
        S.mx.a[id] = rho * uu; S.my.a[id] = rho * vv; S.mz.a[id] = rho * ww;
        S.Egy.a[id] = p_ / (1.4 - 1.0) + 0.5 * rho * (uu * uu + vv * vv + ww * ww);
    }
    const double m0 = S.total(S.rho), px0 = S.total(S.mx), py0 = S.total(S.my),
                 pz0 = S.total(S.mz), e0 = S.total(S.Egy);
    const size_t M = S.N();
    auto dkedt = [&]() {                       // semi-discrete sum_i u_i R_mi - 1/2|u|^2 R_rho
        S.rhs();
        double s = 0, sc = 0;
        for (size_t i = 0; i < M; ++i) {
            const double uu = S.u.a[i], vv = S.v.a[i], ww = S.w.a[i];
            s += uu * S.Rmx.a[i] + vv * S.Rmy.a[i] + ww * S.Rmz.a[i]
               - 0.5 * (uu * uu + vv * vv + ww * ww) * S.Rr.a[i];
            sc += std::fabs(uu * S.Rmx.a[i]) + std::fabs(vv * S.Rmy.a[i]) + std::fabs(ww * S.Rmz.a[i]);
        }
        return std::pair<double, double>(s / M, std::fabs(s) / sc);
    };
    // CONVECTIVE KE rate (pressure off): must be round-off, IC-independent --
    // the discrete proof that the split convection produces NO spurious KE.
    S.with_pressure = false; auto kc = dkedt();
    std::printf("   convective dKE/dt   = %.2e  (rel %.2e)  -- split convection conserves KE\n",
                kc.first, kc.second);
    S.with_pressure = true;
    const double dt = 0.2 * h / S.max_wavespeed();
    const int nsteps = 300;
    for (int n = 0; n < nsteps; ++n) S.step(dt);
    std::printf("   %d steps, dt=%.4f, min(rho)=%.4f\n", nsteps, dt, S.min_density());
    std::printf("   |dMass|     = %.2e   (round-off: mass conserved)\n", std::fabs(S.total(S.rho) - m0));
    std::printf("   |dMom_x|    = %.2e   (round-off: momentum conserved)\n", std::fabs(S.total(S.mx) - px0));
    std::printf("   |dMom_y|    = %.2e\n", std::fabs(S.total(S.my) - py0));
    std::printf("   |dMom_z|    = %.2e\n", std::fabs(S.total(S.mz) - pz0));
    std::printf("   |dEnergy|   = %.2e   (round-off: TOTAL energy conserved -- the\n"
                "                            inviscid invariant; KE alone is not, it\n"
                "                            trades with internal energy by pdV work)\n",
                std::fabs(S.total(S.Egy) - e0));
    std::printf("\n");
}

// ----------------------------------------------------------------------------
// (2) isentropic vortex
// ----------------------------------------------------------------------------
struct Vortex {
    double gamma, beta, uinf, vinf, L, x0, y0;
    void state(double x, double y, double t, double& rho, double& u, double& v, double& p) const {
        const double cx = x0 + uinf * t, cy = y0 + vinf * t;
        double dx = x - cx; dx -= L * std::round(dx / L);
        double dy = y - cy; dy -= L * std::round(dy / L);
        const double r2 = dx * dx + dy * dy;
        const double ex = std::exp((1.0 - r2) * 0.5);
        const double du = -beta / (2 * PI) * dy * ex;
        const double dv =  beta / (2 * PI) * dx * ex;
        const double dT = -(gamma - 1.0) * beta * beta / (8 * gamma * PI * PI) * std::exp(1.0 - r2);
        const double T = 1.0 + dT;
        rho = std::pow(T, 1.0 / (gamma - 1.0));
        u = uinf + du; v = vinf + dv;
        p = std::pow(rho, gamma);                 // isentropic p = rho^gamma
    }
};

static double vortex_error(int N, int p, double Tend, double& measured_h) {
    const double gamma = 1.4, L = 10.0;
    Vortex vx{gamma, 5.0, 1.0, 1.0, L, 5.0, 5.0};
    SimConfig c = periodic_cfg(N, p, false, L);
    CompressibleNS S(c, gamma);
    const double h = L / N; measured_h = h;
    auto set_state = [&](double t, CompressibleNS& So) {
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            const double x = i * h, y = j * h;
            double rho, u, v, p_; vx.state(x, y, t, rho, u, v, p_);
            const size_t id = So.rho.idx(i, j, 0);
            So.rho.a[id] = rho; So.mx.a[id] = rho * u; So.my.a[id] = rho * v; So.mz.a[id] = 0;
            So.Egy.a[id] = p_ / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    };
    set_state(0.0, S);
    // fixed small dt (same for all N -> temporal error is a constant floor, so
    // the order is read from the spatially-dominated coarse-to-medium slope).
    const double dt = 0.0025;
    const int nsteps = (int)std::lround(Tend / dt);
    for (int n = 0; n < nsteps; ++n) S.step(dt);
    // exact at Tend
    double err = 0.0, ref = 0.0;
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h;
        double rho, u, v, p_; vx.state(x, y, nsteps * dt, rho, u, v, p_);
        const size_t id = S.rho.idx(i, j, 0);
        const double d = S.rho.a[id] - rho;
        err += d * d; ref += rho * rho;
    }
    return std::sqrt(err / ref);
}

static void test_vortex() {
    std::printf("=================================================================\n");
    std::printf(" (2) Isentropic vortex: spatial order (rel-L2 of density)\n");
    std::printf("=================================================================\n");
    const double Tend = 1.0;
    for (int p : {2, 4, 6}) {
        const int formal = 2 * ((p + 1) / 2);      // BMC D1 order = 2*ceil(p/2)
        std::printf("  p=%d  (formal order %d)\n", p, formal);
        std::printf("     N        h          rel-L2        order\n");
        double prev_e = 0, prev_h = 0, rep_ord = 0;
        for (int N : {24, 32, 48, 64}) {
            double h; const double e = vortex_error(N, p, Tend, h);
            double ord = 0;
            if (prev_e > 0) ord = std::log(prev_e / e) / std::log(prev_h / h);
            if (N == 48) rep_ord = ord;                 // asymptotic, pre-floor
            std::printf("    %3d   %.5f   %.4e   %s%.2f\n", N, h, e,
                        prev_e > 0 ? "" : "   -- ", prev_e > 0 ? ord : 0.0);
            prev_e = e; prev_h = h;
        }
        std::printf("     => asymptotic order ~%.1f  (formal %d; finest N flattens at the\n"
                    "        ~1e-6 vortex periodicity floor for high p)\n\n", rep_ord, formal);
    }
}

// ----------------------------------------------------------------------------
// (3) viscous operator MMS:  rhs(mu) - rhs(0)  vs analytic d_j tau_ij
// ----------------------------------------------------------------------------
static double viscous_error(int N, int p, double& measured_h) {
    const double gamma = 1.4, L = 2 * PI, mu = 0.7, Rgas = 1.0, T0 = 1.0;
    SimConfig c = periodic_cfg(N, p, true, L);
    CompressibleNS S0(c, gamma, Rgas, 0.0);        // inviscid
    CompressibleNS Sm(c, gamma, Rgas, mu, 1e30);   // viscous, Pr huge -> kappa~0 (isolate stress)
    const double h = L / N; measured_h = h;
    // manufactured: rho=1, u=sin x sin y sin z, v=w=0, T=T0 const
    auto fill = [&](CompressibleNS& S) {
        for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            const double x = i * h, y = j * h, z = k * h;
            const double u = std::sin(x) * std::sin(y) * std::sin(z);
            const size_t id = S.rho.idx(i, j, k);
            S.rho.a[id] = 1.0; S.mx.a[id] = u; S.my.a[id] = 0; S.mz.a[id] = 0;
            S.Egy.a[id] = (1.0 * Rgas * T0) / (gamma - 1.0) + 0.5 * u * u;  // p=rho R T0
        }
    };
    fill(S0); fill(Sm);
    S0.rhs(); Sm.rhs();
    // analytic viscous momentum: visc_x=-10/3 mu u, visc_y=mu/3 u_xy, visc_z=mu/3 u_xz
    // analytic viscous energy: d_j b_j with b=(4/3 mu u u_x, mu u u_y, mu u u_z)
    double err = 0, ref = 0;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h, z = k * h;
        const double sx = std::sin(x), sy = std::sin(y), sz = std::sin(z);
        const double cx = std::cos(x), cy = std::cos(y), cz = std::cos(z);
        const double u = sx * sy * sz;
        const double ux = cx * sy * sz, uy = sx * cy * sz, uz = sx * sy * cz;
        const double uxx = -u, uyy = -u, uzz = -u;
        const double vx_mom = -10.0 / 3.0 * mu * u;
        const double vy_mom = mu / 3.0 * (cx * cy * sz);   // u_xy
        const double vz_mom = mu / 3.0 * (cx * sy * cz);   // u_xz
        // energy: d_x(4/3 mu u u_x)+d_y(mu u u_y)+d_z(mu u u_z)
        const double ve = 4.0 / 3.0 * mu * (ux * ux + u * uxx)
                        + mu * (uy * uy + u * uyy) + mu * (uz * uz + u * uzz);
        const size_t id = S0.rho.idx(i, j, k);
        const double dvx = Sm.Rmx.a[id] - S0.Rmx.a[id];
        const double dvy = Sm.Rmy.a[id] - S0.Rmy.a[id];
        const double dvz = Sm.Rmz.a[id] - S0.Rmz.a[id];
        const double dve = Sm.Re_.a[id] - S0.Re_.a[id];
        err += (dvx - vx_mom) * (dvx - vx_mom) + (dvy - vy_mom) * (dvy - vy_mom)
             + (dvz - vz_mom) * (dvz - vz_mom) + (dve - ve) * (dve - ve);
        ref += vx_mom * vx_mom + vy_mom * vy_mom + vz_mom * vz_mom + ve * ve;
        (void)x;
    }
    return std::sqrt(err / ref);
}

static void test_viscous() {
    std::printf("=================================================================\n");
    std::printf(" (3) Viscous operator MMS:  rhs(mu)-rhs(0)  vs analytic d_j tau_ij\n");
    std::printf("=================================================================\n");
    for (int p : {2, 4, 6}) {
        const int formal = 2 * ((p + 1) / 2);
        std::printf("  p=%d  (formal order %d)\n", p, formal);
        std::printf("     N        h          rel-L2        order\n");
        double prev_e = 0, prev_h = 0, last_ord = 0;
        for (int N : {16, 24, 32, 48}) {
            double h; const double e = viscous_error(N, p, h);
            double ord = 0;
            if (prev_e > 0) { ord = std::log(prev_e / e) / std::log(prev_h / h); last_ord = ord; }
            std::printf("    %3d   %.5f   %.4e   %s%.2f\n", N, h, e,
                        prev_e > 0 ? "" : "   -- ", prev_e > 0 ? ord : 0.0);
            prev_e = e; prev_h = h;
        }
        std::printf("     => measured ~%.1f  (formal %d)\n\n", last_ord, formal);
    }
}

// ----------------------------------------------------------------------------
// (4)/(7) full inviscid RHS MMS: discrete split RHS vs analytic Euler RHS.
//   channel=false: triperiodic.  channel=true: x,y periodic + z clamped wall.
// ----------------------------------------------------------------------------
static double inviscid_rhs_error(int N, int p, double& measured_h, bool channel) {
    const double Lz = channel ? 2.0 : 2 * PI;
    SimConfig c; c.p = p; c.route = Route::BMC; c.nu = 0.0;
    c.ax[0] = {N, 2 * PI, BC::Periodic, true};
    c.ax[1] = {N, 2 * PI, BC::Periodic, true};
    c.ax[2] = {N, Lz, channel ? BC::Clamped : BC::Periodic, true};
    CompressibleNS S(c, 1.4, /*R=*/1.0, /*mu=*/0.0);
    measured_h = Lz / N;
    const ManState ms;
    const std::vector<double> X = axis_nodes(N, 2 * PI, true, p);
    const std::vector<double> Y = axis_nodes(N, 2 * PI, true, p);
    const std::vector<double> Z = axis_nodes(N, Lz, !channel, p);
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        double C[5]; ms.cons(X[i], Y[j], Z[k], C);
        const size_t id = S.rho.idx(i, j, k);
        S.rho.a[id] = C[0]; S.mx.a[id] = C[1]; S.my.a[id] = C[2]; S.mz.a[id] = C[3]; S.Egy.a[id] = C[4];
    }
    S.rhs();
    double err = 0, ref = 0;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        double R[5]; ms.euler_rhs(X[i], Y[j], Z[k], R);
        const size_t id = S.rho.idx(i, j, k);
        const double e[5] = {S.Rr.a[id] - R[0], S.Rmx.a[id] - R[1], S.Rmy.a[id] - R[2],
                             S.Rmz.a[id] - R[3], S.Re_.a[id] - R[4]};
        for (int m = 0; m < 5; ++m) { err += e[m] * e[m]; ref += R[m] * R[m]; }
    }
    return std::sqrt(err / ref);
}
static double full_inviscid_error(int N, int p, double& h) { return inviscid_rhs_error(N, p, h, false); }
static double channel_inviscid_error(int N, int p, double& h) { return inviscid_rhs_error(N, p, h, true); }

static void order_sweep(const char* title, double (*ef)(int, int, double&),
                        std::initializer_list<int> Ns) {
    for (int p : {2, 4, 6}) {
        const int formal = 2 * ((p + 1) / 2);
        std::printf("  p=%d  (formal order %d)\n     N        h          rel-L2        order\n", p, formal);
        double prev_e = 0, prev_h = 0, last = 0;
        for (int N : Ns) {
            double h; const double e = ef(N, p, h);
            double ord = 0;
            if (prev_e > 0) { ord = std::log(prev_e / e) / std::log(prev_h / h); last = ord; }
            std::printf("    %3d   %.5f   %.4e   %s%.2f\n", N, h, e,
                        prev_e > 0 ? "" : "   -- ", prev_e > 0 ? ord : 0.0);
            prev_e = e; prev_h = h;
        }
        std::printf("     => measured ~%.1f  (formal %d)\n\n", last, formal);
    }
    (void)title;
}

// ----------------------------------------------------------------------------
// (5) heat conduction MMS: energy RHS with u=0 equals kappa lap T
// ----------------------------------------------------------------------------
static double heat_error(int N, int p, double& measured_h) {
    const double gamma = 1.4, L = 2 * PI, R = 1.0, mu = 0.5, Pr = 0.7;
    const double kappa = mu * gamma * R / ((gamma - 1.0) * Pr);
    SimConfig c = periodic_cfg(N, p, true, L);
    CompressibleNS S(c, gamma, R, mu, Pr);
    const double h = L / N; measured_h = h;
    const Prim PT{1.0, 0.3, true, false, false};         // T = 1 + 0.3 sin x cos y cos z
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h, z = k * h;
        double T, tx, ty, tz; PT.at(x, y, z, T, tx, ty, tz);
        const double rho = 1.0, p_ = rho * R * T;        // u=v=w=0
        const size_t id = S.rho.idx(i, j, k);
        S.rho.a[id] = rho; S.mx.a[id] = S.my.a[id] = S.mz.a[id] = 0.0;
        S.Egy.a[id] = p_ / (gamma - 1.0);
    }
    S.rhs();
    double err = 0, ref = 0;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const double x = i * h, y = j * h, z = k * h;
        double T, tx, ty, tz; PT.at(x, y, z, T, tx, ty, tz);
        const double lapT = -3.0 * (T - 1.0);            // lap(sin x cos y cos z) = -3*(.)
        const double RE = kappa * lapT;                  // only heat term survives (u=0)
        const double e = S.Re_.a[S.rho.idx(i, j, k)] - RE;
        err += e * e; ref += RE * RE;
    }
    return std::sqrt(err / ref);
}

// ----------------------------------------------------------------------------
// (6) acoustic propagation: a small right-moving pulse translates at c=sqrt(gamma p/rho)
// ----------------------------------------------------------------------------
static void test_acoustic() {
    std::printf("=================================================================\n");
    std::printf(" (6) Acoustic propagation: right-moving pulse speed vs c=sqrt(gamma p/rho)\n");
    std::printf("=================================================================\n");
    const double gamma = 1.4, R = 1.0, L = 2 * PI, rho0 = 1.0, p0 = 1.0;
    const double c = std::sqrt(gamma * p0 / rho0), eps = 1e-3, sig = 0.4, x0 = PI;
    const int nx = 200, nt = 8, p = 4;
    SimConfig cfg; cfg.p = p; cfg.route = Route::BMC; cfg.nu = 0.0;
    cfg.ax[0] = {nx, L, BC::Periodic, true};
    cfg.ax[1] = {nt, L, BC::Periodic, true};
    cfg.ax[2] = {nt, L, BC::Periodic, true};
    CompressibleNS S(cfg, gamma, R, 0.0);
    const double h = L / nx;
    auto pulse = [&](double x) {
        double d = x - x0; d -= L * std::round(d / L); return eps * std::exp(-(d * d) / (sig * sig));
    };
    for (int k = 0; k < nt; ++k) for (int j = 0; j < nt; ++j) for (int i = 0; i < nx; ++i) {
        const double pp = pulse(i * h);                  // right-moving: rho'=p'/c^2, u=p'/(rho0 c)
        const double rho = rho0 + pp / (c * c), u = pp / (rho0 * c), pr = p0 + pp;
        const size_t id = S.rho.idx(i, j, k);
        S.rho.a[id] = rho; S.mx.a[id] = rho * u; S.my.a[id] = 0; S.mz.a[id] = 0;
        S.Egy.a[id] = pr / (gamma - 1.0) + 0.5 * rho * u * u;
    }
    auto centroid = [&]() {                              // mass-weighted centroid of p' along x
        double num = 0, den = 0;
        for (int i = 0; i < nx; ++i) {
            const double pp = (gamma - 1.0) * (S.Egy.a[S.rho.idx(i, 0, 0)]
                              - 0.5 * S.mx.a[S.rho.idx(i, 0, 0)] * S.mx.a[S.rho.idx(i, 0, 0)]
                                / S.rho.a[S.rho.idx(i, 0, 0)]) - p0;
            num += i * h * pp; den += pp;
        }
        return num / den;
    };
    const double dt = 0.4 * h / c;
    const double T = 1.0; const int nsteps = (int)std::lround(T / dt);
    const double c0_cent = centroid();
    for (int n = 0; n < nsteps; ++n) S.step(dt);
    const double cT = centroid();
    const double cnum = (cT - c0_cent) / (nsteps * dt);
    std::printf("   exact c = %.5f,  measured c = %.5f,  rel err = %.2e  (%d steps)\n\n",
                c, cnum, std::fabs(cnum - c) / c, nsteps);
}

int main() {
    std::printf("\n COMPRESSIBLE NS -- verification suite\n\n");
    test_conservation();
    test_vortex();
    test_viscous();
    std::printf("=================================================================\n");
    std::printf(" (4) Full inviscid RHS MMS: discrete split RHS vs analytic Euler RHS\n");
    std::printf("=================================================================\n");
    order_sweep("full", full_inviscid_error, {16, 24, 32, 48});
    std::printf("=================================================================\n");
    std::printf(" (5) Heat conduction MMS: energy RHS (u=0) vs analytic kappa lap T\n");
    std::printf("=================================================================\n");
    order_sweep("heat", heat_error, {16, 24, 32, 48});
    std::printf("=================================================================\n");
    std::printf(" (7) CHANNEL (z-wall) inviscid RHS MMS: discrete split RHS vs analytic\n");
    std::printf("     (x,y periodic + z clamped wall; boundary BMC/BMG/BMF for compressible)\n");
    std::printf("=================================================================\n");
    order_sweep("channel", channel_inviscid_error, {16, 24, 32, 48});
    test_acoustic();
    std::printf("Done.\n");
    return 0;
}
