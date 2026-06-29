// caa_dispersion.cpp
//
// PHASE 1 of the aeroacoustics ("vortex sound") story: the acoustic spatial
// dispersion of the B-spline mimetic operators vs matched-order explicit finite
// differences -- the basis for the claim that these schemes can carry the faint,
// short acoustic waves of computational aeroacoustics where standard high-order
// FD already buries them under numerical dispersion.
//
// Two views, both isolating SPATIAL dispersion:
//   (1) Exact modified wavenumber of the acoustic operator (2nd derivative):
//       apply d2 to cos(k x); the eigenvalue gives k_eff^2; report |k_eff^2/k^2-1|
//       vs points-per-wavelength (PPW). No time integrator.
//   (2) A time-marched 1D acoustic propagation sim (first-order system p_t=-c u_x,
//       u_t=-c p_x, RK4 at small dt so the error is spatial): a narrowband packet
//       at ~4.5 PPW is propagated 20 wavelengths and the waveform error reported.
// Schemes: BMG (spline mimetic Galerkin, the strong wave route) vs explicit E6/E8.

#include <cstdio>
#include <cmath>
#include <vector>
#include "bm_apply.h"

static const double PI = 3.14159265358979323846;

enum Scheme { BMG, E6, E8, C6, DRP };          // BMG vs explicit FD and the compact C6 / DRP schemes

// Apply the 1D second derivative along x (rows) and y (cols) of a periodic NxN
// field, out = lap(p). BMG uses the spline mimetic Galerkin d2; E6/E8 use the
// explicit centered periodic stencils.
struct Lap2D {
    int N; double h; Scheme sc; FastDeriv f;
    Lap2D(int N_, int p, double L, Scheme sc_)
        : N(N_), h(L / N_), sc(sc_), f(FastDeriv::build(true, N_, p, L)) {}

    void fd_line(const std::vector<double>& u, std::vector<double>& y) const {
        const double ih2 = 1.0 / (h * h);
        if (sc == E6) {
            const double c0 = -49.0/18, c1 = 3.0/2, c2 = -3.0/20, c3 = 1.0/90;
            for (int i = 0; i < N; ++i) {
                auto U = [&](int o){ return u[((i+o)%N+N)%N]; };
                y[i] = ih2 * (c0*U(0) + c1*(U(1)+U(-1)) + c2*(U(2)+U(-2)) + c3*(U(3)+U(-3)));
            }
        } else { // E8
            const double c0 = -205.0/72, c1 = 8.0/5, c2 = -1.0/5, c3 = 8.0/315, c4 = -1.0/560;
            for (int i = 0; i < N; ++i) {
                auto U = [&](int o){ return u[((i+o)%N+N)%N]; };
                y[i] = ih2 * (c0*U(0) + c1*(U(1)+U(-1)) + c2*(U(2)+U(-2)) + c3*(U(3)+U(-3)) + c4*(U(4)+U(-4)));
            }
        }
    }
    void d2_line(const std::vector<double>& u, std::vector<double>& y, std::vector<double>& a, std::vector<double>& b) const {
        if (sc == BMG)      f.bmg_d2(u, y, a, b);
        else if (sc == C6)  f.compact_d2(u, y);
        else if (sc == DRP) f.fd6_d2(u, y);            // DRP-family diffusion
        else                fd_line(u, y);
    }
};

// Modified wavenumber of the SECOND derivative: apply d2 to cos(k x_j); the
// eigenvalue is -k_eff^2 = (d2 cos_k)[0]. Returns |k_eff^2 / k^2 - 1| (relative
// phase-speed-squared error of the acoustic operator). Exact spatial dispersion,
// no time integrator.
static double d2_modwave_err(int N, int p, double L, Scheme sc, int m) {
    Lap2D lap(N, p, L, sc);
    std::vector<double> cosk(N), out(N), a(N), b(N);
    const double k = 2*PI*m/L, h = L/N;
    for (int j = 0; j < N; ++j) cosk[j] = std::cos(k * j * h);
    lap.d2_line(cosk, out, a, b);
    const double keff2 = -out[0];                 // cos_k[0] = 1
    return std::fabs(keff2 / (k*k) - 1.0);
}

// ---- 1D acoustic propagation sim (first-order system, RK4) -----------------
// p_t = -c u_x,  u_t = -c p_x ;  D1 from BMG (the main D1 route) or
// explicit E6/E8. RK4 in time at small dt, so the error is SPATIAL dispersion.
// A narrowband packet at ~4.5 PPW is propagated many wavelengths; we report the
// relative L2 waveform error -- the long-haul phase test as an actual simulation.
static std::vector<double> d1_apply(Scheme sc, const FastDeriv& f,
                                    const std::vector<double>& u, double h) {
    const int N = (int)u.size();
    if (sc == BMG) return f.bmg_d1(u);
    if (sc == C6)  { std::vector<double> y; f.compact_d1(u, y); return y; }
    if (sc == DRP) { std::vector<double> y; f.drp_d1(u, y); return y; }
    std::vector<double> y(N); const double ih = 1.0 / h;
    auto U = [&](int i, int o){ return u[((i+o)%N+N)%N]; };
    if (sc == E6) {
        const double c1 = 3.0/4, c2 = -3.0/20, c3 = 1.0/60;
        for (int i = 0; i < N; ++i) y[i] = ih*(c1*(U(i,1)-U(i,-1))+c2*(U(i,2)-U(i,-2))+c3*(U(i,3)-U(i,-3)));
    } else {
        const double c1 = 4.0/5, c2 = -1.0/5, c3 = 4.0/105, c4 = -1.0/280;
        for (int i = 0; i < N; ++i) y[i] = ih*(c1*(U(i,1)-U(i,-1))+c2*(U(i,2)-U(i,-2))+c3*(U(i,3)-U(i,-3))+c4*(U(i,4)-U(i,-4)));
    }
    return y;
}
// First-derivative (advection) modified wavenumber: apply d1 to sin(k x); since
// d1[sin(kx)]_0 = k_eff cos(0) = k_eff, return |k_eff/k - 1| (the phase-speed error
// of the advection operator). The headline resolution metric vs the best schemes.
static double d1_modwave_err(int N, int p, double L, Scheme sc, int m) {
    FastDeriv f = FastDeriv::build(true, N, p, L);
    std::vector<double> sk(N); const double k = 2*PI*m/L, h = L/N;
    for (int j = 0; j < N; ++j) sk[j] = std::sin(k * j * h);
    std::vector<double> out = d1_apply(sc, f, sk, h);
    return std::fabs(out[0] / k - 1.0);
}
static double acoustic1d(int N, int p, double L, Scheme sc, int m, double dist) {
    const double c = 1.0, h = L / N;
    FastDeriv f = FastDeriv::build(true, N, p, L);
    const double k0 = 2*PI*m/L, x0 = L/2, sig = L/10;
    auto wrap = [&](double x){ while (x < 0) x += L; while (x >= L) x -= L; return x; };
    auto pack = [&](double x){ double d = wrap(x - x0) - L/2; return std::exp(-(d*d)/(2*sig*sig)) * std::cos(k0*(x - x0)); };
    std::vector<double> pp(N), uu(N);
    for (int i = 0; i < N; ++i) { pp[i] = pack(i*h); uu[i] = pack(i*h); }   // rightward: u=p
    const double dt = 0.2 * h / c; const int nst = (int)std::lround(dist / c / dt);
    std::vector<double> kp1,ku1,kp2,ku2,kp3,ku3,kp4,ku4,tp(N),tu(N);
    auto rhs = [&](const std::vector<double>& P, const std::vector<double>& Uu,
                   std::vector<double>& dP, std::vector<double>& dU){
        dP = d1_apply(sc, f, Uu, h); for (double& z : dP) z *= -c;
        dU = d1_apply(sc, f, P,  h); for (double& z : dU) z *= -c;
    };
    for (int n = 0; n < nst; ++n) {
        rhs(pp, uu, kp1, ku1);
        for (int i=0;i<N;++i){ tp[i]=pp[i]+0.5*dt*kp1[i]; tu[i]=uu[i]+0.5*dt*ku1[i]; }
        rhs(tp, tu, kp2, ku2);
        for (int i=0;i<N;++i){ tp[i]=pp[i]+0.5*dt*kp2[i]; tu[i]=uu[i]+0.5*dt*ku2[i]; }
        rhs(tp, tu, kp3, ku3);
        for (int i=0;i<N;++i){ tp[i]=pp[i]+dt*kp3[i]; tu[i]=uu[i]+dt*ku3[i]; }
        rhs(tp, tu, kp4, ku4);
        for (int i=0;i<N;++i){ pp[i]+=dt/6*(kp1[i]+2*kp2[i]+2*kp3[i]+kp4[i]);
                               uu[i]+=dt/6*(ku1[i]+2*ku2[i]+2*ku3[i]+ku4[i]); }
    }
    const double travelled = nst * dt * c;
    double num = 0, den = 0;
    for (int i = 0; i < N; ++i) { const double e = pp[i] - pack(i*h - travelled); num += e*e; den += pack(i*h - travelled)*pack(i*h - travelled); }
    return std::sqrt(num / den);
}

int main() {
    const int N = 96; const double L = 2*PI;
    std::printf("=================================================================\n");
    std::printf(" RESOLUTION vs THE BEST LOCAL SCHEMES: advection-operator (1st-deriv)\n");
    std::printf(" modified wavenumber |k_eff/k - 1| vs PPW (exact; lower = resolves\n");
    std::printf(" shorter waves per point). C6 = compact Pade, DRP = dispersion-relation preserving.\n");
    std::printf("=================================================================\n");
    std::printf("\n  PPW   C6(6th)   DRP       BMG2(6th) BMG3(8th) BMG4(10th) BMG5(12th)\n");
    for (int m : {24, 16, 12, 10, 8, 6, 5, 4, 3}) {
        const double ppw = (double)N / m;
        std::printf("  %4.1f  %.2e  %.2e  %.2e  %.2e  %.2e   %.2e\n", ppw,
            d1_modwave_err(N, 2, L, C6, m), d1_modwave_err(N, 2, L, DRP, m),
            d1_modwave_err(N, 2, L, BMG, m), d1_modwave_err(N, 3, L, BMG, m),
            d1_modwave_err(N, 4, L, BMG, m), d1_modwave_err(N, 5, L, BMG, m));
    }
    std::printf("\n=================================================================\n");
    std::printf(" Acoustic operator (2nd-derivative) modified wavenumber error\n");
    std::printf(" |k_eff^2/k^2 - 1| vs PPW -- exact spatial dispersion (no time int)\n");
    std::printf("=================================================================\n");
    std::printf("\n  PPW   E6        E8        BMG3      BMG4      BMG5\n");
    for (int m : {24, 16, 12, 10, 8, 6, 5, 4, 3}) {
        const double ppw = (double)N / m;
        std::printf("  %4.1f  %.2e  %.2e  %.2e  %.2e  %.2e\n", ppw,
            d2_modwave_err(N, 3, L, E6, m), d2_modwave_err(N, 4, L, E8, m),
            d2_modwave_err(N, 3, L, BMG, m), d2_modwave_err(N, 4, L, BMG, m),
            d2_modwave_err(N, 5, L, BMG, m));
    }
    std::printf("\n 1D acoustic packet propagation (RK4 sim): rel L2 waveform error\n");
    std::printf("   carrier ~4.5 PPW, narrowband packet, propagated 20 wavelengths\n");
    std::printf("   N      BMG2 (=BMC5)   E6         E8        | BMG2 vs E6\n");
    for (int N : {128, 192, 256}) {
        const int m = (int)std::lround(N / 4.5);             // ~4.5 PPW carrier
        const double L = 2*PI, dist = 20.0 * (L / m);        // 20 wavelengths
        const double eg = acoustic1d(N, 2, L, BMG, m, dist);
        const double e6 = acoustic1d(N, 2, L, E6, m, dist);
        const double e8 = acoustic1d(N, 2, L, E8, m, dist);
        std::printf("  %4d   %.3e      %.3e  %.3e | %5.1fx\n", N, eg, e6, e8, e6/eg);
    }
    std::printf("\nDone.\n");
    return 0;
}
