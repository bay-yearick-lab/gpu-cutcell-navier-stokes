// config.h
//
// One place to describe a simulation. Everything downstream -- which operators
// to build, how many points, what degree, periodic or walled, 2D or 3D, which
// route -- is read from a SimConfig. Flip a field here and the whole solver
// reconfigures; nothing else needs editing. validate() refuses nonsense up front
// with a clear message so a bad config fails loudly at setup, never mid-run.

#ifndef CONFIG_H
#define CONFIG_H

#include <cstdio>

enum class BC    { Periodic, Clamped };          // per-axis boundary condition
// BM collocated routes; explicit-FD baselines; and the local high-resolution
// competitors -- compact Pade C6 and the dispersion-relation-preserving (DRP) scheme.
enum class Route { BMC, BMG, FD2, FD4, FD6, C6, DRP };

inline const char* bc_name(BC b) { return b == BC::Periodic ? "periodic" : "clamped"; }
inline const char* route_name(Route r) {
    switch (r) { case Route::BMC: return "BMC"; case Route::BMG: return "BMG";
                 case Route::FD2: return "FD2"; case Route::FD4: return "FD4"; case Route::FD6: return "FD6";
                 case Route::C6: return "C6(compact)"; default: return "DRP"; }
}

// One coordinate axis: resolution, physical length, boundary condition, and
// whether it participates at all (an inactive third axis gives a 2D run).
struct AxisSpec {
    int    N   = 32;
    double L   = 6.283185307179586;     // 2*pi
    BC     bc  = BC::Periodic;
    bool   on  = true;
};

struct SimConfig {
    AxisSpec ax[3];                     // x, y, z
    int      p     = 4;                 // spline degree (shared across axes)
    Route    route = Route::BMC;        // first-derivative route
    double   nu    = 1.0e-2;            // kinematic viscosity (for Navier-Stokes)

    int  ndim()        const { return ax[2].on ? 3 : 2; }
    bool periodic(int a) const { return ax[a].bc == BC::Periodic; }
    bool active(int a)   const { return ax[a].on; }

    // Robustness gate. Returns false and a reason on any unusable setting so the
    // caller can abort cleanly. The bounds match what the operators require.
    bool validate(const char*& why) const {
        if (p < 1) { why = "degree p must be >= 1"; return false; }
        if (!ax[0].on || !ax[1].on) { why = "axes x and y must be active"; return false; }
        for (int a = 0; a < 3; ++a) {
            if (!ax[a].on) continue;
            if (ax[a].L <= 0.0) { why = "axis length L must be positive"; return false; }
            if (ax[a].N < p + 2) { why = "each active axis needs N >= p + 2"; return false; }
            if (ax[a].bc == BC::Periodic && ax[a].N < 2 * (p + 1)) {
                why = "a periodic axis needs N >= 2(p+1)"; return false;
            }
        }
        return true;
    }

    void print() const {
        const char xyz[3] = {'x', 'y', 'z'};
        std::printf("SimConfig: %dD  p=%d  route=%s  nu=%.3g\n",
                    ndim(), p, route_name(route), nu);
        for (int a = 0; a < 3; ++a) {
            if (!ax[a].on) { std::printf("   %c: (inactive)\n", xyz[a]); continue; }
            std::printf("   %c: N=%-5d L=%.4g  %s\n", xyz[a], ax[a].N, ax[a].L, bc_name(ax[a].bc));
        }
    }
};

#endif // CONFIG_H
