// mms_check.cpp
//
// Method of Manufactured Solutions (MMS) verification for every B-spline
// mimetic derivative operator, at arbitrary spline degree:
//
//   PERIODIC, flux route     BMC : D1 = B' B^{-1},   D2 = B'' B^{-1}
//   PERIODIC, Galerkin route BMG : D1 = B M^{-1}K B^{-1}, D2 = -B M^{-1}S B^{-1}
//   BOUNDED,  flux route     BMF : D1 = G (staggered),    D2 = Dv G
//   BOUNDED,  Galerkin route BMG : D1 = B M^{-1}K B^{-1}, D2 = H^{-1}(-D^T H D + dD)
//
// The program runs in four stages:
//   1. Structural identity audit (Green, SBP, skew) at machine precision.
//   2. Periodic collapse identity audit  BMC_{2p+1} = BMG_p  (D1 and D2).
//   3. MMS convergence sweeps: a known smooth u is sampled, differentiated by
//      each operator, and the max-norm error is measured against the exact
//      derivative on a refined sequence of grids. The observed (numerical)
//      order is the log-log slope; the formal order is the construction value.
//
//   4. Fixed-grid spectral-like periodic sweep: choose a resolved Fourier mode
//      and raise p until the error hits round-off / conditioning.
//
// Stage 3 writes data/mms_convergence.csv, which figures/plot_convergence.py
// turns into the convergence figures (each curve annotated with degree, formal
// order, and measured order). Stage 4 writes data/mms_spectral.csv for a
// fixed-grid, high-degree periodic sweep.

#include <cstdio>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <sys/stat.h>
#include "bm_operators.h"

static const double PI = 3.14159265358979323846;
static const char* DATA_DIR = "data";

// ---------------------------------------------------------------------------
// Formal (construction) orders, by route.
//
//   periodic BMC : parity ladder  D1 = 2 ceil(p/2),  D2 = 2 floor(p/2).
//   periodic BMG : superconvergent via the collapse BMG_p = BMC_{2p+1}, so
//                  D1 = 2(p+1) and D2 = 2p (effectively spectral for large p).
//   bounded  BMF/BMG : the wall row caps the order at the spline degree, so
//                  D1 = p and the compatible (wide) D2 = p-1.
// ---------------------------------------------------------------------------
static int formal_order(const char* family, bool periodic, int deriv, int p) {
    if (periodic) {
        if (family[2] == 'C')                              // BMC
            return (deriv == 1) ? 2 * ((p + 1) / 2) : 2 * (p / 2);
        return (deriv == 1) ? 2 * (p + 1) : 2 * p;        // BMG (collapse)
    }
    return (deriv == 1) ? p : (p - 1);                    // bounded BMF / BMG
}

// ---------------------------------------------------------------------------
// data/ helpers.
// ---------------------------------------------------------------------------
static void ensure_data_dir() {
    struct stat st;
    if (stat(DATA_DIR, &st) == 0) {
        if ((st.st_mode & S_IFDIR) != 0) return;
        std::fprintf(stderr, "error: %s exists but is not a directory\n", DATA_DIR);
        std::exit(1);
    }
    if (mkdir(DATA_DIR, 0775) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "error: could not create %s/: %s\n", DATA_DIR, std::strerror(errno));
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Manufactured field: a deliberately HARD, broadband solution
//
//     u(x) = exp( g(x) ),   g(x) = cos(k1 x) sin(k2 x).
//
// The exponential of a trig product has infinitely many Fourier modes, so it is
// not band-limited like a single sine. That stresses the operators across their
// whole resolved band and removes any chance of accidental exactness flattering
// the order. Exact derivatives come from the chain rule:
//     u'  = g' u,    u'' = (g'' + g'^2) u.
// The periodic field uses integer wavenumbers so it is C-infinity and 2pi-
// periodic; the bounded field uses higher wavenumbers on the short [0, L].
// ---------------------------------------------------------------------------
static void manufactured(double x, double k1, double k2, double& u, double& du, double& d2u) {
    const double c1 = std::cos(k1 * x), s1 = std::sin(k1 * x);
    const double c2 = std::cos(k2 * x), s2 = std::sin(k2 * x);
    const double g   = c1 * s2;
    const double gp  = -k1 * s1 * s2 + k2 * c1 * c2;
    const double gpp = -(k1 * k1 + k2 * k2) * c1 * s2 - 2.0 * k1 * k2 * s1 * c2;
    u   = std::exp(g);
    du  = gp * u;
    d2u = (gpp + gp * gp) * u;
}

// Max-norm error of D1 u and D2 u against the exact derivatives at the output
// node sets. Bounded routes are measured on the interior away from the very
// first/last node so a single endpoint row does not mask the interior order;
// this matches how wall-row order is reported in the paper.
struct ErrPair { double e1, e2; };
static ErrPair operator_error(const OperatorSet& os, double k1, double k2) {
    std::vector<double> u(os.in_nodes.size());
    double val, d1e, d2e;
    for (size_t i = 0; i < u.size(); ++i) { manufactured(os.in_nodes[i], k1, k2, val, d1e, d2e); u[i] = val; }
    std::vector<double> d1 = matvec(os.D1, u), d2 = matvec(os.D2, u);
    double e1 = 0, e2 = 0;
    for (size_t i = 0; i < d1.size(); ++i) {
        manufactured(os.d1_nodes[i], k1, k2, val, d1e, d2e);
        e1 = std::fmax(e1, std::fabs(d1[i] - d1e));
    }
    for (size_t i = 0; i < d2.size(); ++i) {
        manufactured(os.d2_nodes[i], k1, k2, val, d1e, d2e);
        e2 = std::fmax(e2, std::fabs(d2[i] - d2e));
    }
    return {e1, e2};
}

// Band-limited Fourier mode used only for the fixed-grid spectral-like sweep.
// This avoids the aliasing floor of the broadband manufactured field above:
// the exact spectral derivative is available on the same grid, so increasing p
// can show the cardinal B-spline's approach to the sinc/spectral limit.
static void fourier_mode(double x, int k, double& u, double& du, double& d2u) {
    const double s = std::sin(k * x);
    const double c = std::cos(k * x);
    u = s;
    du = k * c;
    d2u = -static_cast<double>(k * k) * s;
}

static ErrPair operator_error_fourier(const OperatorSet& os, int k) {
    std::vector<double> u(os.in_nodes.size());
    double val, d1e, d2e;
    for (size_t i = 0; i < u.size(); ++i) {
        fourier_mode(os.in_nodes[i], k, val, d1e, d2e);
        u[i] = val;
    }
    std::vector<double> d1 = matvec(os.D1, u), d2 = matvec(os.D2, u);
    double e1 = 0, e2 = 0;
    for (size_t i = 0; i < d1.size(); ++i) {
        fourier_mode(os.d1_nodes[i], k, val, d1e, d2e);
        e1 = std::fmax(e1, std::fabs(d1[i] - d1e));
    }
    for (size_t i = 0; i < d2.size(); ++i) {
        fourier_mode(os.d2_nodes[i], k, val, d1e, d2e);
        e2 = std::fmax(e2, std::fabs(d2[i] - d2e));
    }
    return {e1, e2};
}

// Worst structural-identity residual of an assembled scheme (used to flag the
// degree at which high-p conditioning starts to corrupt the assembly).
static double worst_identity(const OperatorSet& os) {
    double r = 0.0;
    for (auto& id : os.identities) r = std::fmax(r, id.second);
    return r;
}

// Measured order = LS slope of log(err) vs log(h) over the finest points of the
// asymptotic tail. With a broadband manufactured solution the coarse grids are
// pre-asymptotic and the transition overshoots, so we isolate the tail: from the
// finest grid above the round-off floor, walk coarse-ward while the error keeps
// growing (normal convergence), then fit the finest few points of that run. The
// plotting script uses the identical rule, so console and figures agree.
static double fitted_order(const std::vector<double>& h, const std::vector<double>& e) {
    const int n = (int)e.size();
    if (n < 2) return 0.0;
    int fine = -1;                                       // finest grid above the floor
    for (int i = n - 1; i >= 0; --i)
        if (std::isfinite(e[i]) && e[i] > 1e-12) { fine = i; break; }
    if (fine < 1) return 0.0;
    int coarse = fine;                                   // extend the monotonic tail
    for (int i = fine - 1; i >= 0; --i) {
        if (std::isfinite(e[i]) && e[i] > e[i + 1]) coarse = i; else break;
    }
    int lo = fine - 2;                                   // finest <= 3 points of the tail
    if (lo < coarse) lo = coarse;
    const int m = fine - lo + 1;
    if (m < 2) return 0.0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = lo; i <= fine; ++i) {
        const double X = std::log(h[i]), Y = std::log(e[i]);
        sx += X; sy += Y; sxx += X * X; sxy += X * Y;
    }
    return (m * sxy - sx * sy) / (m * sxx - sx * sx);
}

using Builder = std::function<OperatorSet(int, int, double)>;

// Run a refinement sweep for one route, print a table, and append CSV rows.
static void sweep(FILE* csv, const char* domain, const char* family,
                  const Builder& build, bool periodic, double L, double k1, double k2,
                  const std::vector<int>& pdegs, const std::vector<int>& ns) {
    std::printf("\n=== %s  %s  (manufactured solution sweep, k=(%.0f,%.0f)) ===\n",
                domain, family, k1, k2);
    for (int p : pdegs) {
        std::vector<double> hh, ee1, ee2;
        std::printf("  p=%d  formal: D1=%d D2=%d\n", p,
                    formal_order(family, periodic, 1, p), formal_order(family, periodic, 2, p));
        std::printf("      n        h          err(D1)     ord       err(D2)     ord\n");
        double ph = 0, pe1 = 0, pe2 = 0;
        for (size_t s = 0; s < ns.size(); ++s) {
            const int n = ns[s];
            if (n < p + 2) continue;
            OperatorSet os = build(n, p, L);
            const double h = periodic ? (L / n) : (L / (n - p));
            ErrPair er = operator_error(os, k1, k2);
            double o1 = (s > 0 && ph > 0) ? std::log(pe1 / er.e1) / std::log(ph / h) : 0.0;
            double o2 = (s > 0 && ph > 0 && pe2 > 0 && er.e2 > 0)
                            ? std::log(pe2 / er.e2) / std::log(ph / h) : 0.0;
            std::printf("   %5d  %.6f   %.4e  %6.2f   %.4e  %6.2f\n", n, h, er.e1, o1, er.e2, o2);
            std::fprintf(csv, "%s,%s,%d,%d,%.10e,%.10e,%.10e\n", domain, family, p, n, h, er.e1, er.e2);
            hh.push_back(h); ee1.push_back(er.e1); ee2.push_back(er.e2);
            ph = h; pe1 = er.e1; pe2 = er.e2;
        }
        std::printf("      fitted order:  D1=%.2f  D2=%.2f\n", fitted_order(hh, ee1), fitted_order(hh, ee2));
    }
}

// Spectral-like approach: hold the periodic grid N fixed and raise the spline
// degree. The cardinal B-spline approaches the spectral (sinc) limit as p
// grows, so a band-limited Fourier mode should fall geometrically toward
// machine precision until conditioning/noise takes over. We choose mode k=8 on
// N=64: 8 points per wavelength, far below Nyquist but still hard enough that
// low degrees visibly miss it.
static void spectral_sweep(double Lp) {
    std::printf("\n--- Spectral-like approach: error vs degree p at fixed N ---\n");
    FILE* csv = std::fopen("data/mms_spectral.csv", "w");
    if (!csv) { std::fprintf(stderr, "error: cannot open data/mms_spectral.csv\n"); return; }
    std::fprintf(csv, "domain,family,N,mode,p,points_per_wave,err_d1,err_d2,resid\n");

    struct Route { const char* domain; const char* family; bool periodic;
                   double L; int N; int mode; int pmax; Builder build; };
    const std::vector<Route> routes = {
        {"periodic", "BMC", true, Lp, 64, 8, 30, build_bmc_periodic},
        {"periodic", "BMG", true, Lp, 64, 8, 14, build_bmg_periodic},
    };
    for (const Route& r : routes) {
        const double ppw = static_cast<double>(r.N) / r.mode;
        std::printf("  %-8s %s  N=%d  mode k=%d  %.1f points/wavelength\n",
                    r.domain, r.family, r.N, r.mode, ppw);
        std::printf("      p      err(D1)      err(D2)     identity\n");
        for (int p = 2; p <= r.pmax; ++p) {
            if (r.periodic && 2 * (p + 1) > r.N) break;     // keep the stencil unaliased
            OperatorSet os = r.build(r.N, p, r.L);
            ErrPair er = operator_error_fourier(os, r.mode);
            const double resid = worst_identity(os);
            std::printf("   %4d   %.4e   %.4e   %.1e\n", p, er.e1, er.e2, resid);
            std::fprintf(csv, "%s,%s,%d,%d,%d,%.10e,%.10e,%.10e,%.10e\n",
                         r.domain, r.family, r.N, r.mode, p, ppw, er.e1, er.e2, resid);
        }
    }
    std::fclose(csv);
    std::printf("\nWrote data/mms_spectral.csv\n");
}

int main(int argc, char** argv) {
    const double Lp = 2.0 * PI, Lb = 2.0;
    const bool spectral_only = (argc > 1 && std::strcmp(argv[1], "--spectral-only") == 0);

    if (spectral_only) {
        ensure_data_dir();
        spectral_sweep(Lp);
        return 0;
    }

    std::printf("=================================================================\n");
    std::printf(" B-spline mimetic derivative operators -- identities + MMS check\n");
    std::printf("=================================================================\n");

    // --- Stage 1: structural identity audit ---------------------------------
    std::printf("\n--- Structural identities (machine precision) ---\n");
    auto audit = [](const OperatorSet& os) {
        std::printf("  %-4s %-8s p=%d:", os.name.c_str(), os.periodic ? "periodic" : "bounded", os.p);
        for (auto& id : os.identities) std::printf("  %s=%.1e", id.first.c_str(), id.second);
        std::printf("\n");
    };
    for (int p = 2; p <= 6; ++p) {
        audit(build_bmc_periodic(64, p, Lp));
        audit(build_bmg_periodic(64, p, Lp));
        audit(build_bmf_bounded(40, p, Lb));
        audit(build_bmg_bounded(40, p, Lb));
    }

    // --- Stage 2: periodic collapse identity BMC_{2p+1} = BMG_p -------------
    std::printf("\n--- Periodic collapse identity  BMC_(2p+1) = BMG_p ---\n");
    for (int p = 1; p <= 4; ++p) {
        OperatorSet a = build_bmc_periodic(48, 2 * p + 1, Lp);
        OperatorSet b = build_bmg_periodic(48, p, Lp);
        std::printf("  BMC_%d vs BMG_%d:  ||D1 diff||=%.2e   ||D2 diff||=%.2e\n",
                    2 * p + 1, p, max_abs_diff(a.D1, b.D1), max_abs_diff(a.D2, b.D2));
    }

    // --- Stage 3: MMS convergence sweeps ------------------------------------
    ensure_data_dir();
    FILE* csv = std::fopen("data/mms_convergence.csv", "w");
    if (!csv) { std::fprintf(stderr, "error: cannot open data/mms_convergence.csv\n"); return 1; }
    std::fprintf(csv, "domain,family,p,n,h,err_d1,err_d2\n");

    const std::vector<int> pdegs = {2, 3, 4, 5, 6};
    const std::vector<int> ns    = {16, 24, 32, 48, 64, 96, 128, 192};

    sweep(csv, "periodic", "BMC", build_bmc_periodic, true,  Lp, 2, 3, pdegs, ns);
    sweep(csv, "periodic", "BMG", build_bmg_periodic, true,  Lp, 2, 3, pdegs, ns);
    sweep(csv, "bounded",  "BMF", build_bmf_bounded,  false, Lb, 3, 4, pdegs, ns);
    sweep(csv, "bounded",  "BMG", build_bmg_bounded,  false, Lb, 3, 4, pdegs, ns);

    std::fclose(csv);
    std::printf("\nWrote data/mms_convergence.csv\n");

    // --- Stage 4: spectral-like approach -- error vs degree at FIXED grid ----
    spectral_sweep(Lp);
    return 0;
}
