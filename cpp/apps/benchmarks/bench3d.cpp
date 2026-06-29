// bench3d.cpp
//
// A true 3D PDE problem on the B-spline mimetic operators (3d_to_1d reduction):
//
//   1. CORRECTNESS. The discrete Laplacian of a triperiodic mode reproduces the
//      exact -(k1^2+k2^2+k3^2) eigenvalue; the max error confirms the 3D operator
//      is assembled and applied correctly and shows its accuracy.
//
//   2. SPEED. Throughput of the 3D Laplacian and of a linear advection-diffusion
//      right-hand side / RK3 step on grids up to 256^3, with OpenMP scaling.
//
// Everything is the directional collapse: a 3D apply is a stack of independent
// 1D line solves, parallel across all cores. No dense or 3D operator is formed.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include "bm3d.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
static int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

enum Family { FAMILY_BMC, FAMILY_BMG, FAMILY_BMF };

struct Options {
    Family family = FAMILY_BMC;
    int p = 4;
    int single_N = 0;
    bool periodic = true;
    bool compare_batching = true;
};

static const char* family_name(Family f) {
    if (f == FAMILY_BMC) return "BMC";
    if (f == FAMILY_BMG) return "BMG";
    return "BMF";
}

static OpKind d1_kind(Family f) { return (f == FAMILY_BMG) ? BMG_D1 : BMC_D1; }
static OpKind d2_kind(Family f) { return (f == FAMILY_BMG) ? BMG_D2 : BMC_D2; }

static bool usable_periodic_size(int N, int p) {
    return N > 2 * (p + 1);
}

static void print_usage(const char* exe) {
    std::printf("usage: %s [--op bmc|bmg] [--p degree] [--N grid] [--domain periodic] [--no-batching-compare]\n", exe);
    std::printf("\n");
    std::printf("3D timed PDE kernels currently support periodic same-grid BMC and BMG.\n");
    std::printf("BMC uses the optimized batched y/z path; BMG uses the generic configurable path.\n");
    std::printf("Use operator_probe for bounded BMG/BMF dense operator construction.\n");
}

static int parse_options(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            print_usage(argv[0]);
            return 1;
        } else if (std::strcmp(a, "--op") == 0 || std::strcmp(a, "--family") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after %s\n", a); return -1; }
            if (std::strcmp(argv[i], "bmc") == 0 || std::strcmp(argv[i], "BMC") == 0) opt.family = FAMILY_BMC;
            else if (std::strcmp(argv[i], "bmg") == 0 || std::strcmp(argv[i], "BMG") == 0) opt.family = FAMILY_BMG;
            else if (std::strcmp(argv[i], "bmf") == 0 || std::strcmp(argv[i], "BMF") == 0) opt.family = FAMILY_BMF;
            else { std::fprintf(stderr, "unknown operator family: %s\n", argv[i]); return -1; }
        } else if (std::strcmp(a, "--p") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after --p\n"); return -1; }
            opt.p = std::atoi(argv[i]);
            if (opt.p < 1) { std::fprintf(stderr, "--p must be positive\n"); return -1; }
        } else if (std::strcmp(a, "--N") == 0 || std::strcmp(a, "--n") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after %s\n", a); return -1; }
            opt.single_N = std::atoi(argv[i]);
            if (opt.single_N < 4) { std::fprintf(stderr, "--N must be at least 4\n"); return -1; }
        } else if (std::strcmp(a, "--domain") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after --domain\n"); return -1; }
            if (std::strcmp(argv[i], "periodic") == 0) opt.periodic = true;
            else if (std::strcmp(argv[i], "bounded") == 0 || std::strcmp(argv[i], "boundary") == 0) opt.periodic = false;
            else { std::fprintf(stderr, "unknown domain: %s\n", argv[i]); return -1; }
        } else if (std::strcmp(a, "--no-batching-compare") == 0) {
            opt.compare_batching = false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a);
            return -1;
        }
    }
    return 0;
}

static void apply_laplacian(const Deriv3D& D, const Options& opt,
                            const Field3D& u, Field3D& lap) {
    if (opt.family == FAMILY_BMC) D.laplacian(u, lap);
    else D.laplacian_generic(u, lap, d2_kind(opt.family));
}

static void apply_rhs(const Deriv3D& D, const Options& opt,
                      const Field3D& u, Field3D& rhs, Field3D& scratch,
                      double cx, double cy, double cz, double nu) {
    if (opt.family == FAMILY_BMC) D.advdiff_rhs(u, rhs, cx, cy, cz, nu);
    else D.advdiff_rhs_generic(u, rhs, scratch, cx, cy, cz, nu,
                               d1_kind(opt.family), d2_kind(opt.family));
}

// Fill u with a triperiodic mode cos(k1 x) cos(k2 y) cos(k3 z) on [0, 2pi)^3.
static void fill_mode(Field3D& u, int k1, int k2, int k3) {
    const double hx = 2 * PI / u.nx, hy = 2 * PI / u.ny, hz = 2 * PI / u.nz;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < u.nz; ++k)
        for (int j = 0; j < u.ny; ++j) {
            const double cy = std::cos(k2 * j * hy), cz = std::cos(k3 * k * hz);
            for (int i = 0; i < u.nx; ++i)
                u.a[u.idx(i, j, k)] = std::cos(k1 * i * hx) * cy * cz;
        }
}

// ---------------------------------------------------------------------------
// 1. 3D Laplacian correctness on a triperiodic eigenmode.
// ---------------------------------------------------------------------------
static void check_3d(const Options& opt) {
    std::printf("--- 3D Laplacian on a triperiodic mode cos(2x)cos(3y)cos(4z) ---\n");
    std::printf("   exact eigenvalue lambda = -(4+9+16) = -29\n");
    std::printf("   route=%s  p=%d  path=%s\n", family_name(opt.family), opt.p,
                opt.family == FAMILY_BMC ? "batched BMC" : "generic configurable");
    std::printf("   p    N     max|lap u - lambda u|\n");
    const int k1 = 2, k2 = 3, k3 = 4;
    const double lam = -(k1 * k1 + k2 * k2 + k3 * k3);
    for (int N : {24, 48}) {
        if (!usable_periodic_size(N, opt.p)) {
            std::printf("  %2d  %3d    skipped (degree too high for this periodic grid)\n", opt.p, N);
            continue;
        }
        Deriv3D D(N, N, N, opt.p, 2 * PI, 2 * PI, 2 * PI, true);
        Field3D u(N, N, N), lap(N, N, N);
        fill_mode(u, k1, k2, k3);
        apply_laplacian(D, opt, u, lap);
        double err = 0.0;
        for (size_t i = 0; i < u.size(); ++i)
            err = std::fmax(err, std::fabs(lap.a[i] - lam * u.a[i]));
        std::printf("  %2d  %3d    %.3e\n", opt.p, N, err);
    }
}

// ---------------------------------------------------------------------------
// 2. Throughput of the 3D Laplacian and an advection-diffusion RK3 step.
// ---------------------------------------------------------------------------
static void benchmark_3d(const Options& opt) {
    const int p = opt.p;
    std::printf("\n--- 3D throughput, %s degree p=%d, %d threads ---\n",
                family_name(opt.family), p, max_threads());
    std::printf("   path: %s\n", opt.family == FAMILY_BMC
                ? "optimized BMC with batched y/z line solves"
                : "generic configurable same-grid line solves");

    std::printf("\n   Laplacian  (out = d2x + d2y + d2z):\n");
    std::printf("     grid        points     time/apply     throughput      ns/point\n");
    std::vector<int> grids;
    if (opt.single_N > 0) grids.push_back(opt.single_N);
    else if (opt.family == FAMILY_BMC) grids = {128, 192, 256};
    else grids = {64, 96, 128};
    for (int N : grids) {
        if (!usable_periodic_size(N, p)) {
            std::printf("     %3d^3   skipped (degree too high for this periodic grid)\n", N);
            continue;
        }
        Deriv3D D(N, N, N, p, 2 * PI, 2 * PI, 2 * PI, true);
        Field3D u(N, N, N), lap(N, N, N);
        fill_mode(u, 2, 3, 4);
        apply_laplacian(D, opt, u, lap);                   // warm up
        const int reps = (N <= 128) ? 8 : (N <= 192 ? 6 : 4);
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) apply_laplacian(D, opt, u, lap);
        const double t = ms_since(t0) / reps;
        const double pts = static_cast<double>(N) * N * N;
        std::printf("     %3d^3   %9.1f M   %8.1f ms   %8.0f M pts/s   %6.2f\n",
                    N, pts / 1e6, t, pts / 1e6 / (t / 1e3), t * 1e6 / pts);
    }

    if (opt.family == FAMILY_BMC && opt.compare_batching) {
        const int N = (opt.single_N > 0) ? opt.single_N : 128;
        if (usable_periodic_size(N, p)) {
            Deriv3D D(N, N, N, p, 2 * PI, 2 * PI, 2 * PI, true);
            Field3D u(N, N, N), lap_fast(N, N, N), lap_generic(N, N, N);
            fill_mode(u, 2, 3, 4);
            D.laplacian(u, lap_fast);
            D.laplacian_generic(u, lap_generic, BMC_D2);
            const double err = max_abs_diff(lap_fast.a, lap_generic.a);
            const int reps = 4;
            auto t0 = Clock::now();
            for (int r = 0; r < reps; ++r) D.laplacian(u, lap_fast);
            const double t_fast = ms_since(t0) / reps;
            t0 = Clock::now();
            for (int r = 0; r < reps; ++r) D.laplacian_generic(u, lap_generic, BMC_D2);
            const double t_generic = ms_since(t0) / reps;
            std::printf("\n   Batching comparison at %d^3:\n", N);
            std::printf("     optimized batched: %.1f ms\n", t_fast);
            std::printf("     generic strided:   %.1f ms\n", t_generic);
            std::printf("     speedup: %.2fx   agreement: %.2e\n", t_generic / t_fast, err);
        }
    }

    // Parallel scaling of the Laplacian at a fixed grid.
    {
        const int N = opt.single_N > 0 ? opt.single_N : (opt.family == FAMILY_BMC ? 192 : 96);
        if (!usable_periodic_size(N, p)) {
            std::printf("\n   Laplacian parallel scaling skipped at %d^3 (degree too high)\n", N);
        } else {
            Deriv3D D(N, N, N, p, 2 * PI, 2 * PI, 2 * PI, true);
            Field3D u(N, N, N), lap(N, N, N);
            fill_mode(u, 2, 3, 4);
            const double pts = static_cast<double>(N) * N * N;
            std::printf("\n   Laplacian parallel scaling at %d^3:\n", N);
#ifdef _OPENMP
            int saved = omp_get_max_threads();
            omp_set_num_threads(1);
#endif
            apply_laplacian(D, opt, u, lap);
            auto t0 = Clock::now();
            for (int r = 0; r < 2; ++r) apply_laplacian(D, opt, u, lap);
            double t1 = ms_since(t0) / 2;
#ifdef _OPENMP
            omp_set_num_threads(saved);
#endif
            apply_laplacian(D, opt, u, lap);
            t0 = Clock::now();
            for (int r = 0; r < 6; ++r) apply_laplacian(D, opt, u, lap);
            double tp = ms_since(t0) / 6;
            std::printf("     1 thread:   %7.1f ms   (%.0f M pts/s)\n", t1, pts / 1e6 / (t1 / 1e3));
            std::printf("     %2d threads: %7.1f ms   (%.0f M pts/s)   speedup %.1fx\n",
                        max_threads(), tp, pts / 1e6 / (tp / 1e3), t1 / tp);
        }
    }

    // Advection-diffusion RK3 step: du/dt = -c.grad u + nu lap u.
    {
        const int N = opt.single_N > 0 ? opt.single_N : (opt.family == FAMILY_BMC ? 256 : 128);
        if (!usable_periodic_size(N, p)) {
            std::printf("\n   Advection-diffusion skipped at %d^3 (degree too high)\n", N);
            return;
        }
        Deriv3D D(N, N, N, p, 2 * PI, 2 * PI, 2 * PI, true);
        Field3D u(N, N, N), k1(N, N, N), k2(N, N, N), tmp(N, N, N), scratch(N, N, N);
        fill_mode(u, 2, 3, 4);
        const double cx = 1.0, cy = 0.5, cz = -0.7, nu = 0.01, dt = 1e-3;
        const double pts = static_cast<double>(N) * N * N;

        apply_rhs(D, opt, u, k1, scratch, cx, cy, cz, nu);      // warm up
        auto t0 = Clock::now();
        const int reps = opt.family == FAMILY_BMC ? 6 : 3;
        for (int r = 0; r < reps; ++r) apply_rhs(D, opt, u, k1, scratch, cx, cy, cz, nu);
        double t_rhs = ms_since(t0) / reps;

        // SSP-RK3 step (3 RHS evals + combines); timed end to end.
        auto rk3 = [&]() {
            apply_rhs(D, opt, u, k1, scratch, cx, cy, cz, nu); // u1 = u + dt k1
            tmp = u; axpy(tmp, dt, k1);
            apply_rhs(D, opt, tmp, k2, scratch, cx, cy, cz, nu); // u2 = 3/4 u + 1/4(u1 + dt k2)
            for (size_t i = 0; i < u.size(); ++i)
                tmp.a[i] = 0.75 * u.a[i] + 0.25 * (tmp.a[i] + dt * k2.a[i]);
            apply_rhs(D, opt, tmp, k2, scratch, cx, cy, cz, nu); // u = 1/3 u + 2/3(u2 + dt k2)
            for (size_t i = 0; i < u.size(); ++i)
                u.a[i] = (1.0 / 3.0) * u.a[i] + (2.0 / 3.0) * (tmp.a[i] + dt * k2.a[i]);
        };
        rk3();                                                 // warm up
        t0 = Clock::now();
        const int steps = opt.family == FAMILY_BMC ? 4 : 2;
        for (int r = 0; r < steps; ++r) rk3();
        double t_step = ms_since(t0) / steps;

        std::printf("\n   Advection-diffusion, %s p=%d at %d^3 (%.1f M points):\n",
                    family_name(opt.family), p, N, pts / 1e6);
        std::printf("     RHS eval:   %6.1f ms   (%.0f M pts/s)\n", t_rhs, pts / 1e6 / (t_rhs / 1e3));
        std::printf("     RK3 step:   %6.1f ms   (%.0f steps/s)\n", t_step, 1000.0 / t_step);
        std::printf("     => 1000 steps on %d^3 in about %.1f s\n", N, 1000.0 * t_step / 1e3);
    }
}

int main(int argc, char** argv) {
    Options opt;
    const int parsed = parse_options(argc, argv, opt);
    if (parsed > 0) return 0;
    if (parsed < 0) { print_usage(argv[0]); return 1; }

    if (!opt.periodic) {
        std::fprintf(stderr, "bench3d timed PDE kernels currently support only --domain periodic.\n");
        std::fprintf(stderr, "Use ./operator_probe --domain bounded --family bmg|bmf ... for boundary operators.\n");
        return 1;
    }
    if (opt.family == FAMILY_BMF) {
        std::fprintf(stderr, "BMF is a bounded staggered route and is not a same-grid 3D timed PDE path here.\n");
        std::fprintf(stderr, "Use ./operator_probe --domain bounded --family bmf ... to build BMF D1/D2 operators.\n");
        return 1;
    }

    std::printf("==================================================================\n");
    std::printf(" 3D B-spline mimetic PDE engine: correctness + speed\n");
    std::printf(" (%d cores, OpenMP across independent 1D lines)\n", max_threads());
    std::printf("==================================================================\n\n");
    check_3d(opt);
    benchmark_3d(opt);
    std::printf("\nDone.\n");
    return 0;
}
