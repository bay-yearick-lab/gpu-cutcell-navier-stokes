// bench.cpp
//
// Two jobs for the production application path (bm_apply.h):
//
//   1. CORRECTNESS. The reduce-solve-expand banded apply must reproduce the
//      dense operators of bm_operators.h to machine precision. If it does not,
//      the fast path is wrong and the timing below is meaningless.
//
//   2. SPEED. The whole point of the banded chain is that B and M are factored
//      ONCE and every apply costs O(n * bandwidth), versus O(n^2) for a dense
//      matvec (and O(n^3) just to materialize the dense operator). We time both
//      on the host and report the per-apply speedup, then push the fast path
//      to a line length where the dense operator could not even be stored.
//
// A 1D line of length n is the unit the whole tensor-product solver reduces to
// (the 3d_to_1d paper): a 3D field is millions of such line solves per step, so
// per-apply cost and one-time-factor reuse are exactly what set the wall clock.

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include "bm_operators.h"
#include "bm_apply.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static const double PI = 3.14159265358979323846;
using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<double> test_field(int n) {
    std::vector<double> u(n);
    for (int i = 0; i < n; ++i) u[i] = std::sin(0.7 * i) + 0.5 * std::cos(0.13 * i + 0.4);
    return u;
}

// ---------------------------------------------------------------------------
// 1. Correctness: fast apply vs dense operator, max-norm difference.
// ---------------------------------------------------------------------------
static void check_correctness() {
    const double Lp = 2.0 * PI, Lb = 2.0;
    std::printf("--- Fast reduce-solve-expand apply vs dense operator (machine precision) ---\n");
    std::printf("   route            p     ||fast - dense||_inf (D1)   (D2)\n");
    for (int p = 2; p <= 6; ++p) {
        const int np = 64, nb = 48;
        std::vector<double> up = test_field(np), ub = test_field(nb);

        FastDeriv fp = FastDeriv::build(true, np, p, Lp);
        OperatorSet bmc = build_bmc_periodic(np, p, Lp);
        OperatorSet bmgp = build_bmg_periodic(np, p, Lp);
        double e_bmc1 = max_abs_diff(fp.bmc_d1(up), matvec(bmc.D1, up));
        double e_bmc2 = max_abs_diff(fp.bmc_d2(up), matvec(bmc.D2, up));
        double e_bmg1 = max_abs_diff(fp.bmg_d1(up), matvec(bmgp.D1, up));
        double e_bmg2 = max_abs_diff(fp.bmg_d2_periodic(up), matvec(bmgp.D2, up));

        FastDeriv fb = FastDeriv::build(false, nb, p, Lb);
        OperatorSet bmgb = build_bmg_bounded(nb, p, Lb);
        OperatorSet bmf = build_bmf_bounded(nb, p, Lb);
        double e_bmgb1 = max_abs_diff(fb.bmg_d1(ub), matvec(bmgb.D1, ub));
        double e_bmf1  = max_abs_diff(fb.bmf_grad(ub), matvec(bmf.D1, ub));

        std::printf("   BMC  periodic   %d     %.2e                %.2e\n", p, e_bmc1, e_bmc2);
        std::printf("   BMG  periodic   %d     %.2e                %.2e\n", p, e_bmg1, e_bmg2);
        std::printf("   BMG  bounded    %d     %.2e                   -\n", p, e_bmgb1);
        std::printf("   BMF  bounded    %d     %.2e (gradient)        -\n", p, e_bmf1);
    }
}

// ---------------------------------------------------------------------------
// 2. Speed: dense matvec vs banded reduce-solve-expand for BMG D1 (periodic).
// ---------------------------------------------------------------------------
static void benchmark() {
    const double Lp = 2.0 * PI;
    const int p = 4;
    std::printf("\n--- Speed: BMG first derivative on a periodic line (degree p=%d) ---\n", p);
    std::printf("   apply = one D1 of a length-n field; setup is paid once and reused.\n\n");

    // Head to head at a size where materializing the dense operator is feasible.
    {
        const int n = 512, reps = 3000;
        std::vector<double> u = test_field(n);

        auto t0 = Clock::now();
        OperatorSet os = build_bmg_periodic(n, p, Lp);     // O(n^3): materialize D1
        double t_mat = ms_since(t0);

        volatile double sink = 0.0;
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) { std::vector<double> y = matvec(os.D1, u); sink += y[0]; }
        double t_dense = ms_since(t0) / reps;

        t0 = Clock::now();
        FastDeriv f = FastDeriv::build(true, n, p, Lp);     // O(n*bw^2): factor B, M once
        double t_setup = ms_since(t0);

        std::vector<double> y(n);
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) { f.bmg_d1(u, y); sink += y[0]; }
        double t_fast = ms_since(t0) / reps;

        std::printf("   n=%d:\n", n);
        std::printf("     setup (once):   dense materialize %.2f ms   |   banded factor %.3f ms\n",
                    t_mat, t_setup);
        std::printf("     per apply:      dense matvec %.4f ms        |   reduce-solve-expand %.4f ms\n",
                    t_dense, t_fast);
        std::printf("     per-apply speedup: %.0fx\n", t_dense / t_fast);
        (void)sink;
    }

    // The circulant storage is O(n*bw), so a single line scales to millions of
    // points. The dense operator is never formed (at n=2^20 it would need
    // n^2 * 8 bytes ~ 8 TB). Per-apply cost stays linear: ns/point ~ constant.
    std::printf("\n   reduce-solve-expand scaling (periodic, dense never formed):\n");
    for (int n : {4096, 32768, 262144, 1048576}) {
        std::vector<double> u = test_field(n), y(n);
        auto t0 = Clock::now();
        FastDeriv f = FastDeriv::build(true, n, p, Lp);
        double t_setup = ms_since(t0);
        const int reps = (n <= 32768) ? 200 : (n <= 262144 ? 40 : 8);
        volatile double sink = 0.0;
        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) { f.bmg_d1(u, y); sink += y[0]; }
        double t = ms_since(t0) / reps;
        std::printf("     n=%-8d  setup %6.2f ms   %.4f ms/apply   (%.2f ns/point)\n",
                    n, t_setup, t, t * 1e6 / n);
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// 3. Throughput across many independent lines (the solver's real workload):
//    a 2D field is a stack of 1D lines, each an independent apply. They share
//    one read-only factorization and parallelize across all cores.
// ---------------------------------------------------------------------------
static void throughput() {
    const double Lp = 2.0 * PI;
    const int p = 4, n = 4096, lines = 3000;     // ~12.3 M points, one D1 each
    const double Mpts = static_cast<double>(n) * lines / 1e6;
    std::printf("\n--- Throughput: BMG D1 over %d lines of length %d (%.1f M points) ---\n",
                lines, n, Mpts);

    FastDeriv f = FastDeriv::build(true, n, p, Lp);
    std::vector<double> u = test_field(n);

    double sink = 0.0;
    auto t0 = Clock::now();
    {
        std::vector<double> a(n), b(n), y(n);
        for (int line = 0; line < lines; ++line) { f.bmg_d1(u, y, a, b); sink += y[0]; }
    }
    double t_serial = ms_since(t0);

    t0 = Clock::now();
    #pragma omp parallel
    {
        std::vector<double> a(n), b(n), y(n);
        double local = 0.0;
        #pragma omp for nowait
        for (int line = 0; line < lines; ++line) { f.bmg_d1(u, y, a, b); local += y[0]; }
        #pragma omp atomic
        sink += local;
    }
    double t_par = ms_since(t0);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("   serial:    %.1f ms   (%.1f ns/point,  %.0f M points/s)\n",
                t_serial, t_serial * 1e6 / (Mpts * 1e6), Mpts / (t_serial / 1e3));
    std::printf("   %2d threads: %.1f ms   (%.1f ns/point,  %.0f M points/s)\n",
                threads, t_par, t_par * 1e6 / (Mpts * 1e6), Mpts / (t_par / 1e3));
    std::printf("   parallel speedup: %.1fx on %d cores\n", t_serial / t_par, threads);
    (void)sink;
}

int main() {
    std::printf("==================================================================\n");
    std::printf(" Fast banded apply: correctness + speed (AVX2/FMA)\n");
    std::printf("==================================================================\n\n");
    check_correctness();
    benchmark();
    throughput();
    std::printf("\nDone.\n");
    return 0;
}
