// fair_compare.cpp
//
// FAIR basis-of-comparison study (resolution of the first derivative): B-spline
// mimetic (BMG_p, formal order 2(p+1)) vs compact Pade of the SAME formal order,
// AND with the recurring cost reported, so the comparison is honest.
//
// Compact coefficients are DERIVED here by Taylor-matching the modified wavenumber
// (no hand-copied tables): a centered compact with LHS half-bandwidth beta and RHS
// half-bandwidth r reaches formal order 2(beta+r). We compare, at matched order:
//   - BMG_p  (order 2(p+1))         cost F = 3*r_rhs + 4*r_solve + 1 = 7p+1
//   - compact tridiagonal (beta=1, r=p)   cost = 3p + 4 + 1
//   - compact pentadiagonal (beta=2, r=p-1) cost = 3(p-1) + 8 + 1
// reporting the resolution error |k_eff/k - 1| at a few points-per-wavelength.
//
// The honest question this answers: at the SAME order (and at what cost), does the
// spline actually resolve shorter waves than the world's best compact?

#include <cstdio>
#include <cmath>
#include <vector>
#include "bm_apply.h"          // FastDeriv (BMG); linalg DenseMatrix, dense_lu_solve

static const double PI = 3.14159265358979323846;
static double fact(int n) { double f = 1; for (int i = 2; i <= n; ++i) f *= i; return f; }

// Centered compact D1: sum_j alpha_j f'_{i+j} = (1/h) sum_k c_k (f_{i+k}-f_{i-k}),
// alpha_0=1, symmetric alpha, antisymmetric RHS. Max order 2(beta+r); solve the
// order conditions for alpha_1..beta, c_1..r.
static void compact_coeffs(int beta, int r, std::vector<double>& al, std::vector<double>& c) {
    const int n = beta + r;
    DenseMatrix A(n); std::vector<double> b(n, 0.0);
    for (int e = 0; e < n; ++e) {                      // order index e (theta^{2e+1})
        for (int j = 1; j <= beta; ++j) A(e, j - 1)      = -2.0 * std::pow((double)j, 2 * e)     / fact(2 * e);
        for (int k = 1; k <= r;    ++k) A(e, beta + k - 1) = 2.0 * std::pow((double)k, 2 * e + 1) / fact(2 * e + 1);
        b[e] = (e == 0) ? 1.0 : 0.0;
    }
    std::vector<double> x; dense_lu_solve(A, b, x);
    al.assign(beta, 0.0); for (int j = 1; j <= beta; ++j) al[j - 1] = x[j - 1];
    c.assign(r, 0.0);     for (int k = 1; k <= r;    ++k) c[k - 1]   = x[beta + k - 1];
}
static double compact_err(int beta, int r, double theta) {
    std::vector<double> al, c; compact_coeffs(beta, r, al, c);
    double Ls = 1.0, Rs = 0.0;
    for (int j = 1; j <= beta; ++j) Ls += 2.0 * al[j - 1] * std::cos(j * theta);
    for (int k = 1; k <= r;    ++k) Rs += 2.0 * c[k - 1]  * std::sin(k * theta);
    return std::fabs((Rs / Ls) / theta - 1.0);
}
// BMG_p first-derivative modified-wavenumber error at mode m on an N grid.
static double bmg_err(int p, int m, int N, double L) {
    FastDeriv f = FastDeriv::build(true, N, p, L);
    std::vector<double> sk(N); const double k = 2 * PI * m / L, h = L / N;
    for (int j = 0; j < N; ++j) sk[j] = std::sin(k * j * h);
    std::vector<double> out = f.bmg_d1(sk);
    return std::fabs(out[0] / k - 1.0);
}

int main() {
    const int N = 120; const double L = 2 * PI;        // N divisible by 4,6,8,10,12,...
    std::printf("=================================================================\n");
    std::printf(" FAIR comparison: B-spline (BMG_p) vs MATCHED-ORDER compact Pade\n");
    std::printf(" 1st-deriv modified-wavenumber error |k_eff/k-1|; cost = 3 r_rhs + 4 r_solve + 1\n");
    std::printf("=================================================================\n");
    // sanity: derived C6 (beta=1,r=2) must reproduce alpha=1/3, c=(7/9,1/36)
    { std::vector<double> al, c; compact_coeffs(1, 2, al, c);
      std::printf("\n [check] derived C6: alpha=%.5f (1/3=%.5f), c1=%.5f (7/9=%.5f), c2=%.5f (1/36=%.5f)\n",
                  al[0], 1.0/3, c[0], 7.0/9, c[1], 1.0/36); }

    const int ms[] = {30, 20, 15, 12};                 // PPW = N/m = 4, 6, 8, 10
    for (int p = 2; p <= 5; ++p) {                      // BMG_p : order 2(p+1) = 6,8,10,12
        const int ord = 2 * (p + 1);
        const int costBMG = 7 * p + 1;
        const int rt = p,  costT = 3 * rt + 4 * 1 + 1;             // tridiag compact, same order
        const int rp = p - 1, bp = 2, costP = 3 * rp + 4 * bp + 1; // pentadiag compact, same order
        std::printf("\n--- formal order %d ---  BMG%d (cost %d) | compact-tridiag (cost %d) | compact-penta (cost %d)\n",
                    ord, p, costBMG, costT, costP);
        std::printf("  PPW    BMG%d        compact-tri   compact-penta\n", p);
        for (int m : ms) {
            const double ppw = (double)N / m, th = 2 * PI * m / N;
            std::printf("  %4.1f   %.3e   %.3e     %.3e\n", ppw,
                        bmg_err(p, m, N, L), compact_err(1, rt, th), compact_err(bp, rp, th));
        }
    }
    // ---- cost-to-target-accuracy (the honest cost-accuracy Pareto) ----
    // asymptotic err ~ err8 * (8/PPW)^order  =>  PPW_need = 8*(err8/eps)^{1/order}.
    // recurring work to a target eps:  1D ~ cost/pt * PPW ;  3D DNS+CFL ~ cost/pt * PPW^4.
    std::printf("\n=================================================================\n");
    std::printf(" COST TO REACH TARGET ACCURACY (matched order; lower = cheaper)\n");
    std::printf(" 3D+time work ~ (cost/pt) * PPW_need^4 ; ratio = best-compact / BMG\n");
    std::printf(" (ratio > 1  =>  BMG cheaper for the same accuracy at the same order)\n");
    std::printf("=================================================================\n");
    auto ppw_need = [](double err8, int order, double eps){ return 8.0 * std::pow(err8/eps, 1.0/order); };
    std::printf("\n  order  eps      BMG PPW(cost/pt)  bestCompact PPW(cost/pt)  work-ratio(compact/BMG)\n");
    for (int p = 2; p <= 5; ++p) {
        const int ord = 2*(p+1), costBMG = 7*p+1;
        const int rt = p, costT = 3*rt+5;  const int rp = p-1, costP = 3*rp+9;
        const double th8 = 2*PI*15/N;                       // PPW=8 reference point
        const double eB = bmg_err(p, 15, N, L);
        const double eT = compact_err(1, rt, th8), eP = compact_err(2, rp, th8);
        const bool pentaBetter = (eP < eT);
        const double eC = pentaBetter ? eP : eT; const int costC = pentaBetter ? costP : costT;
        const char* cname = pentaBetter ? "penta" : "tri ";
        for (double eps : {1e-6, 1e-8, 1e-10}) {
            const double pB = ppw_need(eB, ord, eps), pC = ppw_need(eC, ord, eps);
            const double wB = costBMG*std::pow(pB,4), wC = costC*std::pow(pC,4);
            std::printf("   %2d   %.0e   %5.2f (%2d)        %5.2f (%2d) %s         %.2fx\n",
                        ord, eps, pB, costBMG, pC, costC, cname, wC/wB);
        }
    }
    std::printf("\nDone.\n");
    return 0;
}
