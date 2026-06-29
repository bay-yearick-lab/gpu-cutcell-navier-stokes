// bspline_basis.cpp
//
// Building block #1 of the operator assembly: evaluate the p+1 nonzero
// uniform (cardinal) B-spline basis functions at a point.
//
// This is the Cox-de Boor recurrence, but written as an ITERATIVE triangular
// fill -- no recursion, no call stack. It is specialized to a uniform knot
// grid (knot spacing = 1), which makes the denominator a small integer j and
// removes all knot lookups, so the inner loop is branch-free and SIMD-friendly.
//
// Convention: we work in a LOCAL coordinate s in [0, 1] inside one knot span.
// The function returns the p+1 basis functions that are nonzero on that span,
// in order N[0..p]. With the uniform/centered convention, evaluating at the
// span values that correspond to grid points reproduces the classic cardinal
// samples, e.g. for p=2 at s=0.5 you get [1/8, 3/4, 1/8].

#include <cstdio>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Evaluate the p+1 nonzero uniform B-spline basis functions at local s in [0,1].
//
//   p   : spline degree (1=linear, 2=quadratic, 3=cubic, ...)
//   s   : local coordinate within the span, 0 <= s <= 1
//   N   : output array of length p+1, filled with the basis values
//
// Cost: O(p^2). The only division is by the integer j (1..p); we could hoist
// reciprocals into a small precomputed table, but at setup that is not worth
// the clutter yet.
// ---------------------------------------------------------------------------
void cardinal_bspline_basis(int p, double s, double* N)
{
    N[0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        const double inv_j = 1.0 / static_cast<double>(j);
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            const double temp   = N[r] * inv_j;
            const double rightw = static_cast<double>(r + 1) - s;        // right[r+1]
            const double leftw  = s + static_cast<double>(j - r) - 1.0;  // left[j-r]
            N[r]  = saved + rightw * temp;
            saved = leftw * temp;
        }
        N[j] = saved;
    }
}

// ---------------------------------------------------------------------------
// Tiny test driver: validate against exact known values + partition of unity.
//
// Partition of unity (the p+1 basis values sum to exactly 1 for any s) is a
// convention-independent correctness check: if the algorithm is right, it
// holds to machine precision for every degree and every point.
// ---------------------------------------------------------------------------
static bool close(double a, double b, double tol = 1e-13)
{
    return std::fabs(a - b) <= tol;
}

int main()
{
    bool all_ok = true;

    // --- Exact-value checks ------------------------------------------------
    // p=1 (linear) at s=0.3 -> [1-s, s] = [0.7, 0.3]
    {
        double N[2];
        cardinal_bspline_basis(1, 0.3, N);
        const bool ok = close(N[0], 0.7) && close(N[1], 0.3);
        printf("p=1 s=0.30  N = [%.6f %.6f]                 %s\n",
               N[0], N[1], ok ? "OK" : "FAIL");
        all_ok &= ok;
    }
    // p=2 (quadratic) at s=0.5 -> [1/8, 3/4, 1/8]
    {
        double N[3];
        cardinal_bspline_basis(2, 0.5, N);
        const bool ok = close(N[0], 0.125) && close(N[1], 0.75) && close(N[2], 0.125);
        printf("p=2 s=0.50  N = [%.6f %.6f %.6f]        %s\n",
               N[0], N[1], N[2], ok ? "OK" : "FAIL");
        all_ok &= ok;
    }
    // p=3 (cubic) at s=0.0 -> [1/6, 2/3, 1/6, 0]
    {
        double N[4];
        cardinal_bspline_basis(3, 0.0, N);
        const bool ok = close(N[0], 1.0 / 6.0) && close(N[1], 2.0 / 3.0)
                     && close(N[2], 1.0 / 6.0) && close(N[3], 0.0);
        printf("p=3 s=0.00  N = [%.6f %.6f %.6f %.6f] %s\n",
               N[0], N[1], N[2], N[3], ok ? "OK" : "FAIL");
        all_ok &= ok;
    }

    // --- Partition of unity over a sweep of points and degrees -------------
    double max_pou_err = 0.0;
    std::vector<double> N(16);
    for (int p = 1; p <= 8; ++p) {
        for (int k = 0; k <= 10; ++k) {
            const double s = k / 10.0;
            cardinal_bspline_basis(p, s, N.data());
            double sum = 0.0;
            for (int i = 0; i <= p; ++i) sum += N[i];
            max_pou_err = std::fmax(max_pou_err, std::fabs(sum - 1.0));
        }
    }
    const bool pou_ok = max_pou_err < 1e-14;
    printf("\npartition-of-unity max error (p=1..8) = %.3e   %s\n",
           max_pou_err, pou_ok ? "OK" : "FAIL");
    all_ok &= pou_ok;

    printf("\n%s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_ok ? 0 : 1;
}
