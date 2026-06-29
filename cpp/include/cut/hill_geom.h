// hill_geom.h
//
// The canonical periodic-hill wall y = h(x) as a sharp embedded curve, the
// geometry for the S6 capstone (sec:s6-hill).  Port of code/periodic_hill_
// geometry.py: a six-segment piecewise cubic (standard benchmark definition), repaired
// to be EXACTLY C2 across all joints (the published 4-significant-figure cubics
// jump ~4e-4, which breaks the divergence/Gauss identity to roundoff; the C2
// repair restores it while staying within the table's own precision).
//
// Units: lengths in hill heights H = 1.  Crest-to-crest spacing Lx = 9, channel
// height Ly = 3.036.  The wall is symmetric about x = Lx/2: the windward profile
// (argument xhat = 28*x in [0,54], beyond which h = 0 in the flat trough) is
// mirrored on the leeward side.  Fluid region = { y >= h(x) }.

#ifndef HILL_GEOM_H
#define HILL_GEOM_H

#include <algorithm>
#include <cmath>
#include <vector>

struct HillProfile {
    double Lx = 9.0;        // crest-to-crest spacing (H units)
    double Ly = 3.036;      // channel height (H units)
    // Segment breaks in xhat = 28*xm where xm = distance from the nearest crest.
    static constexpr double BR[7] = {0.0, 9.0, 14.0, 20.0, 30.0, 40.0, 54.0};
    // C2-repaired LOCAL cubic coefficients [a,b,c,d]: y/H = a + b u + c u^2 + d u^3,
    // u = xhat - BR[seg].  Extracted verbatim from the Python C2 repair.
    static constexpr double C[6][4] = {
        { 1.0000000000000004,  1.5020812190951664e-15,  2.3904900851070415e-04, -7.557135858141106e-05},
        { 0.9642714492836167, -1.4060957982077751e-02, -1.8013776731852192e-03,  6.642249618292267e-05},
        { 0.8572350295664743, -2.7093047500234838e-02, -8.0504023044439350e-04,  5.910719772570910e-05},
        { 0.6784624509778259, -3.0369952911172730e-02,  2.5888932860460340e-04, -7.663273788288154e-06},
        { 0.3929885809383813, -2.7491148475546758e-02,  2.8991114957172600e-05,  2.2274667324788843e-05},
        { 0.1432508750033985, -2.0228925978970975e-02,  6.9723113469980320e-04,  1.201452733618056e-06},
    };

    // Wall height h(x) on the periodic cell [0, Lx].
    double h(double x) const {
        const double xm = std::min(x, Lx - x);       // distance from nearest crest
        const double xhat = 28.0 * xm;
        if (xhat >= BR[6] - 1e-12) return 0.0;        // flat trough
        for (int s = 0; s < 6; ++s) {
            if (xhat <= BR[s + 1] + 1e-9) {
                const double u = xhat - BR[s];
                return C[s][0] + u*(C[s][1] + u*(C[s][2] + u*C[s][3]));
            }
        }
        return 0.0;
    }

    // Wall slope h'(x).  dxhat/dxm = 28; dxm/dx = +1 (windward) or -1 (leeward).
    double hp(double x) const {
        const double xm = std::min(x, Lx - x);
        const double xhat = 28.0 * xm;
        if (xhat >= BR[6] - 1e-12) return 0.0;
        const double sgn = (x < 0.5 * Lx) ? 1.0 : -1.0;
        for (int s = 0; s < 6; ++s) {
            if (xhat <= BR[s + 1] + 1e-9) {
                const double u = xhat - BR[s];
                const double dy = C[s][1] + u*(2.0*C[s][2] + u*3.0*C[s][3]);
                return 28.0 * dy * sgn;
            }
        }
        return 0.0;
    }

    // Streamwise locations (in x) where h has a kink: the nearest-crest fold at
    // x = Lx/2, plus the six segment breaks mapped to x on both sides.  The
    // volume / wall quadrature must split panels here for clean convergence.
    std::vector<double> kinks() const {
        std::vector<double> ks{0.5 * Lx};
        for (int s = 1; s < 7; ++s) {
            const double xm = BR[s] / 28.0;
            if (xm < 0.5 * Lx) { ks.push_back(xm); ks.push_back(Lx - xm); }
        }
        std::sort(ks.begin(), ks.end());
        return ks;
    }
};

#endif // HILL_GEOM_H
