// operator_probe.cpp
//
// Small configurable front door for building one dense B-spline mimetic operator.
// This is for inspection and learning: it materializes D1/D2 and prints shape,
// formal order, identity residuals, and a small matrix block. Production applies
// these maps through the banded/circulant chains in bm_apply.h and bm3d.h.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "bm_operators.h"

static const double PI = 3.14159265358979323846;

enum Family { FAMILY_BMC, FAMILY_BMG, FAMILY_BMF };

struct Options {
    bool periodic = true;
    Family family = FAMILY_BMG;
    int p = 4;
    int n = 32;
    int deriv = 1;
    double L = -1.0;
};

static const char* family_name(Family f) {
    if (f == FAMILY_BMC) return "BMC";
    if (f == FAMILY_BMG) return "BMG";
    return "BMF";
}

static void usage(const char* exe) {
    std::printf("usage: %s --domain periodic|bounded --family bmc|bmg|bmf --p degree --n points --deriv 1|2 [--L length]\n", exe);
    std::printf("\n");
    std::printf("supported combinations:\n");
    std::printf("  periodic: BMC, BMG\n");
    std::printf("  bounded:  BMF, BMG\n");
    std::printf("\n");
    std::printf("examples:\n");
    std::printf("  %s --domain periodic --family bmg --p 6 --n 64 --deriv 1\n", exe);
    std::printf("  %s --domain bounded  --family bmf --p 5 --n 64 --deriv 2\n", exe);
}

static int parse(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            usage(argv[0]);
            return 1;
        } else if (std::strcmp(a, "--domain") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after --domain\n"); return -1; }
            if (std::strcmp(argv[i], "periodic") == 0) opt.periodic = true;
            else if (std::strcmp(argv[i], "bounded") == 0 || std::strcmp(argv[i], "boundary") == 0) opt.periodic = false;
            else { std::fprintf(stderr, "unknown domain: %s\n", argv[i]); return -1; }
        } else if (std::strcmp(a, "--family") == 0 || std::strcmp(a, "--op") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after %s\n", a); return -1; }
            if (std::strcmp(argv[i], "bmc") == 0 || std::strcmp(argv[i], "BMC") == 0) opt.family = FAMILY_BMC;
            else if (std::strcmp(argv[i], "bmg") == 0 || std::strcmp(argv[i], "BMG") == 0) opt.family = FAMILY_BMG;
            else if (std::strcmp(argv[i], "bmf") == 0 || std::strcmp(argv[i], "BMF") == 0) opt.family = FAMILY_BMF;
            else { std::fprintf(stderr, "unknown family: %s\n", argv[i]); return -1; }
        } else if (std::strcmp(a, "--p") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after --p\n"); return -1; }
            opt.p = std::atoi(argv[i]);
        } else if (std::strcmp(a, "--n") == 0 || std::strcmp(a, "--N") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after %s\n", a); return -1; }
            opt.n = std::atoi(argv[i]);
        } else if (std::strcmp(a, "--deriv") == 0 || std::strcmp(a, "--d") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after %s\n", a); return -1; }
            opt.deriv = std::atoi(argv[i]);
        } else if (std::strcmp(a, "--L") == 0) {
            if (++i >= argc) { std::fprintf(stderr, "missing value after --L\n"); return -1; }
            opt.L = std::atof(argv[i]);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a);
            return -1;
        }
    }
    if (opt.p < 2) { std::fprintf(stderr, "--p must be at least 2 for these operators\n"); return -1; }
    if (opt.n < opt.p + 2) { std::fprintf(stderr, "--n must be at least p+2\n"); return -1; }
    if (opt.deriv != 1 && opt.deriv != 2) { std::fprintf(stderr, "--deriv must be 1 or 2\n"); return -1; }
    if (opt.periodic && opt.n <= 2 * (opt.p + 1)) {
        std::fprintf(stderr, "periodic --n must be greater than 2*(p+1) for unaliased narrow stencils\n");
        return -1;
    }
    return 0;
}

static int formal_order(const Options& opt) {
    if (opt.periodic) {
        if (opt.family == FAMILY_BMC)
            return (opt.deriv == 1) ? 2 * ((opt.p + 1) / 2) : 2 * (opt.p / 2);
        return (opt.deriv == 1) ? 2 * (opt.p + 1) : 2 * opt.p;
    }
    return (opt.deriv == 1) ? opt.p : opt.p - 1;
}

static void print_block(const Mat& A) {
    const int rr = A.r < 6 ? A.r : 6;
    const int cc = A.c < 6 ? A.c : 6;
    std::printf("\nTop-left %dx%d block:\n", rr, cc);
    for (int i = 0; i < rr; ++i) {
        std::printf("  ");
        for (int j = 0; j < cc; ++j) std::printf("% .4e ", A(i, j));
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    Options opt;
    const int parsed = parse(argc, argv, opt);
    if (parsed > 0) return 0;
    if (parsed < 0) { usage(argv[0]); return 1; }

    if (opt.L <= 0.0) opt.L = opt.periodic ? 2.0 * PI : 2.0;

    if (opt.periodic && opt.family == FAMILY_BMF) {
        std::fprintf(stderr, "BMF is the bounded staggered flux route; use periodic BMC or BMG.\n");
        return 1;
    }
    if (!opt.periodic && opt.family == FAMILY_BMC) {
        std::fprintf(stderr, "bounded flux route is BMF, not BMC; use bounded BMF or BMG.\n");
        return 1;
    }

    OperatorSet os;
    if (opt.periodic && opt.family == FAMILY_BMC) os = build_bmc_periodic(opt.n, opt.p, opt.L);
    else if (opt.periodic && opt.family == FAMILY_BMG) os = build_bmg_periodic(opt.n, opt.p, opt.L);
    else if (!opt.periodic && opt.family == FAMILY_BMF) os = build_bmf_bounded(opt.n, opt.p, opt.L);
    else os = build_bmg_bounded(opt.n, opt.p, opt.L);

    const Mat& D = (opt.deriv == 1) ? os.D1 : os.D2;
    std::printf("Built %s %s D%d\n", opt.periodic ? "periodic" : "bounded",
                family_name(opt.family), opt.deriv);
    std::printf("  p=%d  n=%d  L=%.10g  formal order=%d\n", opt.p, opt.n, opt.L, formal_order(opt));
    std::printf("  operator shape: %d x %d\n", D.r, D.c);
    if (os.staggered && opt.deriv == 1)
        std::printf("  note: BMF D1 is staggered, mapping primal nodes to the derivative grid.\n");
    for (auto& id : os.identities)
        std::printf("  identity: %s = %.3e\n", id.first.c_str(), id.second);
    print_block(D);
    return 0;
}
