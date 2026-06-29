// bm_operators.h
//
// Assembly of the B-spline mimetic first- and second-derivative operators for
// arbitrary spline degree p, in the four routes of the sibling papers:
//
//   PERIODIC, flux route     BMC : D1 = B' B^{-1},          D2 = B'' B^{-1}
//   PERIODIC, Galerkin route BMG : D1 = B M^{-1} K B^{-1},  D2 = -B M^{-1} S B^{-1}
//   BOUNDED,  flux route     BMF : D1 = G = B_eta D B^{-1}, D2 = Dv G  (staggered)
//   BOUNDED,  Galerkin route BMG : D1 = B M^{-1} K B^{-1},  D2 = H^{-1}(-D^T H D + d D)
//
// See cpp/README.md for the operator definitions (BMF staggered flux complex,
// BMG same-space Galerkin operator, and D = M^{-1}K on periodic grids).
//
// Each operator is materialized as a dense Mat purely for verification and MMS:
// the production solver applies the same maps as a chain of banded solves and
// never forms these matrices. Every builder also returns the machine-precision
// structural identities (Green, SBP) so a wrong assembly can never pass quietly.

#ifndef BM_OPERATORS_H
#define BM_OPERATORS_H

#include <string>
#include <utility>
#include <vector>
#include "bspline.h"

// One assembled scheme: the two derivative operators plus the node sets they
// act on/produce, plus the structural identities verified at build time.
struct OperatorSet {
    std::string name;                 // e.g. "BMG" / "BMF" / "BMC"
    int p = 0;
    bool periodic = false;
    bool staggered = false;           // true for BMF: D1 is (N-1) x N
    std::vector<double> in_nodes;     // sample points of the input field u
    std::vector<double> d1_nodes;     // output points of D1 u
    std::vector<double> d2_nodes;     // output points of D2 u
    Mat D1, D2;
    Mat Dv;                           // BMF staggered divergence (derivative grid -> primal); empty otherwise
    std::vector<std::pair<std::string, double>> identities;   // name -> residual
};

// d = -e_0 e_0^T + e_{N-1} e_{N-1}^T : the boundary trace operator.
inline Mat boundary_trace(int n) {
    Mat P(n, n);
    P(0, 0) = -1.0;
    P(n - 1, n - 1) = 1.0;
    return P;
}

// --- PERIODIC, collocation (BMC): same-point derivative B^{(k)} B^{-1} -------
inline OperatorSet build_bmc_periodic(int n, int p, double L) {
    SplineOperators op = build_periodic(n, p, L);
    Mat Binv = inverse(to_mat(op.B));
    OperatorSet os;
    os.name = "BMC"; os.p = p; os.periodic = true;
    os.in_nodes = os.d1_nodes = os.d2_nodes = op.nodes;
    os.D1 = matmul(to_mat(op.Bp), Binv);
    os.D2 = matmul(to_mat(op.Bpp), Binv);
    // BMC first derivative is skew in the diagonal norm H = h I: D1 + D1^T = 0.
    {
        Mat D1t = transpose(os.D1), s(n, n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) s(i, j) = os.D1(i, j) + D1t(i, j);
        os.identities.push_back({"skew  ||D1 + D1^T||", max_abs(s)});
    }
    return os;
}

// --- PERIODIC, Galerkin (BMG): D = B M^{-1} K B^{-1}, D2 = -B M^{-1} S B^{-1} -
inline OperatorSet build_bmg_periodic(int n, int p, double L) {
    SplineOperators op = build_periodic(n, p, L);
    CoeffForms cf = assemble_forms_periodic(n, p, op.h);
    Mat B = to_mat(op.B), Binv = inverse(B);
    Mat Minv = inverse(cf.M);
    Mat MK = matmul(Minv, cf.K);              // coefficient-space derivative
    Mat MS = matmul(Minv, cf.S);
    OperatorSet os;
    os.name = "BMG"; os.p = p; os.periodic = true;
    os.in_nodes = os.d1_nodes = os.d2_nodes = op.nodes;
    os.D1 = matmul(matmul(B, MK), Binv);
    os.D2 = matmul(matmul(B, MS), Binv);
    for (double& v : os.D2.a) v = -v;         // \int N_i N_j'' = -S in the interior
    // Periodic Green identity: K + K^T = 0.
    {
        Mat Kt = transpose(cf.K), s(n, n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) s(i, j) = cf.K(i, j) + Kt(i, j);
        os.identities.push_back({"green ||K + K^T||", max_abs(s)});
    }
    return os;
}

// --- BOUNDED, Galerkin (BMG): natural spline mass metric --------------------
inline OperatorSet build_bmg_bounded(int npts, int p, double L) {
    SplineOperators op = build_clamped(npts, p, L);
    std::vector<double> knots = open_uniform_knots(npts, p, L);
    CoeffForms cf = assemble_forms_clamped(npts, p, knots);
    Mat B = to_mat(op.B), Binv = inverse(B), BinvT = transpose(Binv);
    Mat Minv = inverse(cf.M);
    Mat MK = matmul(Minv, cf.K);
    Mat D = matmul(matmul(B, MK), Binv);          // D_BMG = B M^{-1} K B^{-1}
    Mat H = matmul(matmul(BinvT, cf.M), Binv);    // H = B^{-T} M B^{-1}
    Mat Hinv = inverse(H);
    Mat P = boundary_trace(npts);
    // D2 = H^{-1} ( -D^T H D + d D )   (the compatible diffusion flux form)
    Mat DtHD = matmul(transpose(D), matmul(H, D));
    Mat PD = matmul(P, D);
    Mat rhs(npts, npts);
    for (int i = 0; i < npts; ++i) for (int j = 0; j < npts; ++j) rhs(i, j) = -DtHD(i, j) + PD(i, j);
    OperatorSet os;
    os.name = "BMG"; os.p = p; os.periodic = false;
    os.in_nodes = os.d1_nodes = os.d2_nodes = op.nodes;
    os.D1 = D;
    os.D2 = matmul(Hinv, rhs);
    // SBP first-derivative identity: H D + D^T H = d.
    {
        Mat HD = matmul(H, D), DtH = matmul(transpose(D), H), s(npts, npts);
        for (int i = 0; i < npts; ++i) for (int j = 0; j < npts; ++j) s(i, j) = HD(i, j) + DtH(i, j) - P(i, j);
        os.identities.push_back({"sbp   ||HD + D^T H - d||", max_abs(s)});
    }
    // Coefficient Green identity: K + K^T = d (in coefficient space).
    {
        Mat Kt = transpose(cf.K), s(npts, npts);
        for (int i = 0; i < npts; ++i) for (int j = 0; j < npts; ++j) s(i, j) = cf.K(i, j) + Kt(i, j) - P(i, j);
        os.identities.push_back({"green ||K + K^T - d||", max_abs(s)});
    }
    return os;
}

// --- BOUNDED, flux (BMF): staggered gradient-divergence complex --------------
inline OperatorSet build_bmf_bounded(int npts, int p, double L) {
    SplineOperators op = build_clamped(npts, p, L);
    std::vector<double> knots = open_uniform_knots(npts, p, L);
    CoeffForms cf = assemble_forms_clamped(npts, p, knots);
    DerivativeSpace ds = build_derivative_space(npts, p, knots);

    Mat B = to_mat(op.B), Binv = inverse(B), BinvT = transpose(Binv);
    Mat Bhat_inv = inverse(ds.Bhat), Bhat_invT = transpose(Bhat_inv);

    // Gradient G = B_eta * Delta * B^{-1} : R^N -> R^{N-1}.
    Mat G = matmul(matmul(ds.Bhat, ds.Delta), Binv);
    // Metrics H0 (primal) and H1 (derivative grid).
    Mat H0 = matmul(matmul(BinvT, cf.M), Binv);
    Mat H1 = matmul(matmul(Bhat_invT, ds.forms.M), Bhat_inv);
    Mat H0inv = inverse(H0);

    // Endpoint reader rows r_0^T, r_1^T of the derivative-space spline: evaluate
    // the degree-(p-1) basis at the two ends, times B_eta^{-1}.
    const int m = ds.m;
    std::vector<std::vector<double>> dbuf;
    std::vector<double> e0(m, 0.0), e1(m, 0.0);
    {
        int s0 = find_span(m - 1, ds.p1, 0.0, ds.knots);
        ders_basis_funs(s0, 0.0, ds.p1, 0, ds.knots, dbuf);
        for (int j = 0; j <= ds.p1; ++j) e0[s0 - ds.p1 + j] = dbuf[0][j];
        int s1 = find_span(m - 1, ds.p1, L, ds.knots);
        ders_basis_funs(s1, L, ds.p1, 0, ds.knots, dbuf);
        for (int j = 0; j <= ds.p1; ++j) e1[s1 - ds.p1 + j] = dbuf[0][j];
    }
    // r_w^T = [Bhat_j(x_w)] * Bhat^{-1}  (a length-m row vector).
    std::vector<double> r0(m, 0.0), r1(m, 0.0);
    for (int j = 0; j < m; ++j) {
        double a0 = 0.0, a1 = 0.0;
        for (int kk = 0; kk < m; ++kk) { a0 += e0[kk] * Bhat_inv(kk, j); a1 += e1[kk] * Bhat_inv(kk, j); }
        r0[j] = a0; r1[j] = a1;
    }
    // Staggered endpoint matrix Bd = -e_0 r_0^T + e_{N-1} r_1^T  (N x (N-1)).
    Mat Bd(npts, m);
    for (int j = 0; j < m; ++j) { Bd(0, j) = -r0[j]; Bd(npts - 1, j) = r1[j]; }
    // Divergence Dv = H0^{-1} ( Bd - G^T H1 )  : R^{N-1} -> R^N.
    Mat GtH1 = matmul(transpose(G), H1);
    Mat inner(npts, m);
    for (int i = 0; i < npts; ++i) for (int j = 0; j < m; ++j) inner(i, j) = Bd(i, j) - GtH1(i, j);
    Mat Dv = matmul(H0inv, inner);

    OperatorSet os;
    os.name = "BMF"; os.p = p; os.periodic = false; os.staggered = true;
    os.in_nodes = op.nodes;
    os.d1_nodes = ds.eta;             // gradient lives on the derivative grid
    os.d2_nodes = op.nodes;           // diffusion returns to the primal grid
    os.D1 = G;
    os.Dv = Dv;                       // staggered divergence (derivative grid -> primal)
    os.D2 = matmul(Dv, G);            // compatible diffusion D2 = Dv G
    // Staggered SBP identity: H0 Dv + G^T H1 = Bd.
    {
        Mat H0Dv = matmul(H0, Dv), s(npts, m);
        for (int i = 0; i < npts; ++i) for (int j = 0; j < m; ++j) s(i, j) = H0Dv(i, j) + GtH1(i, j) - Bd(i, j);
        os.identities.push_back({"sbp   ||H0 Dv + G^T H1 - Bd||", max_abs(s)});
    }
    return os;
}

#endif // BM_OPERATORS_H
