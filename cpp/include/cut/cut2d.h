// cut2d.h
//
// Two-dimensional sharp-cut B-spline kernel for an oblique (half-plane) cut: the
// C++ port of code/oblique_cut_sbp.py.  Each background cell is clipped against
// the half-plane {n . x >= offset} (Sutherland-Hodgman), the clipped polygon is
// fan-triangulated, and product Gauss-Duffy quadrature accumulates the cut forms
//
//   H   = \int_fluid  phi_i phi_j,           K  = \int_fluid grad phi_i . grad phi_j,
//   Qx  = \int_fluid  phi_i dphi_j/dx,        Qy = \int_fluid phi_i dphi_j/dy,
//   Sx  = \oint_dOmega phi_i phi_j n_x,        Sy = \oint_dOmega phi_i phi_j n_y,
//
// with the tensor index convention  row = iy*nx + ix  (y-major, x-fast).  The
// cut-domain Green identities  Qx + Qx^T = Sx  and  Qy + Qy^T = Sy  hold to
// roundoff once the polygon quadrature resolves the cut.  The polynomial-
// preserving extension E reconstructs every sliver mode from nearby interior
// modes while reproducing all tensor monomials x^r y^s, r+s <= p.

#ifndef CUT2D_H
#define CUT2D_H

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "bspline.h"
#include "cut1d.h"

using P2 = std::array<double, 2>;

struct HalfPlaneCut {              // fluid region: n . x >= offset
    double nx_ = 1.0, ny_ = 0.0, offset = 0.0;
    double f(const P2& p) const { return nx_ * p[0] + ny_ * p[1] - offset; }
    bool inside(const P2& p, double tol = 1e-14) const { return f(p) >= -tol; }
};

inline P2 seg_intersect(const P2& a, const P2& b, const HalfPlaneCut& c) {
    const double fa = c.f(a), fb = c.f(b), den = fa - fb;
    if (std::fabs(den) < 1e-30) return {0.5*(a[0]+b[0]), 0.5*(a[1]+b[1])};
    double t = fa / den; t = std::min(1.0, std::max(0.0, t));
    return {a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])};
}

// Sutherland-Hodgman clip of a CCW polygon against the half-plane.
inline std::vector<P2> clip_halfplane(const std::vector<P2>& poly, const HalfPlaneCut& c) {
    if (poly.empty()) return {};
    std::vector<P2> out;
    P2 prev = poly.back(); bool prev_in = c.inside(prev);
    for (const P2& cur : poly) {
        const bool cur_in = c.inside(cur);
        if (cur_in) { if (!prev_in) out.push_back(seg_intersect(prev, cur, c)); out.push_back(cur); }
        else if (prev_in) out.push_back(seg_intersect(prev, cur, c));
        prev = cur; prev_in = cur_in;
    }
    if (out.empty()) return {};
    std::vector<P2> cl;
    for (const P2& q : out)
        if (cl.empty() || std::hypot(q[0]-cl.back()[0], q[1]-cl.back()[1]) > 1e-13) cl.push_back(q);
    if (cl.size() > 1 && std::hypot(cl.front()[0]-cl.back()[0], cl.front()[1]-cl.back()[1]) <= 1e-13)
        cl.pop_back();
    return cl;
}

inline double polygon_area(const std::vector<P2>& p) {
    if (p.size() < 3) return 0.0;
    double a = 0.0; const int n = (int)p.size();
    for (int i = 0; i < n; ++i) { const P2& u = p[i]; const P2& v = p[(i+1)%n]; a += u[0]*v[1] - u[1]*v[0]; }
    return 0.5 * a;
}

// All cut forms for a 2D oblique cut, plus tensor bookkeeping.
struct CutForms2D {
    int nx = 0, ny = 0, n = 0, degree = 0, n_elements = 0;
    Mat H, Qx, Qy, K, Sx, Sy, Hbg;
    std::vector<double> gx, gy;          // tensor Greville coords, length n (row=iy*nx+ix)
    std::vector<double> knots;           // shared x=y open-uniform knot vector
};

// Evaluate the p+1 local 1D basis values (nu=0) and derivatives (nu=1) at x on
// the span containing x; returns the starting global index lo (funcs lo..lo+p).
inline int local_1d(const std::vector<double>& knots, int p, int nb, double x,
                    std::vector<double>& v, std::vector<double>& dv) {
    const double xc = std::min(std::max(x, knots[p]), knots[nb]);
    const int span = find_span(nb - 1, p, xc, knots);
    std::vector<std::vector<double>> d;
    ders_basis_funs(span, xc, p, 1, knots, d);
    v.assign(p+1, 0.0); dv.assign(p+1, 0.0);
    for (int j = 0; j <= p; ++j) { v[j] = d[0][j]; dv[j] = d[1][j]; }
    return span - p;
}

inline CutForms2D assemble_oblique_cut_2d(int n_elements, int p, const HalfPlaneCut& cut,
                                          int q = -1) {
    if (q < 0) q = std::max(10, p + 5);
    const int nb = n_elements + p;          // 1D basis count per axis
    std::vector<double> knots = open_uniform_knots(nb, p, 1.0);
    std::vector<double> grev  = greville_points(knots, p, nb);
    const int n = nb * nb;
    const double h = 1.0 / n_elements;

    CutForms2D cf;
    cf.nx = cf.ny = nb; cf.n = n; cf.degree = p; cf.n_elements = n_elements; cf.knots = knots;
    cf.H = Mat(n,n); cf.Qx = Mat(n,n); cf.Qy = Mat(n,n); cf.K = Mat(n,n);
    cf.Sx = Mat(n,n); cf.Sy = Mat(n,n); cf.Hbg = Mat(n,n);
    cf.gx.resize(n); cf.gy.resize(n);
    for (int iy = 0; iy < nb; ++iy) for (int ix = 0; ix < nb; ++ix) {
        cf.gx[iy*nb+ix] = grev[ix]; cf.gy[iy*nb+ix] = grev[iy];
    }

    std::vector<double> gleg, wleg; gauss_legendre(q, gleg, wleg);
    // Map Gauss [-1,1] -> [0,1] for the Duffy/edge parameter.
    std::vector<double> u01(q), w01(q);
    for (int i = 0; i < q; ++i) { u01[i] = 0.5*(gleg[i]+1.0); w01[i] = 0.5*wleg[i]; }

    auto accumulate_volume = [&](double xq, double yq, double wq) {
        std::vector<double> vx, dvx, vy, dvy;
        const int lox = local_1d(knots, p, nb, xq, vx, dvx);
        const int loy = local_1d(knots, p, nb, yq, vy, dvy);
        for (int ja = 0; ja <= p; ++ja) for (int ia = 0; ia <= p; ++ia) {
            const int ra = (loy+ja)*nb + (lox+ia);
            const double phi_a = vy[ja]*vx[ia];
            const double dphix_a = vy[ja]*dvx[ia];
            const double dphiy_a = dvy[ja]*vx[ia];
            for (int jb = 0; jb <= p; ++jb) for (int ib = 0; ib <= p; ++ib) {
                const int rb = (loy+jb)*nb + (lox+ib);
                const double phi_b = vy[jb]*vx[ib];
                const double dphix_b = vy[jb]*dvx[ib];
                const double dphiy_b = dvy[jb]*vx[ib];
                cf.H (ra,rb) += wq * phi_a * phi_b;
                cf.Qx(ra,rb) += wq * phi_a * dphix_b;
                cf.Qy(ra,rb) += wq * phi_a * dphiy_b;
                cf.K (ra,rb) += wq * (dphix_a*dphix_b + dphiy_a*dphiy_b);
            }
        }
    };

    auto accumulate_edge = [&](const P2& a, const P2& b) {
        const double len = std::hypot(b[0]-a[0], b[1]-a[1]);
        if (len <= 1e-15) return;
        const double nrmx = (b[1]-a[1])/len, nrmy = -(b[0]-a[0])/len;  // outward for CCW fluid
        for (int g = 0; g < q; ++g) {
            const double t = u01[g], we = w01[g]*len;
            const double xq = a[0] + t*(b[0]-a[0]), yq = a[1] + t*(b[1]-a[1]);
            std::vector<double> vx, dvx, vy, dvy;
            const int lox = local_1d(knots, p, nb, xq, vx, dvx);
            const int loy = local_1d(knots, p, nb, yq, vy, dvy);
            for (int ja = 0; ja <= p; ++ja) for (int ia = 0; ia <= p; ++ia) {
                const int ra = (loy+ja)*nb + (lox+ia);
                const double phi_a = vy[ja]*vx[ia];
                for (int jb = 0; jb <= p; ++jb) for (int ib = 0; ib <= p; ++ib) {
                    const int rb = (loy+jb)*nb + (lox+ib);
                    const double bm = we * phi_a * (vy[jb]*vx[ib]);
                    cf.Sx(ra,rb) += nrmx * bm;
                    cf.Sy(ra,rb) += nrmy * bm;
                }
            }
        }
    };

    auto is_phys_edge = [&](const P2& a, const P2& b) {
        const P2 m{0.5*(a[0]+b[0]), 0.5*(a[1]+b[1])};
        const bool on_cut = std::fabs(cut.f(m)) <= 1e-10;
        const bool on_sq = std::fabs(m[0]) <= 1e-12 || std::fabs(m[0]-1.0) <= 1e-12
                        || std::fabs(m[1]) <= 1e-12 || std::fabs(m[1]-1.0) <= 1e-12;
        return on_cut || on_sq;
    };

    for (int iy = 0; iy < n_elements; ++iy) for (int ix = 0; ix < n_elements; ++ix) {
        std::vector<P2> cell{{ix*h, iy*h}, {(ix+1)*h, iy*h}, {(ix+1)*h, (iy+1)*h}, {ix*h, (iy+1)*h}};
        std::vector<P2> cl = clip_halfplane(cell, cut);
        if (cl.size() < 3 || std::fabs(polygon_area(cl)) <= 1e-16) continue;
        if (polygon_area(cl) < 0.0) std::reverse(cl.begin(), cl.end());
        // Fan-triangulate from vertex 0; product Gauss-Duffy per triangle.
        const P2 v0 = cl[0];
        for (size_t k = 1; k + 1 < cl.size(); ++k) {
            const P2 e1{cl[k][0]-v0[0], cl[k][1]-v0[1]};
            const P2 e2{cl[k+1][0]-v0[0], cl[k+1][1]-v0[1]};
            const double det = std::fabs(e1[0]*e2[1] - e1[1]*e2[0]);
            for (int gu = 0; gu < q; ++gu) for (int gv = 0; gv < q; ++gv) {
                const double uu = u01[gu], vv = u01[gv], omu = 1.0 - uu;
                const double xq = v0[0] + uu*e1[0] + omu*vv*e2[0];
                const double yq = v0[1] + uu*e1[1] + omu*vv*e2[1];
                const double wq = det * omu * w01[gu] * w01[gv];
                accumulate_volume(xq, yq, wq);
            }
        }
        // Boundary forms along physical edges (cut or square boundary).
        for (size_t k = 0; k < cl.size(); ++k)
            if (is_phys_edge(cl[k], cl[(k+1)%cl.size()]))
                accumulate_edge(cl[k], cl[(k+1)%cl.size()]);
    }

    // Background (uncut) tensor mass H_bg = kron(myH, mxH); here mx==my.
    CutForms1D m1 = assemble_cut_1d(n_elements, p, 0.0, 1.0);
    for (int iy = 0; iy < nb; ++iy) for (int ix = 0; ix < nb; ++ix) {
        const int ra = iy*nb + ix;
        for (int jy = 0; jy < nb; ++jy) for (int jx = 0; jx < nb; ++jx) {
            const int rb = jy*nb + jx;
            cf.Hbg(ra,rb) = m1.H(iy,jy) * m1.H(ix,jx);
        }
    }
    return cf;
}

// ---- active sets, tensor monomials, extension, pullback ----

struct ActiveSets2D { std::vector<double> gamma; std::vector<int> active, interior, small; };

inline ActiveSets2D cut2d_active_sets(const CutForms2D& cf,
                                      double support_threshold = 0.05, double active_tol = 1e-22) {
    ActiveSets2D a; a.gamma.assign(cf.n, 0.0);
    for (int i = 0; i < cf.n; ++i) { const double bg = cf.Hbg(i,i); a.gamma[i] = (bg>0.0)? cf.H(i,i)/bg : 0.0; }
    for (int i = 0; i < cf.n; ++i) {
        if (a.gamma[i] > active_tol) a.active.push_back(i);
        if (a.gamma[i] >= support_threshold) a.interior.push_back(i);
    }
    for (int i : a.active) if (a.gamma[i] < support_threshold) a.small.push_back(i);
    return a;
}

inline int max_total_degree_by_count(int n_interior) {
    int m = 0; while ((m+1)*(m+2)/2 <= n_interior) ++m; return m - 1;
}

// Tensor monomial coefficient rows: row for x^sx y^sy = kron(cy[sy], cx[sx]),
// ordered by total degree then sx.  Returns (n_terms x n) matrix.
inline Mat cut2d_tensor_poly(const CutForms2D& cf, int max_total) {
    const int nb = cf.nx, p = cf.degree, n = cf.n;
    CutForms1D m1; m1.n = nb; m1.degree = p; m1.knots = cf.knots;
    m1.greville.assign(cf.knots.size(), 0.0);
    { std::vector<double> g = greville_points(cf.knots, p, nb); m1.greville = g; }
    Mat c1 = cut1d_poly_coeffs(m1, max_total);     // (max_total+1 x nb): rows = 1D monomial coeffs
    int n_terms = 0; for (int tot = 0; tot <= max_total; ++tot) n_terms += tot+1;
    Mat poly(n_terms, n);
    int row = 0;
    for (int tot = 0; tot <= max_total; ++tot)
        for (int sx = 0; sx <= tot; ++sx) {
            const int sy = tot - sx;
            for (int iy = 0; iy < nb; ++iy) for (int ix = 0; ix < nb; ++ix)
                poly(row, iy*nb+ix) = c1(sy, iy) * c1(sx, ix);
            ++row;
        }
    return poly;
}

struct Extension2D { Mat E; std::vector<int> interior; double poly_residual = 0.0;
                     int degree = 0; double max_weight = 0.0; };

// Thin Householder QR of a tall matrix A (rows x cols, rows >= cols): A = Q R with
// Q (rows x cols) orthonormal columns and R (cols x cols) upper triangular.
inline void householder_qr(const Mat& A, Mat& Q, Mat& R) {
    const int m = A.r, k = A.c;
    Mat Rm = A;                       // working copy, will hold R in its top kxk
    Q = identity_mat(m); Q.c = k;     // we only need the first k columns of Q
    std::vector<std::vector<double>> Qfull(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) Qfull[i][i] = 1.0;
    for (int j = 0; j < k; ++j) {
        double norm = 0.0; for (int i = j; i < m; ++i) norm += Rm(i,j)*Rm(i,j);
        norm = std::sqrt(norm);
        if (norm < 1e-300) continue;
        const double alpha = (Rm(j,j) > 0 ? -norm : norm);
        std::vector<double> v(m, 0.0);
        v[j] = Rm(j,j) - alpha;
        for (int i = j+1; i < m; ++i) v[i] = Rm(i,j);
        double vn2 = 0.0; for (int i = j; i < m; ++i) vn2 += v[i]*v[i];
        if (vn2 < 1e-300) continue;
        // Rm <- (I - 2 v v^T / vn2) Rm  (columns j..k-1)
        for (int c = j; c < k; ++c) {
            double dot = 0.0; for (int i = j; i < m; ++i) dot += v[i]*Rm(i,c);
            const double f = 2.0*dot/vn2;
            for (int i = j; i < m; ++i) Rm(i,c) -= f*v[i];
        }
        // Qfull <- Qfull (I - 2 v v^T / vn2)
        for (int i = 0; i < m; ++i) {
            double dot = 0.0; for (int r2 = j; r2 < m; ++r2) dot += Qfull[i][r2]*v[r2];
            const double f = 2.0*dot/vn2;
            for (int r2 = j; r2 < m; ++r2) Qfull[i][r2] -= f*v[r2];
        }
    }
    Q = Mat(m, k); R = Mat(k, k);
    for (int i = 0; i < m; ++i) for (int j = 0; j < k; ++j) Q(i,j) = Qfull[i][j];
    for (int i = 0; i < k; ++i) for (int j = i; j < k; ++j) R(i,j) = Rm(i,j);
}

// Stable minimum-norm solution of an underdetermined C w = t (C is m x k, m <= k,
// full row rank): with A = C^T = Q R, solve R^T z = t then w = Q z.  ||w|| minimal.
inline std::vector<double> minnorm_solve(const Mat& C, const std::vector<double>& t) {
    const int m = C.r, k = C.c;
    Mat A(k, m); for (int i = 0; i < m; ++i) for (int j = 0; j < k; ++j) A(j,i) = C(i,j);  // A = C^T
    Mat Q, R; householder_qr(A, Q, R);    // Q (k x m), R (m x m)
    std::vector<double> z(m, 0.0);        // forward solve R^T z = t
    for (int i = 0; i < m; ++i) {
        double s = t[i];
        for (int j = 0; j < i; ++j) s -= R(j,i)*z[j];
        z[i] = (std::fabs(R(i,i)) > 1e-300) ? s / R(i,i) : 0.0;
    }
    return matvec(Q, z);                  // w = Q z (length k)
}

// The polynomial-preserving extension E, following local_aggregation.py: each
// sliver mode is reconstructed from interior modes in a growing Chebyshev index
// neighborhood, with the monomial block LOCALLY SCALED (centered at the sliver
// Greville point, scaled by h) for conditioning, then a stable min-norm solve.
inline Extension2D cut2d_extension(const CutForms2D& cf, const ActiveSets2D& as,
                                   int preserve_degree = -1) {
    const int n = cf.n, nb = cf.nx, n_int = (int)as.interior.size();
    const double h = 1.0 / cf.n_elements;
    int req = (preserve_degree < 0) ? cf.degree : preserve_degree;
    int degree = std::min(req, max_total_degree_by_count(n_int));
    Mat poly = cut2d_tensor_poly(cf, degree);
    const int n_terms = poly.r;

    // Monomial powers, in the same (total-degree, then sx) order as cut2d_tensor_poly.
    std::vector<std::array<int,2>> powers;
    for (int tot = 0; tot <= degree; ++tot) for (int sx = 0; sx <= tot; ++sx)
        powers.push_back({sx, tot - sx});

    std::vector<int> col(n, -1);
    for (int j = 0; j < n_int; ++j) col[as.interior[j]] = j;
    Extension2D ext; ext.E = Mat(n, n_int); ext.interior = as.interior; ext.degree = degree;
    for (int g : as.interior) ext.E(g, col[g]) = 1.0;

    const int max_radius = std::max(nb, nb);
    for (int b : as.small) {
        const int ibx = b % nb, iby = b / nb;
        const double xb = cf.gx[b], yb = cf.gy[b];
        // Local scaling map T (n_terms x n_terms): poly_local = T @ poly reproduces
        // monomials ((x-xb)/h)^p ((y-yb)/h)^q.
        Mat T(n_terms, n_terms);
        for (int kr = 0; kr < n_terms; ++kr) {
            const int P = powers[kr][0], Qd = powers[kr][1];
            const double scale = 1.0 / (std::pow(h,P) * std::pow(h,Qd));
            for (int kc = 0; kc < n_terms; ++kc) {
                const int I = powers[kc][0], J = powers[kc][1];
                if (I <= P && J <= Qd) {
                    auto binom = [](int a, int bb){ double r=1; for(int t=0;t<bb;++t) r=r*(a-t)/(t+1); return r; };
                    T(kr,kc) = scale * binom(P,I) * binom(Qd,J)
                             * std::pow(-xb, P-I) * std::pow(-yb, Qd-J);
                }
            }
        }
        Mat poly_local = matmul(T, poly);                 // n_terms x n
        std::vector<double> target(n_terms);
        for (int r = 0; r < n_terms; ++r) target[r] = poly_local(r, b);

        // Grow a Chebyshev index neighborhood until the interior candidates span
        // every monomial row (full row rank n_terms), then take the MIN-NORM
        // least-squares reconstruction over the whole neighborhood.  Min-norm
        // gives the smallest-magnitude weights that reproduce all monomials, so
        // the conditioning floor is respected without brittle stencil selection.
        std::vector<int> cand;
        for (int radius = 0; radius <= max_radius; ++radius) {
            cand.clear();
            for (int g : as.interior) {
                const int gx = g % nb, gy = g / nb;
                if (std::max(std::abs(gx-ibx), std::abs(gy-iby)) <= radius) cand.push_back(g);
            }
            if ((int)cand.size() < n_terms) continue;
            // Full-row-rank test via QR of the scaled block transpose.
            Mat C(n_terms, (int)cand.size());
            for (int r = 0; r < n_terms; ++r) for (int cc = 0; cc < (int)cand.size(); ++cc) C(r,cc) = poly_local(r, cand[cc]);
            Mat A((int)cand.size(), n_terms);
            for (int r = 0; r < n_terms; ++r) for (int cc = 0; cc < (int)cand.size(); ++cc) A(cc,r) = C(r,cc);
            Mat Qd2, Rd2; householder_qr(A, Qd2, Rd2);
            double rmin = 1e30; for (int d = 0; d < n_terms; ++d) rmin = std::min(rmin, std::fabs(Rd2(d,d)));
            if (rmin > 1e-9) break;          // well-conditioned full-rank neighborhood
        }
        if ((int)cand.size() < n_terms) continue;

        Mat C(n_terms, (int)cand.size());
        for (int r = 0; r < n_terms; ++r) for (int cc = 0; cc < (int)cand.size(); ++cc) C(r,cc) = poly_local(r, cand[cc]);
        std::vector<double> w = minnorm_solve(C, target);
        for (int cc = 0; cc < (int)cand.size(); ++cc) {
            ext.E(b, col[cand[cc]]) = w[cc];
            ext.max_weight = std::max(ext.max_weight, std::fabs(w[cc]));
        }
    }

    double res = 0.0;
    for (int r = 0; r < n_terms; ++r) {
        std::vector<double> cr(n_int);
        for (int j = 0; j < n_int; ++j) cr[j] = poly(r, as.interior[j]);
        std::vector<double> Ecr = matvec(ext.E, cr);
        for (int i = 0; i < n; ++i) {
            const double tgt = (as.gamma[i] > 1e-22) ? poly(r, i) : 0.0;
            res = std::max(res, std::fabs(Ecr[i] - tgt));
        }
    }
    ext.poly_residual = res;
    return ext;
}

inline Mat cut2d_pullback(const Mat& A, const Mat& E) { return matmul(transpose(E), matmul(A, E)); }

// ---- helpers for accuracy gates: fluid quadrature points and point basis ----

struct FluidQuad2D { std::vector<double> x, y, w; };

// Collect every volume (fluid) quadrature point and weight, with the same
// clipped-cell Gauss-Duffy rule used by the assembly.
inline FluidQuad2D cut2d_fluid_quadrature(int n_elements, int p, const HalfPlaneCut& cut, int q = -1) {
    if (q < 0) q = std::max(10, p + 5);
    const double h = 1.0 / n_elements;
    std::vector<double> gleg, wleg; gauss_legendre(q, gleg, wleg);
    std::vector<double> u01(q), w01(q);
    for (int i = 0; i < q; ++i) { u01[i] = 0.5*(gleg[i]+1.0); w01[i] = 0.5*wleg[i]; }
    FluidQuad2D fq;
    for (int iy = 0; iy < n_elements; ++iy) for (int ix = 0; ix < n_elements; ++ix) {
        std::vector<P2> cell{{ix*h, iy*h}, {(ix+1)*h, iy*h}, {(ix+1)*h, (iy+1)*h}, {ix*h, (iy+1)*h}};
        std::vector<P2> cl = clip_halfplane(cell, cut);
        if (cl.size() < 3 || std::fabs(polygon_area(cl)) <= 1e-16) continue;
        if (polygon_area(cl) < 0.0) std::reverse(cl.begin(), cl.end());
        const P2 v0 = cl[0];
        for (size_t k = 1; k + 1 < cl.size(); ++k) {
            const P2 e1{cl[k][0]-v0[0], cl[k][1]-v0[1]};
            const P2 e2{cl[k+1][0]-v0[0], cl[k+1][1]-v0[1]};
            const double det = std::fabs(e1[0]*e2[1] - e1[1]*e2[0]);
            for (int gu = 0; gu < q; ++gu) for (int gv = 0; gv < q; ++gv) {
                const double uu = u01[gu], vv = u01[gv], omu = 1.0 - uu;
                fq.x.push_back(v0[0] + uu*e1[0] + omu*vv*e2[0]);
                fq.y.push_back(v0[1] + uu*e1[1] + omu*vv*e2[1]);
                fq.w.push_back(det * omu * w01[gu] * w01[gv]);
            }
        }
    }
    return fq;
}

// Nonzero tensor basis values at a point (the (p+1)^2 local functions).
inline void cut2d_point_basis(const CutForms2D& cf, double x, double y,
                              std::vector<int>& rows, std::vector<double>& phi) {
    const int p = cf.degree, nb = cf.nx;
    std::vector<double> vx, dvx, vy, dvy;
    const int lox = local_1d(cf.knots, p, nb, x, vx, dvx);
    const int loy = local_1d(cf.knots, p, nb, y, vy, dvy);
    rows.clear(); phi.clear();
    for (int ja = 0; ja <= p; ++ja) for (int ia = 0; ia <= p; ++ia) {
        rows.push_back((loy+ja)*nb + (lox+ia));
        phi.push_back(vy[ja]*vx[ia]);
    }
}

// Nonzero tensor basis x- and y-derivatives at a point (for gradient sampling).
inline void cut2d_point_grad(const CutForms2D& cf, double x, double y,
                             std::vector<int>& rows, std::vector<double>& dphix,
                             std::vector<double>& dphiy) {
    const int p = cf.degree, nb = cf.nx;
    std::vector<double> vx, dvx, vy, dvy;
    const int lox = local_1d(cf.knots, p, nb, x, vx, dvx);
    const int loy = local_1d(cf.knots, p, nb, y, vy, dvy);
    rows.clear(); dphix.clear(); dphiy.clear();
    for (int ja = 0; ja <= p; ++ja) for (int ia = 0; ia <= p; ++ia) {
        rows.push_back((loy+ja)*nb + (lox+ia));
        dphix.push_back(dvx[ia]*vy[ja]);
        dphiy.push_back(vx[ia]*dvy[ja]);
    }
}

#endif // CUT2D_H
