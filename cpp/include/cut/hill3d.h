// hill3d.h
//
// Three-dimensional periodic-hill operator layer: the verified 2D-curved cut
// (hill_cut2d.h) in the x-y plane TENSORED with a periodic spanwise direction z.
// The fluid is { y >= h(x) } x [0,Lz] with x,z periodic and the curved no-slip
// wall in y.  A 3D field is (n_red_2d) x (nz): the x-y part carries the reduced
// 2D-curved modes (dense, since the curved wall couples x and y), the z part is
// periodic (circulant).  Every operator is (2D apply) x (I_z) + (I_xy) x (z apply),
// so each apply is a stack of 2D matvecs plus a stack of circulant z applies.
//
// This is the geometry the stress-resolved DNS (sec:s6-hill) runs on.  The best-
// approximation gate here certifies the 3D tensor space inherits the curved-cut
// order p+1 from the x-y plane and the spectral resolution from periodic z.

#ifndef HILL3D_H
#define HILL3D_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "bspline.h"
#include "cut2d.h"
#include "hill_cut2d.h"

// --- periodic spanwise (z) 1D pieces: mass, basis eval, load, norm ---
struct PeriodicZ {
    int nz = 0, p = 0; double Lz = 1.0, hz = 0.0;
    Mat M;                                  // periodic mass (nz x nz, circulant dense)

    void build(int nz_, int p_, double Lz_) {
        nz = nz_; p = p_; Lz = Lz_; hz = Lz / nz;
        CoeffForms cf = assemble_forms_periodic(nz, p, hz);
        M = cf.M;
    }
    // Periodic cardinal basis value N_i(z) (centered, wrapped into [-Lz/2,Lz/2]).
    double basis(int i, double z) const {
        double d = z - i * hz;
        d -= Lz * std::floor(d / Lz + 0.5);   // wrap to nearest image
        return beta_value(p, d / hz);
    }
    double dbasis(int i, double z) const {    // d N_i / dz
        double d = z - i * hz;
        d -= Lz * std::floor(d / Lz + 0.5);
        return beta_deriv1(p, d / hz) / hz;
    }
    // L2 load b_i = \int_0^Lz N_i g, and the exact norm \int g^2, by panel Gauss.
    template <class G>
    void load_and_norm(G g, std::vector<double>& b, double& gnorm2, int q = -1) const {
        if (q < 0) q = p + 6;
        std::vector<double> gx, gw; gauss_legendre(q, gx, gw);
        b.assign(nz, 0.0); gnorm2 = 0.0;
        for (int e = 0; e < nz; ++e) {
            const double a = e * hz, c = (e + 1) * hz, half = 0.5 * hz, mid = 0.5 * (a + c);
            for (int k = 0; k < q; ++k) {
                const double z = mid + half * gx[k], w = gw[k] * half, gv = g(z);
                gnorm2 += w * gv * gv;
                // only the p+1 basis funcs near z contribute
                const int i0 = (int)std::floor(z / hz) - (p + 1);
                for (int di = 0; di <= 2 * (p + 1); ++di) {
                    const int i = ((i0 + di) % nz + nz) % nz;
                    b[i] += w * basis(i, z) * gv;
                }
            }
        }
    }
};

// --- the 3D hill context: 2D-curved reduced operators + periodic z ---
struct Hill3D {
    HillBar HB;
    CutForms2D cf2; ActiveSets2D as2; Extension2D ext2;
    int nr2 = 0;                 // reduced 2D-curved mode count
    Mat M2, Qx2, Qy2;            // reduced 2D mass / weak-derivative forms (nr2 x nr2)
    DenseMatrix M2d;             // dense copy for solves
    Mat M2inv, Mzinv;            // precomputed inverses (mass solve = matvec, not refactor)
    PeriodicZ Z;
    Mat Kz, Sz;                  // periodic z derivative form + stiffness (nz x nz)
    Mat K2, KN2;                 // reduced 2D stiffness + no-slip weak-penalty stiffness (nr2 x nr2)
    Mat Qx2Tr, Qy2Tr, KzTr;      // precomputed transposes (used by hill_apply_QT each step)
    double penalty_beta = 10.0;
    // Galerkin product machinery: x-y fluid quadrature + reduced sample matrix.
    FluidQuad2D fq2;             // 2D quad points (x,y,w), n2q of them
    int n2q = 0, nzq = 0;
    // SPARSE reduced 2D sample/gradient rows (per quad point, the few nonzero reduced
    // modes): basis sp_phi + physical x,y derivatives sp_dpx,sp_dpy.  Built DIRECTLY
    // (no dense n2q x nr2 intermediate, which is 3--35 GB at DNS scale); the sample/load
    // x-y matvecs are O(n2q * nnz) not O(n2q * nr2).
    std::vector<std::vector<int>> sp_idx;
    std::vector<std::vector<double>> sp_phi, sp_dpx, sp_dpy;
    std::vector<double> w2;      // 2D quad weights (n2q)
    Mat Vz, dVz; std::vector<double> wz, zq;   // periodic z basis + derivative at z quad nodes (nzq x nz)

    void build(int ne_xy, int p, int nz, double Lz, bool physical = false) {
        HB = HillBar{}; HB.physical = physical;
        cf2 = assemble_hill_cut_2d(ne_xy, p, HB);
        as2 = cut2d_active_sets(cf2);
        ext2 = cut2d_extension(cf2, as2);
        nr2 = (int)ext2.interior.size();
        M2  = cut2d_pullback(cf2.H,  ext2.E);
        Qx2 = cut2d_pullback(cf2.Qx, ext2.E);
        Qy2 = cut2d_pullback(cf2.Qy, ext2.E);
        M2d = to_dense(M2);
        { bool ok; M2inv = inverse(M2, &ok); }
        Z.build(nz, p, Lz);
        { bool ok; Mzinv = inverse(Z.M, &ok); }
        Qx2Tr = transpose(Qx2); Qy2Tr = transpose(Qy2);
        CoeffForms zf = assemble_forms_periodic(nz, p, Lz/nz);
        Kz = zf.K; Sz = zf.S; KzTr = transpose(Kz);
        // reduced 2D stiffness + symmetric no-slip weak-penalty stiffness on the curved wall.
        K2 = cut2d_pullback(cf2.K, ext2.E);
        WallTraces2D wt = assemble_hill_wall_traces(cf2, HB, ne_xy);
        Mat Mw2 = cut2d_pullback(wt.Mw, ext2.E), Bn2 = cut2d_pullback(wt.Bn, ext2.E);
        const double hcell = (physical ? HB.Ly/ne_xy : 1.0/ne_xy), beta = penalty_beta * p * p;
        KN2 = K2;
        for (int i=0;i<nr2;++i) for (int j=0;j<nr2;++j)
            KN2(i,j) += -Bn2(i,j) - Bn2(j,i) + (beta/hcell)*Mw2(i,j);
        // 2D Galerkin quadrature + reduced sample matrix PHIr2(q,m) = sum_b phi_b(q) E2(b,m).
        fq2 = hill_fluid_quadrature(ne_xy, p, HB);
        n2q = (int)fq2.w.size(); w2 = fq2.w;
        // Physical metric for the Galerkin sample/load machinery (consistent with the
        // operator forms): fluid-volume weights carry the Jacobian Lx*Ly, sampled
        // gradients carry 1/Lx,1/Ly.  physical=false => mLx=mLy=1 (unchanged).
        const double mLx = HB.physical ? HB.Lx : 1.0, mLy = HB.physical ? HB.Ly : 1.0;
        if (HB.physical) for (double& w : w2) w *= mLx*mLy;
        // Sparse E2 rows (most full modes map to a single reduced column).
        const int nfull = cf2.n;
        std::vector<std::vector<std::pair<int,double>>> E2rows(nfull);
        for (int b = 0; b < nfull; ++b) for (int m = 0; m < nr2; ++m) { const double e = ext2.E(b,m); if (e!=0.0) E2rows[b].push_back({m,e}); }
        // Build the SPARSE reduced sample/gradient rows DIRECTLY -- no dense n2q x nr2
        // PHIr2 intermediate (that dense form is 3--35 GB at DNS scale and ~100x larger
        // than the sparse rows it feeds; only the sparse rows are ever used).  Per quad
        // node accumulate per reduced mode in scratch, keep the sorted nonzeros.
        sp_idx.assign(n2q,{}); sp_phi.assign(n2q,{}); sp_dpx.assign(n2q,{}); sp_dpy.assign(n2q,{});
        std::vector<double> ap(nr2,0.0), apx(nr2,0.0), apy(nr2,0.0);
        std::vector<char> seen(nr2,0); std::vector<int> touched; touched.reserve(256);
        for (int q = 0; q < n2q; ++q) {
            std::vector<int> rows; std::vector<double> phi, dpx, dpy;
            cut2d_point_basis(cf2, fq2.x[q], fq2.y[q], rows, phi);
            cut2d_point_grad (cf2, fq2.x[q], fq2.y[q], rows, dpx, dpy);
            for (size_t a = 0; a < rows.size(); ++a) {
                const int b = rows[a];
                for (auto& mw : E2rows[b]) {
                    const int m = mw.first;
                    if (!seen[m]) { seen[m]=1; touched.push_back(m); }
                    ap[m]  += phi[a]      * mw.second;
                    apx[m] += (dpx[a]/mLx) * mw.second;   // physical d/dx = (1/Lx) d/dx'
                    apy[m] += (dpy[a]/mLy) * mw.second;   // physical d/dy = (1/Ly) d/dy'
                }
            }
            std::sort(touched.begin(), touched.end());     // sorted modes => bit-identical to the dense extraction
            for (int m : touched) if (ap[m]!=0.0 || apx[m]!=0.0 || apy[m]!=0.0) {
                sp_idx[q].push_back(m); sp_phi[q].push_back(ap[m]);
                sp_dpx[q].push_back(apx[m]); sp_dpy[q].push_back(apy[m]);
            }
            for (int m : touched) { ap[m]=0.0; apx[m]=0.0; apy[m]=0.0; seen[m]=0; }
            touched.clear();
        }
        // periodic z quadrature + sample matrix Vz(qz,iz) = N_iz(z_qz).
        const int qo = p + 6;
        std::vector<double> gx, gw; gauss_legendre(qo, gx, gw);
        for (int e = 0; e < nz; ++e) {
            const double a = e*Z.hz, c = (e+1)*Z.hz, half = 0.5*Z.hz, mid = 0.5*(a+c);
            for (int k = 0; k < qo; ++k) { zq.push_back(mid+half*gx[k]); wz.push_back(gw[k]*half); }
        }
        nzq = (int)zq.size();
        Vz = Mat(nzq, nz); dVz = Mat(nzq, nz);
        for (int q = 0; q < nzq; ++q) for (int i = 0; i < nz; ++i) { Vz(q,i) = Z.basis(i, zq[q]); dVz(q,i) = Z.dbasis(i, zq[q]); }
    }
    int n() const { return nr2 * Z.nz; }
};

// Quadrature-grid field: (n2q x nzq), q2 the 2D node, qz the spanwise node.
struct QField { int n2q=0, nzq=0; std::vector<double> v;
    QField() {} QField(int a,int b):n2q(a),nzq(b),v((size_t)a*b,0.0){}
    double& at(int q2,int qz){ return v[(size_t)q2*nzq+qz]; }
    double  at(int q2,int qz) const { return v[(size_t)q2*nzq+qz]; }
};

// 3D field: c[m2 * nz + iz], m2 the reduced 2D index, iz the spanwise index.
struct FieldH { int nr2 = 0, nz = 0; std::vector<double> v;
    FieldH() {}
    FieldH(int a, int b) : nr2(a), nz(b), v((size_t)a * b, 0.0) {}
    double& at(int m2, int iz) { return v[(size_t)m2 * nz + iz]; }
    double  at(int m2, int iz) const { return v[(size_t)m2 * nz + iz]; }
};

// Apply a 2D operator (nr2 x nr2) on the x-y part, for every z-slice.
inline FieldH hill_apply2d(const Hill3D& H, const FieldH& f, const Mat& op2) {
    FieldH o(H.nr2, H.Z.nz);
    for (int iz = 0; iz < H.Z.nz; ++iz)
        for (int i = 0; i < H.nr2; ++i) {
            double s = 0.0; const double* row = &op2.a[(size_t)i * op2.c];
            for (int j = 0; j < H.nr2; ++j) s += row[j] * f.at(j, iz);
            o.at(i, iz) = s;
        }
    return o;
}
// Apply a z operator (nz x nz) on the spanwise part, for every 2D mode.
inline FieldH hill_applyz(const Hill3D& H, const FieldH& f, const Mat& opz) {
    FieldH o(H.nr2, H.Z.nz);
    for (int m = 0; m < H.nr2; ++m)
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            double s = 0.0; const double* row = &opz.a[(size_t)iz * opz.c];
            for (int jz = 0; jz < H.Z.nz; ++jz) s += row[jz] * f.at(m, jz);
            o.at(m, iz) = s;
        }
    return o;
}
// Mass apply M3 = M2 (x) Mz, and the separable mass solve M3^{-1}.
inline FieldH hill_mass_apply(const Hill3D& H, const FieldH& f) {
    return hill_applyz(H, hill_apply2d(H, f, H.M2), H.Z.M);
}
inline FieldH hill_mass_solve(const Hill3D& H, const FieldH& b) {
    // solve (M2 x Mz) c = b via the PRECOMPUTED inverses (matvec, no refactor):
    // apply M2^{-1} per z-slice (op along axis 0), then Mz^{-1} per 2D mode (axis 1).
    FieldH t = hill_apply2d(H, b, H.M2inv);
    return hill_applyz(H, t, H.Mzinv);
}

// --- Galerkin product machinery on the 3D hill (comp_cut3d analog) ---

// Sample a reduced field to the quadrature grid: PHIr2 (x-y) then Vz (z).
inline QField hill_sample(const Hill3D& H, const FieldH& c) {
    FieldH tmp(H.n2q, H.Z.nz);                       // (n2q x nz), sparse x-y matvec
    for (int q = 0; q < H.n2q; ++q) {
        const auto& idx = H.sp_idx[q]; const auto& val = H.sp_phi[q];
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            double s = 0.0; for (size_t a = 0; a < idx.size(); ++a) s += val[a] * c.at(idx[a], iz);
            tmp.at(q, iz) = s;
        }
    }
    QField out(H.n2q, H.nzq);
    for (int q = 0; q < H.n2q; ++q)
        for (int qz = 0; qz < H.nzq; ++qz) {
            double s = 0.0; const double* row = &H.Vz.a[(size_t)qz*H.Z.nz];
            for (int iz = 0; iz < H.Z.nz; ++iz) s += row[iz] * tmp.at(q, iz);
            out.at(q, qz) = s;
        }
    return out;
}

// Sample a reduced field's dir-derivative at the quadrature grid.
//   dir 0: dPHIr2x (x) Vz ; dir 1: dPHIr2y (x) Vz ; dir 2: PHIr2 (x) dVz.
inline QField hill_grad(const Hill3D& H, const FieldH& c, int dir) {
    const std::vector<std::vector<double>>& SV = (dir==0)? H.sp_dpx : (dir==1)? H.sp_dpy : H.sp_phi;
    const Mat& B = (dir==2)? H.dVz : H.Vz;
    FieldH tmp(H.n2q, H.Z.nz);
    for (int q = 0; q < H.n2q; ++q) {
        const auto& idx = H.sp_idx[q]; const auto& val = SV[q];
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            double s = 0.0; for (size_t a = 0; a < idx.size(); ++a) s += val[a] * c.at(idx[a], iz);
            tmp.at(q, iz) = s;
        }
    }
    QField out(H.n2q, H.nzq);
    for (int q = 0; q < H.n2q; ++q)
        for (int qz = 0; qz < H.nzq; ++qz) {
            double s = 0.0; const double* row = &B.a[(size_t)qz*H.Z.nz];
            for (int iz = 0; iz < H.Z.nz; ++iz) s += row[iz] * tmp.at(q, iz);
            out.at(q, qz) = s;
        }
    return out;
}

// Galerkin load of a quadrature field: PHIr2^T W2 (x-y) and Vz^T Wz (z) -> reduced.
inline FieldH hill_load(const Hill3D& H, const QField& fq) {
    FieldH tmp(H.n2q, H.Z.nz);                        // integrate z
    for (int q = 0; q < H.n2q; ++q)
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            double s = 0.0;
            for (int qz = 0; qz < H.nzq; ++qz) s += H.Vz(qz,iz) * H.wz[qz] * fq.at(q, qz);
            tmp.at(q, iz) = s;
        }
    FieldH red(H.nr2, H.Z.nz);                         // integrate + reduce x-y (sparse scatter)
    for (int q = 0; q < H.n2q; ++q) {
        const auto& idx = H.sp_idx[q]; const auto& val = H.sp_phi[q]; const double wq = H.w2[q];
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            const double t = wq * tmp.at(q, iz);
            for (size_t a = 0; a < idx.size(); ++a) red.at(idx[a], iz) += val[a] * t;
        }
    }
    return red;
}

inline FieldH hill_l2_project(const Hill3D& H, const QField& fq) {
    return hill_mass_solve(H, hill_load(H, fq));
}

// Gradient load: reduce( \int (d phi_a / dx_dir) g ) = adjoint of hill_grad.
// dir 0: dPHIr2x (x) Vz ; dir 1: dPHIr2y (x) Vz ; dir 2: PHIr2 (x) dVz.
inline FieldH hill_load_grad(const Hill3D& H, const QField& fq, int dir) {
    const std::vector<std::vector<double>>& SV = (dir==0)? H.sp_dpx : (dir==1)? H.sp_dpy : H.sp_phi;
    const Mat& B = (dir==2)? H.dVz : H.Vz;
    FieldH tmp(H.n2q, H.Z.nz);                        // integrate z
    for (int q = 0; q < H.n2q; ++q)
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            double s = 0.0;
            for (int qz = 0; qz < H.nzq; ++qz) s += B(qz,iz) * H.wz[qz] * fq.at(q, qz);
            tmp.at(q, iz) = s;
        }
    FieldH red(H.nr2, H.Z.nz);                         // integrate + reduce x-y (sparse scatter vs grad basis)
    for (int q = 0; q < H.n2q; ++q) {
        const auto& idx = H.sp_idx[q]; const auto& val = SV[q]; const double wq = H.w2[q];
        for (int iz = 0; iz < H.Z.nz; ++iz) {
            const double t = wq * tmp.at(q, iz);
            for (size_t a = 0; a < idx.size(); ++a) red.at(idx[a], iz) += val[a] * t;
        }
    }
    return red;
}

// Apply the reduced 3D weak-derivative form Q_dir: x = Qx2(x)Mz, y = Qy2(x)Mz, z = M2(x)Kz.
inline FieldH hill_apply_Q(const Hill3D& H, const FieldH& f, int dir) {
    if (dir == 0) return hill_applyz(H, hill_apply2d(H, f, H.Qx2), H.Z.M);
    if (dir == 1) return hill_applyz(H, hill_apply2d(H, f, H.Qy2), H.Z.M);
    return hill_applyz(H, hill_apply2d(H, f, H.M2), H.Kz);    // z
}

// Transpose of the reduced weak-derivative form: Q_dir^T.  x: Qx2^T(x)Mz ;
// y: Qy2^T(x)Mz ; z: M2(x)Kz^T  (M2,Mz symmetric).
inline FieldH hill_apply_QT(const Hill3D& H, const FieldH& f, int dir) {
    if (dir == 0) return hill_applyz(H, hill_apply2d(H, f, H.Qx2Tr), H.Z.M);
    if (dir == 1) return hill_applyz(H, hill_apply2d(H, f, H.Qy2Tr), H.Z.M);
    return hill_applyz(H, hill_apply2d(H, f, H.M2), H.KzTr);   // z
}

// Weak divergence load of a velocity:  D u = sum_d Q_d u_d  (load form).
inline FieldH hill_weak_div(const Hill3D& H, const FieldH& ux, const FieldH& uy, const FieldH& uz) {
    FieldH d = hill_apply_Q(H, ux, 0), dy = hill_apply_Q(H, uy, 1), dz = hill_apply_Q(H, uz, 2);
    for (size_t i = 0; i < d.v.size(); ++i) d.v[i] += dy.v[i] + dz.v[i];
    return d;
}

// Pressure-Poisson operator  L p = D M^{-1} D^T p = sum_d Q_d M^{-1} Q_d^T p  (SPD,
// constant-pressure null).
inline FieldH hill_L_apply(const Hill3D& H, const FieldH& p) {
    FieldH out(H.nr2, H.Z.nz);
    for (int d = 0; d < 3; ++d) {
        FieldH qd = hill_apply_Q(H, hill_mass_solve(H, hill_apply_QT(H, p, d)), d);
        for (size_t i = 0; i < out.v.size(); ++i) out.v[i] += qd.v[i];
    }
    return out;
}

// Helmholtz/Hodge projection of a velocity (ux,uy,uz) onto the weakly
// divergence-free space: solve L p = D u by CG (constant-pressure null removed by
// mean subtraction), then u_d <- u_d - M^{-1} Q_d^T p.  Returns the iteration count.
inline int hill_project(Hill3D& H, FieldH& ux, FieldH& uy, FieldH& uz,
                        int max_iter = 400, double tol = 1e-10) {
    auto dot = [](const FieldH& a, const FieldH& b){ double s=0; for(size_t i=0;i<a.v.size();++i) s+=a.v[i]*b.v[i]; return s; };
    auto submean = [](FieldH& x){ double m=0; for(double v:x.v) m+=v; m/=(double)x.v.size(); for(double& v:x.v) v-=m; };
    FieldH g = hill_weak_div(H, ux, uy, uz); submean(g);     // rhs, compatible with the null
    const double gn = std::sqrt(std::max(dot(g,g), 1e-300));
    FieldH p(H.nr2, H.Z.nz), r = g, pdir = g;
    double rs = dot(r,r); int it = 0;
    for (; it < max_iter; ++it) {
        FieldH Ap = hill_L_apply(H, pdir); submean(Ap);
        const double a = rs / std::max(dot(pdir,Ap), 1e-300);
        for (size_t i=0;i<p.v.size();++i){ p.v[i]+=a*pdir.v[i]; r.v[i]-=a*Ap.v[i]; }
        submean(r);
        const double rs2 = dot(r,r);
        if (std::sqrt(rs2)/gn < tol) { ++it; break; }
        const double beta = rs2/rs; for (size_t i=0;i<pdir.v.size();++i) pdir.v[i]=r.v[i]+beta*pdir.v[i];
        rs = rs2;
    }
    submean(p);
    FieldH* U[3] = {&ux,&uy,&uz};
    for (int d=0; d<3; ++d){ FieldH gd = hill_mass_solve(H, hill_apply_QT(H, p, d));
        for (size_t i=0;i<U[d]->v.size();++i) U[d]->v[i] -= gd.v[i]; }
    return it;
}

inline FieldH hill_skew_conv(const Hill3D&, const QField&, const QField&, const QField&, const FieldH&);

// 3D viscous (Laplacian) apply: (K_N (x) Mz) + (M2 (x) Sz).  K_N carries the
// symmetric no-slip weak-penalty terms on the curved wall; Sz is periodic stiffness.
inline FieldH hill_visc_apply(const Hill3D& H, const FieldH& u) {
    FieldH a = hill_applyz(H, hill_apply2d(H, u, H.KN2), H.Z.M);
    FieldH b = hill_applyz(H, hill_apply2d(H, u, H.M2),  H.Sz);
    for (size_t i = 0; i < a.v.size(); ++i) a.v[i] += b.v[i];
    return a;
}

// Incompressible NS state and one SSP-RK3 step (skew advection + weak-penalty viscous,
// then Helmholtz projection to weak-div-free at each stage).
struct HillNS { FieldH ux, uy, uz; };

inline HillNS hill_ns_rhs(const Hill3D& H, const HillNS& s, double nu) {
    QField wx = hill_sample(H, s.ux), wy = hill_sample(H, s.uy), wz = hill_sample(H, s.uz);
    const FieldH* U[3] = {&s.ux,&s.uy,&s.uz};
    HillNS L; FieldH* Lo[3] = {&L.ux,&L.uy,&L.uz};
    for (int i = 0; i < 3; ++i) {
        FieldH conv = hill_skew_conv(H, wx, wy, wz, *U[i]);
        FieldH visc = hill_mass_solve(H, hill_visc_apply(H, *U[i]));
        *Lo[i] = conv;
        for (size_t k = 0; k < conv.v.size(); ++k) Lo[i]->v[k] = -conv.v[k] - nu*visc.v[k];
    }
    return L;
}

inline void hill_ns_step(Hill3D& H, HillNS& s, double dt, double nu) {
    auto stage = [&](const HillNS& u, const HillNS& L){
        HillNS o; o.ux=u.ux; o.uy=u.uy; o.uz=u.uz;
        for (size_t k=0;k<o.ux.v.size();++k){ o.ux.v[k]+=dt*L.ux.v[k]; o.uy.v[k]+=dt*L.uy.v[k]; o.uz.v[k]+=dt*L.uz.v[k]; }
        hill_project(H, o.ux, o.uy, o.uz); return o; };
    auto comb = [&](double a, const HillNS& A, double b, const HillNS& B){
        HillNS o; o.ux=A.ux; o.uy=A.uy; o.uz=A.uz;
        for (size_t k=0;k<o.ux.v.size();++k){ o.ux.v[k]=a*A.ux.v[k]+b*B.ux.v[k]; o.uy.v[k]=a*A.uy.v[k]+b*B.uy.v[k]; o.uz.v[k]=a*A.uz.v[k]+b*B.uz.v[k]; } return o; };
    HillNS u1 = stage(s, hill_ns_rhs(H,s,nu));
    HillNS u2 = comb(0.75, s, 0.25, stage(u1, hill_ns_rhs(H,u1,nu)));
    s = comb(1.0/3.0, s, 2.0/3.0, stage(u2, hill_ns_rhs(H,u2,nu)));
}

// Implicit viscous solve: (M3 + a K_N3) u = rhs by CG (SPD; M3 dominates).
inline FieldH hill_viscous_solve(const Hill3D& H, const FieldH& rhs, double a,
                                 int max_iter = 300, double tol = 1e-9) {
    auto dot = [](const FieldH& x, const FieldH& y){ double s=0; for(size_t i=0;i<x.v.size();++i) s+=x.v[i]*y.v[i]; return s; };
    auto Aapply = [&](const FieldH& u){ FieldH r=hill_mass_apply(H,u), v=hill_visc_apply(H,u);
        for(size_t i=0;i<r.v.size();++i){ r.v[i]+=a*v.v[i]; } return r; };
    const double bn = std::sqrt(std::max(dot(rhs,rhs),1e-300));
    FieldH u(H.nr2,H.Z.nz), r=rhs, pdir=rhs; double rs=dot(r,r);
    for (int it=0; it<max_iter; ++it){ FieldH Ap=Aapply(pdir); double al=rs/std::max(dot(pdir,Ap),1e-300);
        for(size_t i=0;i<u.v.size();++i){ u.v[i]+=al*pdir.v[i]; r.v[i]-=al*Ap.v[i]; }
        double rs2=dot(r,r); if(std::sqrt(rs2)/bn<tol) break; double be=rs2/rs;
        for(size_t i=0;i<pdir.v.size();++i){ pdir.v[i]=r.v[i]+be*pdir.v[i]; } rs=rs2; }
    return u;
}

// IMEX step: SSP-RK3 explicit energy-conserving advection (+ project), then a
// backward-Euler implicit viscous solve (weak-penalty no-slip) so dt is advective only.
inline void hill_ns_step_imex(Hill3D& H, HillNS& s, double dt, double nu, double fx = 0.0) {
    auto advect = [&](const HillNS& u){ QField wx=hill_sample(H,u.ux), wy=hill_sample(H,u.uy), wz=hill_sample(H,u.uz);
        const FieldH* U[3]={&u.ux,&u.uy,&u.uz}; HillNS C; FieldH* Co[3]={&C.ux,&C.uy,&C.uz};
        for(int i=0;i<3;++i){ FieldH c=hill_skew_conv(H,wx,wy,wz,*U[i]); *Co[i]=c; for(auto& v:Co[i]->v) v=-v; } return C; };
    auto stage = [&](const HillNS& u, const HillNS& L){ HillNS o=u;
        for(size_t k=0;k<o.ux.v.size();++k){ o.ux.v[k]+=dt*L.ux.v[k]; o.uy.v[k]+=dt*L.uy.v[k]; o.uz.v[k]+=dt*L.uz.v[k]; }
        hill_project(H,o.ux,o.uy,o.uz); return o; };
    auto comb=[&](double a,const HillNS&A,double b,const HillNS&B){ HillNS o=A;
        for(size_t k=0;k<o.ux.v.size();++k){ o.ux.v[k]=a*A.ux.v[k]+b*B.ux.v[k]; o.uy.v[k]=a*A.uy.v[k]+b*B.uy.v[k]; o.uz.v[k]=a*A.uz.v[k]+b*B.uz.v[k]; } return o; };
    HillNS u1=stage(s,advect(s));
    HillNS u2=comb(0.75,s,0.25,stage(u1,advect(u1)));
    HillNS us=comb(1.0/3.0,s,2.0/3.0,stage(u2,advect(u2)));     // advected, projected
    if (fx != 0.0) for (double& v : us.ux.v) v += dt*fx;        // streamwise body-force drive (constant -> div-free)
    FieldH* U[3]={&us.ux,&us.uy,&us.uz};
    for(int i=0;i<3;++i){ FieldH rhs=hill_mass_apply(H,*U[i]); *U[i]=hill_viscous_solve(H,rhs,dt*nu); }
    hill_project(H,us.ux,us.uy,us.uz);
    s=us;
}

// Saddle Schur operator  S p = G^T A^{-1} G p  (A = M + a K_N3): the consistent
// pressure operator (same A^{-1} as the viscous solve), so the velocity update
// is EXACTLY weakly divergence-free.  Reuses hill_L_apply's structure.
inline FieldH hill_S_apply(const Hill3D& H, const FieldH& p, double a) {
    FieldH acc(H.nr2, H.Z.nz);
    for (int d = 0; d < 3; ++d) {
        FieldH qd = hill_apply_Q(H, hill_viscous_solve(H, hill_apply_QT(H, p, d), a, 200, 1e-10), d);
        for (size_t i = 0; i < acc.v.size(); ++i) acc.v[i] += qd.v[i];
    }
    return acc;
}

// Consistent IMEX SADDLE step (the fix for driven flow): explicit SSP-RK3 skew
// advection (+ forcing, no projection), then the coupled implicit viscous +
// pressure solve  [A G; G^T 0][u;p] = [M u*; 0]  via the Schur complement, so the
// velocity is exactly weakly divergence-free and the no-slip is consistent.
inline void hill_saddle_step(Hill3D& H, HillNS& s, double dt, double nu, double fx = 0.0) {
    const double a = dt*nu;
    auto advect=[&](const HillNS& u){ QField wx=hill_sample(H,u.ux),wy=hill_sample(H,u.uy),wz=hill_sample(H,u.uz);
        const FieldH* U[3]={&u.ux,&u.uy,&u.uz}; HillNS C; FieldH* Co[3]={&C.ux,&C.uy,&C.uz};
        for(int i=0;i<3;++i){ FieldH c=hill_skew_conv(H,wx,wy,wz,*U[i]); *Co[i]=c; for(auto&v:Co[i]->v) v=-v; } return C; };
    auto stage=[&](const HillNS&u,const HillNS&L){ HillNS o=u; for(size_t k=0;k<o.ux.v.size();++k){o.ux.v[k]+=dt*L.ux.v[k];o.uy.v[k]+=dt*L.uy.v[k];o.uz.v[k]+=dt*L.uz.v[k];} return o; };
    auto comb=[&](double c1,const HillNS&A,double c2,const HillNS&B){ HillNS o=A; for(size_t k=0;k<o.ux.v.size();++k){o.ux.v[k]=c1*A.ux.v[k]+c2*B.ux.v[k];o.uy.v[k]=c1*A.uy.v[k]+c2*B.uy.v[k];o.uz.v[k]=c1*A.uz.v[k]+c2*B.uz.v[k];} return o; };
    HillNS u1=stage(s,advect(s));
    HillNS u2=comb(0.75,s,0.25,stage(u1,advect(u1)));
    HillNS us=comb(1.0/3.0,s,2.0/3.0,stage(u2,advect(u2)));
    if(fx!=0.0) for(double& v:us.ux.v) v+=dt*fx;
    // u_tent = A^{-1}(M u*)
    HillNS ut; FieldH* US[3]={&us.ux,&us.uy,&us.uz}; FieldH* UT[3]={&ut.ux,&ut.uy,&ut.uz};
    for(int d=0;d<3;++d) *UT[d]=hill_viscous_solve(H, hill_mass_apply(H,*US[d]), a);
    // Schur solve S p = G^T u_tent (CG, near-null mean-subtracted)
    auto dot=[](const FieldH&x,const FieldH&y){double r=0;for(size_t i=0;i<x.v.size();++i)r+=x.v[i]*y.v[i];return r;};
    // S is non-singular here (boundary terms regularize; S(const) != 0), so NO
    // mean-subtraction (that removes a real component and corrupts the solve).
    FieldH rhs=hill_weak_div(H,ut.ux,ut.uy,ut.uz);
    FieldH p(H.nr2,H.Z.nz),r=rhs,pd=rhs; double rs=dot(r,r), bn=std::sqrt(std::max(rs,1e-300));
    for(int it=0;it<600;++it){ FieldH Ap=hill_S_apply(H,pd,a); double al=rs/std::max(dot(pd,Ap),1e-300);
        for(size_t i=0;i<p.v.size();++i){p.v[i]+=al*pd.v[i];r.v[i]-=al*Ap.v[i];} double rs2=dot(r,r);
        if(std::sqrt(rs2)/bn<1e-9){ break; } double be=rs2/rs; for(size_t i=0;i<pd.v.size();++i){ pd.v[i]=r.v[i]+be*pd.v[i]; } rs=rs2; }
    // u = u_tent - A^{-1} G p
    for(int d=0;d<3;++d){ FieldH gp=hill_viscous_solve(H, hill_apply_QT(H,p,d), a); for(size_t i=0;i<UT[d]->v.size();++i) UT[d]->v[i]-=gp.v[i]; }
    s=ut;
}

// Discrete kinetic energy 0.5 sum_i u_i^T M u_i.
inline double hill_ke(const Hill3D& H, const HillNS& s) {
    FieldH Mx=hill_mass_apply(H,s.ux), My=hill_mass_apply(H,s.uy), Mz=hill_mass_apply(H,s.uz);
    double e=0; for(size_t k=0;k<Mx.v.size();++k) e += s.ux.v[k]*Mx.v[k]+s.uy.v[k]*My.v[k]+s.uz.v[k]*Mz.v[k];
    return 0.5*e;
}

// Conservative weak divergence: M^{-1} sum_dir Q_dir M^{-1} load(F_dir).
inline FieldH hill_wdiv(const Hill3D& H, const QField& Fx, const QField& Fy, const QField& Fz) {
    const QField* F[3] = { &Fx, &Fy, &Fz };
    FieldH acc(H.nr2, H.Z.nz);
    for (int d = 0; d < 3; ++d) {
        FieldH fc = hill_mass_solve(H, hill_load(H, *F[d]));
        FieldH Qf = hill_apply_Q(H, fc, d);
        for (size_t i = 0; i < acc.v.size(); ++i) acc.v[i] += Qf.v[i];
    }
    return hill_mass_solve(H, acc);
}

// Energy-conserving skew-symmetric convection of a scalar u_i by velocity w
// (sampled at quad as wxq,wyq,wzq):  C_skew = 0.5[ proj(w.grad u_i) + div(w u_i) ].
// Its M-quadratic form is the boundary KE flux alone, so for a field vanishing on
// the wall the discrete kinetic energy injection is zero.
inline FieldH hill_skew_conv(const Hill3D& H, const QField& wxq, const QField& wyq,
                             const QField& wzq, const FieldH& ui) {
    QField uiq = hill_sample(H, ui);
    QField gx = hill_grad(H, ui, 0), gy = hill_grad(H, ui, 1), gz = hill_grad(H, ui, 2);
    QField conv(H.n2q, H.nzq), Fx(H.n2q, H.nzq), Fy(H.n2q, H.nzq), Fz(H.n2q, H.nzq);
    for (size_t k = 0; k < conv.v.size(); ++k) {
        conv.v[k] = wxq.v[k]*gx.v[k] + wyq.v[k]*gy.v[k] + wzq.v[k]*gz.v[k];   // w.grad u
        Fx.v[k] = wxq.v[k]*uiq.v[k]; Fy.v[k] = wyq.v[k]*uiq.v[k]; Fz.v[k] = wzq.v[k]*uiq.v[k]; // w u
    }
    // Advective half: load(w.grad u).  Divergence half: the DIRECT weak form
    // -\int grad phi . (w u) (volume only; the boundary KE flux is added by the
    // wall/no-slip treatment).  Sharing the same quadrature product makes the two
    // halves discretely adjoint, so u^T M C = boundary KE flux alone (energy conserving).
    FieldH load_adv = hill_load(H, conv);
    FieldH out(H.nr2, H.Z.nz);
    FieldH dgx = hill_load_grad(H, Fx, 0), dgy = hill_load_grad(H, Fy, 1), dgz = hill_load_grad(H, Fz, 2);
    for (size_t k = 0; k < out.v.size(); ++k)
        out.v[k] = 0.5*(load_adv.v[k] - (dgx.v[k] + dgy.v[k] + dgz.v[k]));
    return hill_mass_solve(H, out);
}

// Total integral over the fluid of a reduced field: ones^T (M3 c).
inline double hill_total(const Hill3D& H, const FieldH& c) {
    FieldH Mc = hill_mass_apply(H, c);
    double s = 0.0; for (double v : Mc.v) s += v; return s;
}

// --- conservative compressible RHS on the hill (comp_rhs analog) ---
struct HillCompState { FieldH rho, mx, my, mz, E; };

inline HillCompState hill_comp_rhs(const Hill3D& H, const HillCompState& s,
                                   double gamma = 1.4, bool with_pressure = true) {
    QField rq=hill_sample(H,s.rho), mxq=hill_sample(H,s.mx), myq=hill_sample(H,s.my),
           mzq=hill_sample(H,s.mz), Eq=hill_sample(H,s.E);
    const size_t Nq = rq.v.size();
    QField uq=rq, vq=rq, wq=rq, pq=rq;
    for (size_t i=0;i<Nq;++i){ const double ir=1.0/rq.v[i];
        uq.v[i]=mxq.v[i]*ir; vq.v[i]=myq.v[i]*ir; wq.v[i]=mzq.v[i]*ir;
        const double ke=0.5*rq.v[i]*(uq.v[i]*uq.v[i]+vq.v[i]*vq.v[i]+wq.v[i]*wq.v[i]);
        pq.v[i]=(gamma-1.0)*(Eq.v[i]-ke); }
    auto mul=[&](const QField&a,const QField&b){ QField o=a; for(size_t i=0;i<Nq;++i)o.v[i]=a.v[i]*b.v[i]; return o; };
    auto neg=[&](FieldH f){ for(double&v:f.v)v=-v; return f; };

    HillCompState R;
    R.rho = neg(hill_wdiv(H, mxq, myq, mzq));
    FieldH pcoef = hill_mass_solve(H, hill_load(H, pq));
    const QField* mq[3]={&mxq,&myq,&mzq}; FieldH* Rm[3]={&R.mx,&R.my,&R.mz};
    for (int i=0;i<3;++i){
        FieldH conv = hill_wdiv(H, mul(*mq[i],uq), mul(*mq[i],vq), mul(*mq[i],wq));
        FieldH gradp = hill_mass_solve(H, hill_apply_Q(H, pcoef, i));
        const double pw = with_pressure?1.0:0.0;
        *Rm[i] = conv; for (size_t k=0;k<conv.v.size();++k) Rm[i]->v[k] = -conv.v[k] - pw*gradp.v[k];
    }
    QField aE=Eq; for(size_t i=0;i<Nq;++i) aE.v[i]=Eq.v[i]+pq.v[i];
    R.E = neg(hill_wdiv(H, mul(aE,uq), mul(aE,vq), mul(aE,wq)));
    return R;
}

#endif // HILL3D_H
