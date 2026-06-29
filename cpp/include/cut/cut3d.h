// cut3d.h
//
// Three-dimensional sharp-cut operator layer for a tensor-product cut box, the
// C++ port of code/cut_3d_sbp.py::assemble_cut_3d.  The 3D cut box places each
// face a sub-cell fraction inside the first interior knot, so the corner cell is
// a 3D sliver whose volume ratio is the product of the three face fractions.
//
// The decisive structural fact (cut_3d_sbp.py): everything is a tensor product
// of the verified 1D cut operators (cut1d.h),
//
//   E  = Ez (x) Ey (x) Ex,         H  = Hz (x) Hy (x) Hx,
//   Qx = Hz (x) Hy (x) Qx_1d,      Sx = Hz (x) Hy (x) Bx_1d,   (and cyclically),
//   K  = sum over axes of  Hz (x) Hy (x) Kx_1d,
//
// and the EᵀAE congruence FACTORS per axis:  reduced Hx = Ex^T Hx_1d Ex, etc.
// So the reduced 3D operators are tensor products of the reduced 1D operators and
// never need to be materialized: every apply is a stack of independent 1D line
// solves (the 3d_to_1d reduction), which is exactly the fast engine this codebase
// is built on.  The 1D SBP identity  Qx_1d + Qx_1d^T = Bx_1d  (verified in
// mms_cut1d) tensors up to  Qx_3d + Qx_3d^T = Sx_3d  automatically.

#ifndef CUT3D_H
#define CUT3D_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "linalg.h"
#include "bspline.h"
#include "cut1d.h"

// Reduced 1D cut operators for one axis (the building block tensored in 3D).
struct CutAxis1D {
    int n = 0;          // full 1D basis count
    int nr = 0;         // reduced count = number of interior modes
    int degree = 0;
    double left = 0.0;  // cut position (fluid = [left, 1])
    Mat E;              // n x nr extension
    Mat Hr, Qr, Kr, Br; // reduced 1D mass / derivative / stiffness / boundary forms
    std::vector<int> interior;
    std::vector<double> knots;
    double gamma_min = 1.0;
};

inline CutAxis1D build_cut_axis(int n_elements, int p, double left, int preserve_degree = -1) {
    CutForms1D cf = assemble_cut_1d(n_elements, p, left, 1.0);
    ActiveSets1D as = cut1d_active_sets(cf);
    Extension1D ext = cut1d_extension(cf, as, preserve_degree);
    CutAxis1D ax;
    ax.n = cf.n; ax.nr = (int)ext.interior.size(); ax.degree = p; ax.left = left;
    ax.E = ext.E; ax.interior = ext.interior; ax.knots = cf.knots;
    ax.Hr = cut1d_pullback(cf.H, ext.E);
    ax.Qr = cut1d_pullback(cf.Q, ext.E);
    ax.Kr = cut1d_pullback(cf.K, ext.E);
    ax.Br = cut1d_pullback(cf.boundary, ext.E);
    for (int i : as.active) ax.gamma_min = std::min(ax.gamma_min, as.gamma[i]);
    return ax;
}

// A 3D field stored flat, x-fastest: index(ix,iy,iz) = ix + nx*(iy + ny*iz).
struct Field3D {
    int nx = 0, ny = 0, nz = 0;
    std::vector<double> v;
    Field3D() {}
    Field3D(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_),
        v(static_cast<size_t>(nx_) * ny_ * nz_, 0.0) {}
    double& operator()(int ix, int iy, int iz) { return v[ix + nx*(static_cast<size_t>(iy) + ny*iz)]; }
    double  operator()(int ix, int iy, int iz) const { return v[ix + nx*(static_cast<size_t>(iy) + ny*iz)]; }
    size_t size() const { return v.size(); }
};

// Apply a 1D operator op (rows x cols) along one axis of a 3D field.  The output
// dimension along that axis becomes op.r (so this serves both reduce, op = E^T,
// and expand, op = E, as well as same-size operator applies).
inline Field3D apply_axis(const Field3D& in, int axis, const Mat& op) {
    Field3D out;
    if (axis == 0) {
        out = Field3D(op.r, in.ny, in.nz);
        for (int iz = 0; iz < in.nz; ++iz) for (int iy = 0; iy < in.ny; ++iy) {
            for (int i = 0; i < op.r; ++i) {
                double s = 0.0; const double* row = &op.a[static_cast<size_t>(i)*op.c];
                for (int j = 0; j < op.c; ++j) s += row[j] * in(j, iy, iz);
                out(i, iy, iz) = s;
            }
        }
    } else if (axis == 1) {
        out = Field3D(in.nx, op.r, in.nz);
        for (int iz = 0; iz < in.nz; ++iz) for (int ix = 0; ix < in.nx; ++ix) {
            for (int i = 0; i < op.r; ++i) {
                double s = 0.0; const double* row = &op.a[static_cast<size_t>(i)*op.c];
                for (int j = 0; j < op.c; ++j) s += row[j] * in(ix, j, iz);
                out(ix, i, iz) = s;
            }
        }
    } else {
        out = Field3D(in.nx, in.ny, op.r);
        for (int iy = 0; iy < in.ny; ++iy) for (int ix = 0; ix < in.nx; ++ix) {
            for (int i = 0; i < op.r; ++i) {
                double s = 0.0; const double* row = &op.a[static_cast<size_t>(i)*op.c];
                for (int j = 0; j < op.c; ++j) s += row[j] * in(ix, iy, j);
                out(ix, iy, i) = s;
            }
        }
    }
    return out;
}

// Solve a 1D SPD system A x = b for every line along one axis (A is dim x dim).
// Used to invert the tensor mass M = Hz (x) Hy (x) Hx one axis at a time.
inline Field3D solve_axis(const Field3D& in, int axis, const DenseMatrix& Afac_unused,
                          const Mat& A) {
    (void)Afac_unused;
    const int dim = A.r;
    DenseMatrix Ad = to_dense(A);
    Field3D out = in;
    auto solve_line = [&](std::vector<double>& line){ std::vector<double> x; dense_lu_solve(Ad, line, x); line = x; };
    if (axis == 0) {
        for (int iz = 0; iz < in.nz; ++iz) for (int iy = 0; iy < in.ny; ++iy) {
            std::vector<double> line(dim); for (int i=0;i<dim;++i) line[i]=in(i,iy,iz);
            solve_line(line); for (int i=0;i<dim;++i) out(i,iy,iz)=line[i];
        }
    } else if (axis == 1) {
        for (int iz = 0; iz < in.nz; ++iz) for (int ix = 0; ix < in.nx; ++ix) {
            std::vector<double> line(dim); for (int i=0;i<dim;++i) line[i]=in(ix,i,iz);
            solve_line(line); for (int i=0;i<dim;++i) out(ix,i,iz)=line[i];
        }
    } else {
        for (int iy = 0; iy < in.ny; ++iy) for (int ix = 0; ix < in.nx; ++ix) {
            std::vector<double> line(dim); for (int i=0;i<dim;++i) line[i]=in(ix,iy,i);
            solve_line(line); for (int i=0;i<dim;++i) out(ix,iy,i)=line[i];
        }
    }
    return out;
}

// Tensor reduce  full -> reduced  (apply Ex^T, Ey^T, Ez^T).
inline Field3D tensor_reduce(const Field3D& full, const CutAxis1D& ax, const CutAxis1D& ay, const CutAxis1D& az) {
    Field3D t = apply_axis(full, 0, transpose(ax.E));
    t = apply_axis(t, 1, transpose(ay.E));
    t = apply_axis(t, 2, transpose(az.E));
    return t;
}
// Tensor expand  reduced -> full  (apply Ex, Ey, Ez).
inline Field3D tensor_expand(const Field3D& red, const CutAxis1D& ax, const CutAxis1D& ay, const CutAxis1D& az) {
    Field3D t = apply_axis(red, 0, ax.E);
    t = apply_axis(t, 1, ay.E);
    t = apply_axis(t, 2, az.E);
    return t;
}
// Apply the reduced tensor mass M = Hrz (x) Hry (x) Hrx to a reduced field.
inline Field3D reduced_mass_apply(const Field3D& red, const CutAxis1D& ax, const CutAxis1D& ay, const CutAxis1D& az) {
    Field3D t = apply_axis(red, 0, ax.Hr);
    t = apply_axis(t, 1, ay.Hr);
    t = apply_axis(t, 2, az.Hr);
    return t;
}
// Solve the reduced tensor mass system M cr = br by three per-axis solves.
inline Field3D reduced_mass_solve(const Field3D& br, const CutAxis1D& ax, const CutAxis1D& ay, const CutAxis1D& az) {
    DenseMatrix dummy;
    Field3D t = solve_axis(br, 0, dummy, ax.Hr);
    t = solve_axis(t, 1, dummy, ay.Hr);
    t = solve_axis(t, 2, dummy, az.Hr);
    return t;
}

#endif // CUT3D_H
