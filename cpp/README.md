# C++17 solver

Header-only C++17 with OpenMP and no external numerical libraries. Incompressible
and compressible Navier-Stokes on B-spline mimetic operators, the sharp-cut
kernels in 1D, 2D and 3D, and the periodic-hill curved-cut layer that the CUDA
kernels in `gpu/` port to the device. On periodic and box domains every
multidimensional operator reduces to independent 1D banded or circulant line
solves, so no 3D matrix is ever formed.

```bash
make -j          # every program into bin/  (CXX=clang++ also works)
make test        # 18 manufactured-solution (MMS) programs; stops at the first failure
make gpu         # CUDA kernels, see gpu/README.md
make gpu-test    # each CUDA kernel against its CPU reference
```

## Operators

The B-spline mimetic routes share one spline space of degree $p$:

| Route | Domain | First derivative | Second derivative |
| --- | --- | --- | --- |
| BMC (flux, collocated) | periodic | $D_1 = B' B^{-1}$ | $D_2 = B'' B^{-1}$ |
| BMG (Galerkin) | periodic | $D_1 = B M^{-1} K B^{-1}$ | $D_2 = -B M^{-1} S B^{-1}$ |
| BMF (staggered flux) | bounded | $D_1 = G = B_\eta D B^{-1}$ | $D_2 = D_v G$ |
| BMG (Galerkin) | bounded | $D_1 = B M^{-1} K B^{-1}$ | $D_2 = H^{-1}(-D^\top H D + dD)$ |

`include/operators/bm_operators.h` assembles them densely for verification;
`include/operators/bm_apply.h` applies the same maps through the
reduce-solve-expand chain (banded LU plus a low-rank cyclic correction for the
periodic wrap), which is what the solvers use.

## Layout

```
include/
  core/        bspline.h  linalg.h  config.h        basis, banded/cyclic LU, run configuration
  operators/   bm_operators.h  bm_apply.h  bm3d.h  bmf3d.h
  solvers/     ns.h  ns_channel.h  compressible.h  poisson.h  fastdiag3.h
               burgers1d.h  wall2d.h  wall_ns2d.h  wall_ns3d.h
  cut/         cut1d.h  cut2d.h  cut3d.h  comp_cut3d.h     sharp-cut kernels and aggregation
               hill_geom.h  hill_cut2d.h  hill3d.h  hill_fastdiag.h   periodic hill
  io/          vtk.h                                 ParaView .vti/.pvd output
apps/
  incompressible/   ns_taylor_green  ns_dns3d  dns_movie  ns_reconnection  channel_dns
                    ns_channel_*  ns_bmf_channel  wall_vortex2d  wall_dns3d
  compressible/     comp_tgv  comp_channel  comp_kh
  acoustics/        acoustic2d  acoustic_vortex  caa_dispersion
  benchmarks/       bench  bench3d  fair_compare  operator_probe  bspline_basis
tests/              mms_*.cpp   manufactured-solution and identity checks
gpu/                CUDA kernels, Makefile, results/
data/               MMS convergence tables and the regression log
```

## Verification

`make test` runs 18 programs. They check the structural identities (Green, SBP,
skew-symmetry) at machine precision, broadband MMS convergence per degree, the
sharp-cut kernels (`mms_cut1d`, `mms_cut2d`, `mms_cut3d`, `mms_comp_cut3d`), the
periodic-hill operators (`mms_hill2d`, `mms_hill3d`, `mms_hill_metric`), inf-sup
stability of the Galerkin velocity-pressure pairing (`mms_infsup`), and the walled
Navier-Stokes loops (`mms_wallns2d`, `mms_wallns3d`, `mms_tgdecay`). Selected
results from `data/mms_regression.txt`:

- cut 1D kernel: Green and SBP identities on the reduced space to 9.1e-15, polynomial reproduction to 3.3e-16;
- 3D walled box: Schur-complement projection idempotent, weak divergence 3.1e-2 before and 4.3e-17 after;
- periodic hill: skew-symmetric convection injects no interior kinetic energy (round-off);
- periodic compressible split form, 300 steps: mass drift 1.1e-14, momentum at most 1.2e-17, total energy 3.6e-14; relative convective kinetic-energy rate 5.3e-19.

The compressible core uses the split skew-symmetric convective form on the
periodic BMC derivative, which is skew-symmetric to round-off ($D^\top = -D$), so
the discrete scheme conserves mass, momentum and total energy, and kinetic energy
in the inviscid limit, with no added dissipation.

## Outputs

`bin/dns_movie [N Re T nframes]` runs a 3D vortex-breakdown DNS and writes
ParaView `.vti` frames (vorticity magnitude, Q-criterion) and a `.pvd` series;
`bin/comp_kh` and `bin/comp_tgv` do the same for the compressible solver.
