# CUDA kernels

Fourteen `.cu` files that move the periodic-hill time step onto the GPU, plus a
periodic B-spline apply with a cuFFT pressure projection.

| File | Content | Check against the CPU |
| --- | --- | --- |
| `gpu_apply2d.cu` | dense 2D operator apply per spanwise slice, cuBLAS DGEMM | rel error vs CPU matmul |
| `gpu_sample.cu` | sparse sample on the curved-cut hill: CSR gather (x-y) + dense z | rel error vs `hill_sample` |
| `gpu_load.cu` | Galerkin load (adjoint of sample): dense z-integrate + `atomicAdd` scatter | rel error vs `hill_load` |
| `gpu_fastdiag.cu` | per-Fourier-mode dense solves, `cublasDgemmStridedBatched` | rel error vs per-mode loop |
| `gpu_grad.cu` | sampled gradient in x, y, z reusing the sample kernels | rel error vs `hill_grad` |
| `gpu_advect.cu` | skew-symmetric convection assembled on the device | rel error vs CPU (pre mass solve) |
| `gpu_dft.cu` | spanwise DFT and inverse, mode-major layout | forward DFT and round trip |
| `gpu_solve.cu` | fast-diagonalization implicit solve (viscous + Schur pressure) | rel error vs `FastDiagHill::solve` |
| `gpu_skewfull.cu` | complete skew convection including the mass solve | rel error vs `hill_skew_conv` |
| `gpu_step.cu` | one full device-resident SSP-RK3 step | rel error vs `hill_fastdiag_step` |
| `gpu_run.cu` | K device steps vs K host steps, ms per step | OK / MISMATCH |
| `gpu_dns.cu` | periodic-hill DNS driver: body force, statistics, checkpoints | production run |
| `gpu_dns_lowmem.cu` | `gpu_dns.cu` with aliased advection buffers (smaller footprint) | production run |
| `dns_gpu.cu` | periodic-cube B-spline apply, one thread per 1D line, cuFFT projection | PASS / FAIL per operator |

Host tools: `ckpt_analyze.cpp` (fluctuation energy fraction and CFL of a
checkpoint), `dump_ckpt_stats.cpp` (statistics JSON from a running checkpoint),
`stats_hill.h` (online Reynolds-stress statistics at the ten hill stations).

## Toolchain

CUDA 12.8 or newer (sm_120 needs 12.8), cuBLAS and cuFFT, and a host C++17
compiler with OpenMP. Any of these works:

- the NVIDIA CUDA toolkit 12.8+ installed system-wide (`CUDA_HOME=/usr/local/cuda`);
- conda-forge: `conda install -c conda-forge cuda-nvcc=12.9 libcublas-dev libcufft-dev cuda-cudart-dev`, then `CUDA_HOME=$CONDA_PREFIX`.

## Build and test

```bash
make -C cpp gpu                      # all kernels into cpp/bin/, -arch=sm_120
make -C cpp gpu ARCH=sm_89           # another architecture
make -C cpp gpu-test                 # every kernel vs its CPU reference
```

`gpu-test` pipes each kernel's output through `check_gpu.sh`, which fails on any
`MISMATCH` or `FAIL` line, any CUDA error, or any printed relative error above
`TOL` (default `1e-8`; the kernels report round-off, about 1e-15).

## Periodic-hill run

`results/hill_re5600_gpu.json` holds the statistics of a 12,000-step GPU run at
$Re_H = 5600$, p = 3, 24 x 24 elements and 16 spanwise modes (simulated time 60,
1,300 samples), written by `gpu_dns` through `stats_hill.h`. Its arguments are
`ne nz nu fx0 target_bulk nsteps print_every avg_start stats_path [ic_amp dt
ckpt_path ckpt_every]`.
