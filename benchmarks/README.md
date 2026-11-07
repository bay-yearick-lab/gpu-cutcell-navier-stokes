# Benchmarks

## GPU sparse Krylov solve

`gpu_pcg_benchmark_3d.py` assembles the screened-Poisson proxy $A = K + H$ on the
aggregated hollow-cube cut geometry (3D, p = 2, sliver parameter 1e-3), then
solves $A x = b$ three ways: Jacobi-preconditioned CG on the CPU (SciPy CSR,
single thread), the same Jacobi-PCG on the GPU (CuPy CSR, tolerance 1e-8 on both),
and a SuperLU factorization plus solve on the CPU. Hardware: NVIDIA GeForce RTX
5060 Ti (16 GB). A warm-up solve runs before timing so kernel compilation is not
charged; host-to-device transfer is timed separately.

```bash
uv sync --group gpu
uv run python benchmarks/gpu_pcg_benchmark_3d.py           # 20, 28, 36, 44 elements per axis
uv run python benchmarks/gpu_pcg_benchmark_3d.py --quick   # 20 only
```

Results (`results/gpu_solver_benchmark_3d.csv`):

| Elements/axis | Unknowns | nnz | CSR memory | PCG iterations CPU / GPU | CPU PCG [s] | GPU PCG [s] | GPU speedup | SuperLU factor [s] | SuperLU solve [s] |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 9,832 | 1.14M | 13.7 MB | 104 / 104 | 0.218 | 0.024 | 9.0x | 2.0 | 0.026 |
| 28 | 23,200 | 3.40M | 40.9 MB | 156 / 156 | 1.212 | 0.046 | 26.1x | 13.5 | 0.100 |
| 36 | 47,480 | 6.69M | 80.5 MB | 178 / 177 | 2.764 | 0.070 | 39.6x | 59.1 | 0.308 |
| 44 | 81,344 | 11.43M | 137.5 MB | 278 / 278 | 5.165 | 0.173 | 29.8x | 279.0 | 0.634 |

CSR memory is float64 values plus int32 column indices and row pointers. Every
solve converges to a relative residual below 1e-8 on both backends, and the
transfer of the matrix and right-hand side to the device takes 2 to 67 ms.

## Small-cell stress test (CPU)

`code/fast_solver_workbench.py` sweeps the cut-cell volume ratio from 10 to
$10^5$ on aggregated p = 4 systems on a 2D oblique cut and solves each with
Jacobi-PCG (tolerance 1e-8). Results in `code/results/fast_solver_stress.csv`.

```bash
uv run python code/fast_solver_workbench.py --production
```

| Elements/axis | Unknowns | nnz | Volume ratios | Iterations | Max relative residual | Max solve time [ms] |
| ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| 8 | 120 | 7,396 | 10 to 1e5 | 104 to 107 | 9.7e-9 | 1.1 |
| 12 | 224 | 15,564 | 10 to 1e5 | 128 to 133 | 9.9e-9 | 2.7 |
| 16 | 360 | 26,162 | 10 to 1e5 | 140 to 145 | 9.8e-9 | 3.7 |
| 24 | 728 | 55,170 | 10 to 1e5 | 161 to 171 | 9.7e-9 | 4.4 |
| 32 | 1,224 | 94,546 | 10 to 1e5 | 181 to 188 | 9.6e-9 | 7.2 |

At a fixed mesh the iteration count changes by at most 10 over four decades of
volume ratio; it grows with the mesh, as for a body-fitted system.
