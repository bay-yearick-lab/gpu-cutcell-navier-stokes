# CUDA kernels for the device-resident periodic-hill time step and the periodic
# B-spline apply. Requires the CUDA toolkit 12.8 or newer (nvcc, cuBLAS, cuFFT).
#
#   make                 build every kernel into ../bin/
#   make ARCH=sm_89      build for another GPU architecture (default sm_120)
#   make gpu-test        run each kernel against its CPU reference; fail on any
#                        MISMATCH/FAIL line or any relative error above TOL
#   make clean
#
# CUDA_HOME defaults to the directory above nvcc on PATH.

SHELL     := /bin/bash
NVCC      ?= nvcc
CUDA_HOME ?= $(abspath $(dir $(shell command -v $(NVCC) 2>/dev/null))/..)
ARCH      ?= sm_120
CXX       ?= g++
TOL       ?= 1e-8

INCLUDES  := -I../include/core -I../include/operators -I../include/solvers \
             -I../include/cut -I../include/io -I.
NVFLAGS   := -O3 -std=c++17 -arch=$(ARCH) -Xcompiler "-O3 -fopenmp" \
             -I$(CUDA_HOME)/include -I$(CUDA_HOME)/targets/x86_64-linux/include \
             $(INCLUDES)
NVLIBS    := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/targets/x86_64-linux/lib -lcudart -lcublas -lcufft -lgomp
CXXFLAGS  := -std=c++17 -O3 -march=native -fopenmp $(INCLUDES)

BIN       := ../bin
HEADERS   := $(wildcard ../include/*/*.h) stats_hill.h

# Kernel-by-kernel checks against the CPU implementation (each prints rel error).
BRICKS    := gpu_apply2d gpu_sample gpu_load gpu_fastdiag gpu_grad gpu_advect \
             gpu_dft gpu_solve gpu_skewfull gpu_step dns_gpu
# Multi-step comparison and production DNS drivers.
DRIVERS   := gpu_run gpu_dns gpu_dns_lowmem
# Host-only checkpoint tools.
HOSTTOOLS := ckpt_analyze dump_ckpt_stats

KERNELS   := $(BRICKS) $(DRIVERS)

.PHONY: all gpu gpu-test clean
all gpu: $(addprefix $(BIN)/,$(KERNELS) $(HOSTTOOLS))

$(BIN)/%: %.cu $(HEADERS) | $(BIN)
	$(NVCC) $(NVFLAGS) $< -o $@ $(NVLIBS)

$(BIN)/%: %.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN):
	@mkdir -p $(BIN)

# Each brick is diff-tested against the CPU reference; gpu_run advances 12 steps
# on the device and on the host and compares the final states.
gpu-test: $(addprefix $(BIN)/,$(BRICKS) gpu_run)
	@set -eo pipefail; for b in $(BRICKS); do \
	  echo "==== $$b ===="; $(BIN)/$$b | ./check_gpu.sh $(TOL); done; \
	echo "==== gpu_run (8 elements, 12 steps) ===="; $(BIN)/gpu_run 8 12 1 | ./check_gpu.sh $(TOL); \
	echo "all GPU kernels match the CPU reference (rel < $(TOL))"

clean:
	rm -f $(addprefix $(BIN)/,$(KERNELS) $(HOSTTOOLS))
