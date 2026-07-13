# Makefile for the C++17 B-spline cut-cell / mimetic Navier-Stokes solver.
# Header-only, no external numerical libraries, OpenMP across independent 1D
# line solves. Requires g++ (or clang++) with C++17 and OpenMP.
#
#   make            compile every program into bin/
#   make test       build + run the MMS correctness suite (the regression gate)
#   make run        build + run every program
#   make gpu        build the CUDA kernels in gpu/ (nvcc; ARCH=sm_XX, default sm_120)
#   make gpu-test   run every GPU kernel against its CPU reference, fail on mismatch
#   make clean      remove bin/ and generated artifacts

# --- Toolchain -------------------------------------------------------------
#   -O3 -march=native   target the host CPU's vector extensions
#   -funroll-loops      unroll the short banded inner loops
#   -fno-exceptions -fno-rtti   the "C-style C++" hot-path discipline
#   -fopenmp            parallelize across independent 1D line solves
CXX      ?= g++
CXXFLAGS := -std=c++17 -O3 -march=native -funroll-loops -Wall -Wextra \
            -fno-exceptions -fno-rtti -fopenmp

# Header library lives under include/ (one -I per layer) plus the acoustics
# app-local header. Flat quote-includes resolve through these paths.
INCLUDES := -Iinclude/core -Iinclude/operators -Iinclude/solvers -Iinclude/io \
            -Iinclude/cut -Iapps/acoustics

# Every header is a prerequisite of every program: editing any header rebuilds
# all programs that could use it (cheap insurance against stale objects).
HEADERS  := $(wildcard include/*/*.h) $(wildcard apps/*/*.h)

# --- Program inventory (grouped by role) -----------------------------------
TESTS         := mms_check mms_ns mms_bmf mms_poisson mms_compressible mms_burgers \
                 mms_wall2d mms_infsup mms_wallns2d mms_wallns3d mms_tgdecay \
                 mms_cut1d mms_cut2d mms_cut3d mms_comp_cut3d mms_hill2d mms_hill3d \
                 mms_hill_metric
APPS_INCOMP   := ns_taylor_green ns_dns3d ns_reconnection dns_movie wall_vortex2d \
                 wall_dns3d channel_dns ns_channel_heat ns_channel_project ns_channel_stokes ns_bmf_channel
APPS_COMP     := comp_tgv comp_channel comp_kh
APPS_ACOUSTIC := acoustic2d acoustic_vortex caa_dispersion
BENCH         := bench bench3d fair_compare operator_probe bspline_basis

PROGRAMS := $(TESTS) $(APPS_INCOMP) $(APPS_COMP) $(APPS_ACOUSTIC) $(BENCH)
BINS     := $(addprefix bin/,$(PROGRAMS))

# Where the .cpp sources live (vpath lets the pattern rule find them by name).
vpath %.cpp tests apps/incompressible apps/compressible apps/acoustics apps/benchmarks

DATA_DIR    := data

.PHONY: all test run gpu gpu-test clean
all: $(BINS)

# Pattern rule: build bin/<name> from <name>.cpp (found via vpath).
bin/%: %.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

bin:
	@mkdir -p bin

# Regression gate: the MMS correctness suite must stay green. Fail-fast: a
# nonzero test exit fails the gate (the old loop swallowed exit codes).
test: $(addprefix bin/,$(TESTS))
	@mkdir -p $(DATA_DIR)
	@for p in $(TESTS); do echo "==== $$p ===="; ./bin/$$p || exit 1; echo; done

# Build and run everything.
run: $(BINS)
	@mkdir -p $(DATA_DIR)
	@for p in $(PROGRAMS); do echo "==== $$p ===="; ./bin/$$p; echo; done

# CUDA kernels (see gpu/Makefile and gpu/README.md).
gpu:
	$(MAKE) -C gpu

gpu-test:
	$(MAKE) -C gpu gpu-test

clean:
	rm -rf bin $(DATA_DIR)/turb $(DATA_DIR)/ctgv $(DATA_DIR)/kh $(DATA_DIR)/wv2d $(DATA_DIR)/wdns $(DATA_DIR)/chan
	$(MAKE) -C gpu clean
