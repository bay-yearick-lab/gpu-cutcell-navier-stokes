# Top-level entry points. Python targets use uv (https://docs.astral.sh/uv/).
#
#   make verify     recompute conditioning, conservation and convergence order (CPU, ~1-2 min)
#   make test       Python test suite (pytest)
#   make cpp        build the C++17 solver and apps into cpp/bin/
#   make cpp-test   C++ manufactured-solution (MMS) regression suite
#   make gpu        build the CUDA kernels (ARCH=sm_XX, default sm_120)
#   make gpu-test   run each CUDA kernel against its CPU reference
#   make assets     regenerate the README figures from the committed results

UV ?= uv

.PHONY: verify test cpp cpp-test gpu gpu-test assets clean
verify:
	$(UV) run python code/verify.py

test:
	$(UV) run pytest -q

cpp:
	$(MAKE) -C cpp -j

cpp-test:
	$(MAKE) -C cpp test

gpu:
	$(MAKE) -C cpp gpu

gpu-test:
	$(MAKE) -C cpp gpu-test

assets:
	$(UV) run python tools/make_readme_assets.py

clean:
	$(MAKE) -C cpp clean
