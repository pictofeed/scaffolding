# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Helix Technologies, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Convenience Makefile — wraps both setuptools and CMake builds.
#

.PHONY: all build build-cmake install clean test help

PYTHON     ?= python3
CMAKE      ?= cmake
BUILD_DIR  ?= build
BUILD_TYPE ?= Release

help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Scaffolding Build System                                ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║  make build         – setuptools + Cython (in-place)     ║"
	@echo "║  make build-cmake   – CMake build                        ║"
	@echo "║  make install       – pip editable install               ║"
	@echo "║  make wheel         – build wheel distribution           ║"
	@echo "║  make clean         – remove build artefacts             ║"
	@echo "║  make test          – run test suite                     ║"
	@echo "╚══════════════════════════════════════════════════════════╝"

# ── Setuptools build (recommended for development) ──
all: build

build:
	$(PYTHON) setup.py build_ext --inplace

# ── Editable pip install ──
install:
	pip install -e ".[dev]"

# ── Wheel ──
wheel:
	$(PYTHON) -m build --wheel

# ── CMake build ──
build-cmake:
	@mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DPython3_EXECUTABLE=$$(which $(PYTHON))
	$(CMAKE) --build $(BUILD_DIR) --parallel

# ── Tests ──
test: build
	$(PYTHON) -m pytest tests/ -v

# ── Clean ──
clean:
	rm -rf $(BUILD_DIR) dist *.egg-info
	rm -f _tensor_ops.c _tensor_ops*.so _tensor_ops*.html
	rm -f _mps_ops.c _mps_ops*.so _mps_ops*.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
