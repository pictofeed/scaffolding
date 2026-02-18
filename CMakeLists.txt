# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# CMake build system for Scaffolding.
#
# Builds Cython extensions (_tensor_ops, _mps_ops, _cuda_ops) and any
# future C/C++/CUDA components under a single unified build.
#
# Usage
# -----
#     mkdir build && cd build
#     cmake .. -DCMAKE_BUILD_TYPE=Release
#     cmake --build . --parallel
#     cmake --install . --prefix ../install
#
# Or use the convenience wrapper:
#     python -m cmake --build . --config Release
#
# Options
# -------
#   -DSCAFFOLDING_USE_MPS=ON/OFF    Force MPS/Accelerate (auto-detected)
#   -DSCAFFOLDING_USE_CUDA=ON/OFF   Force CUDA backend (auto-detected)
#   -DSCAFFOLDING_CUDA_ARCH="70;80;90" Override GPU architectures
#   -DSCAFFOLDING_ANNOTATE=ON       Generate Cython HTML annotations
#   -DPYTHON_EXECUTABLE=/path/...   Override Python interpreter
#
cmake_minimum_required(VERSION 3.18)

project(Scaffolding
    VERSION 0.1.0
    DESCRIPTION "Scaffolding is a lightweight, production-ready deep learning framework built from the ground up using NumPy as its computational backend and Cython-accelerated hot-path operations for maximum throughput. On macOS, Scaffolding leverages Apple's Accelerate framework (BLAS, vDSP, vecLib) through native Cython bindings for near-hardware-level performance on both CPU and Apple Silicon."
    LANGUAGES C CXX
)

# ── Global settings ──
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──
option(SCAFFOLDING_USE_MPS   "Build MPS/Accelerate backend (macOS only)" ON)
option(SCAFFOLDING_USE_CUDA  "Build CUDA GPU backend"                    ON)
option(SCAFFOLDING_ANNOTATE  "Generate Cython HTML annotation files"     OFF)
set(SCAFFOLDING_CUDA_ARCH "" CACHE STRING
    "Semicolon-separated list of CUDA architectures (e.g. 70;80;90)")

# ── Find Python + NumPy ──
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

message(STATUS "Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "NumPy include: ${Python3_NumPy_INCLUDE_DIRS}")

# ── Find Cython ──
find_program(CYTHON_EXECUTABLE
    NAMES cython cython3
    HINTS ${Python3_EXECUTABLE}/../
)
if(NOT CYTHON_EXECUTABLE)
    message(FATAL_ERROR "Cython not found. Install with: pip install cython>=3.0")
endif()
message(STATUS "Cython: ${CYTHON_EXECUTABLE}")

# ── Detect macOS / Accelerate ──
set(_IS_MACOS FALSE)
if(APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_IS_MACOS TRUE)
endif()

if(SCAFFOLDING_USE_MPS AND NOT _IS_MACOS)
    message(STATUS "MPS/Accelerate disabled (not macOS)")
    set(SCAFFOLDING_USE_MPS OFF)
endif()

if(SCAFFOLDING_USE_MPS)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(METAL_FRAMEWORK Metal)
    if(NOT ACCELERATE_FRAMEWORK)
        message(WARNING "Accelerate.framework not found — MPS disabled")
        set(SCAFFOLDING_USE_MPS OFF)
    else()
        message(STATUS "Accelerate: ${ACCELERATE_FRAMEWORK}")
    endif()
    if(METAL_FRAMEWORK)
        message(STATUS "Metal: ${METAL_FRAMEWORK}")
    endif()
endif()

# ── Detect CUDA ──
set(_CUDA_FOUND FALSE)
if(SCAFFOLDING_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(_CUDA_FOUND TRUE)
            message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} at ${CUDAToolkit_TARGET_DIR}")
        else()
            message(WARNING "CUDA compiler found but CUDAToolkit not detected — CUDA disabled")
        endif()
    else()
        message(STATUS "CUDA compiler not found — CUDA extension disabled")
    endif()
endif()

# Set CUDA architectures
if(_CUDA_FOUND)
    if(SCAFFOLDING_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${SCAFFOLDING_CUDA_ARCH})
    else()
        # Broad default: Volta through Hopper
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
        # Filter based on CUDA version
        if(CUDAToolkit_VERSION VERSION_LESS "11.0")
            set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 75)
        elseif(CUDAToolkit_VERSION VERSION_LESS "11.1")
            set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 75 80)
        elseif(CUDAToolkit_VERSION VERSION_LESS "11.8")
            set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 75 80 86)
        elseif(CUDAToolkit_VERSION VERSION_LESS "12.0")
            set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 75 80 86 89 90)
        else()
            # CUDA 12+: sm_35/37 deprecated, start from 50
            set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 75 80 86 89 90)
        endif()
    endif()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ── Cython compilation macro ──
# Runs cython to generate .c from .pyx, then builds a shared library
# that Python can import as a C extension module.
function(add_cython_extension target_name pyx_source)
    # Determine output .c file
    get_filename_component(_pyx_name ${pyx_source} NAME_WE)
    set(_c_file "${CMAKE_CURRENT_BINARY_DIR}/${_pyx_name}.c")

    # Cython flags
    set(_cython_flags -3 --fast-fail)
    if(SCAFFOLDING_ANNOTATE)
        list(APPEND _cython_flags -a)
    endif()

    # Run Cython
    add_custom_command(
        OUTPUT ${_c_file}
        COMMAND ${CYTHON_EXECUTABLE} ${_cython_flags}
                -o ${_c_file}
                ${CMAKE_CURRENT_SOURCE_DIR}/${pyx_source}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${pyx_source}
        COMMENT "Cythonizing ${pyx_source} → ${_pyx_name}.c"
        VERBATIM
    )

    # Build the extension module
    Python3_add_library(${target_name} MODULE ${_c_file})
    target_include_directories(${target_name} PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )
    target_compile_options(${target_name} PRIVATE
        -O3 -ffast-math
    )
    # Set output name to match import (no "lib" prefix)
    set_target_properties(${target_name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${_pyx_name}"
    )

    # Forward any extra arguments as link libraries
    foreach(_lib IN LISTS ARGN)
        target_link_libraries(${target_name} PRIVATE ${_lib})
    endforeach()
endfunction()


# ── Extension 1: _tensor_ops (generic CPU, macOS only) ──
# On Linux the pre-generated .c is not shipped and the runtime
# falls back to pure-NumPy / CUDA, so we only build on macOS.
if(_IS_MACOS)
    add_cython_extension(scaffolding_tensor_ops
        _tensor_ops.pyx
    )
    message(STATUS "_tensor_ops extension: ENABLED (macOS)")
else()
    message(STATUS "_tensor_ops extension: DISABLED (not macOS)")
endif()

# ── Extension 2: _mps_ops (macOS Accelerate) ──
if(SCAFFOLDING_USE_MPS)
    add_cython_extension(scaffolding_mps_ops
        _mps_ops.pyx
        ${ACCELERATE_FRAMEWORK}
    )
    target_compile_definitions(scaffolding_mps_ops PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
    )
    if(METAL_FRAMEWORK)
        target_link_libraries(scaffolding_mps_ops PRIVATE
            ${METAL_FRAMEWORK}
        )
    endif()
    message(STATUS "MPS/Accelerate extension: ENABLED")
else()
    message(STATUS "MPS/Accelerate extension: DISABLED")
endif()

# ── Extension 3: _cuda_ops (NVIDIA GPU) ──
if(_CUDA_FOUND)
    # 1. Build the CUDA kernels as a static library
    add_library(scaffolding_cuda_kernels STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/_cuda_kernels.cu
    )
    target_include_directories(scaffolding_cuda_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(scaffolding_cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    # Optimise for each arch
    target_compile_options(scaffolding_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG>
    )

    # 2. Cythonize + link the Cython binding against kernels + CUDA libs
    add_cython_extension(scaffolding_cuda_ops
        _cuda_ops.pyx
    )
    target_include_directories(scaffolding_cuda_ops PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(scaffolding_cuda_ops PRIVATE
        scaffolding_cuda_kernels
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    message(STATUS "CUDA extension: ENABLED")
else()
    message(STATUS "CUDA extension: DISABLED")
endif()


# ── Install targets ──
# Install compiled extensions into the scaffolding package directory.
# The DESTINATION should be the scaffolding package root so that
#   import scaffolding._tensor_ops
# works correctly.
set(_SCAFFOLDING_INSTALL_DIR
    "${Python3_SITEARCH}/scaffolding"
)

if(_IS_MACOS)
    install(TARGETS scaffolding_tensor_ops
        LIBRARY DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
        RUNTIME DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
    )
endif()

if(SCAFFOLDING_USE_MPS)
    install(TARGETS scaffolding_mps_ops
        LIBRARY DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
        RUNTIME DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
    )
endif()

if(_CUDA_FOUND)
    install(TARGETS scaffolding_cuda_ops
        LIBRARY DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
        RUNTIME DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
    )
endif()

# Install pure-Python sources
install(DIRECTORY scaffolding/
    DESTINATION ${_SCAFFOLDING_INSTALL_DIR}
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyx" EXCLUDE
)

# ── Summary ──
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║  Scaffolding ${PROJECT_VERSION} — Build Configuration              ║")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  Python:     ${Python3_VERSION}                                    ║")
message(STATUS "║  NumPy:      found                                      ║")
message(STATUS "║  Cython:     found                                      ║")
if(SCAFFOLDING_USE_MPS)
message(STATUS "║  MPS/Accel:  ENABLED                                    ║")
else()
message(STATUS "║  MPS/Accel:  disabled                                   ║")
endif()
if(_CUDA_FOUND)
message(STATUS "║  CUDA:       ENABLED (${CUDAToolkit_VERSION})                      ║")
else()
message(STATUS "║  CUDA:       disabled                                   ║")
endif()
