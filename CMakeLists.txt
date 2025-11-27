cmake_minimum_required(VERSION 3.14)
project(puzzle86 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress the "RIPEMD160 is deprecated" warning from OpenSSL 3.0
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
add_compile_definitions(OPENSSL_SUPPRESS_DEPRECATED)

# 1. Find Dependencies
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# 2. Find GMP Manually
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY NAMES gmp mpir)

# 3. Find secp256k1 Manually (The critical fix)
# We look specifically for the main lib AND the precomputed lib
find_path(SECP256K1_INCLUDE_DIR secp256k1.h)
find_library(SECP256K1_LIB NAMES secp256k1 unofficial-secp256k1)
find_library(SECP256K1_PRECOMP_LIB NAMES secp256k1_precomputed unofficial-secp256k1_precomputed)

if(NOT GMP_LIBRARY)
    message(FATAL_ERROR "GMP library not found via vcpkg.")
endif()
if(NOT SECP256K1_LIB)
    message(FATAL_ERROR "secp256k1 library not found via vcpkg.")
endif()

add_executable(puzzle86 src/main.cpp src/utils.cpp)

# --- LINKING ---

# Link OpenSSL and Threads
target_link_libraries(puzzle86 PRIVATE OpenSSL::Crypto Threads::Threads)

# Link GMP
target_include_directories(puzzle86 PRIVATE ${GMP_INCLUDE_DIR})
target_link_libraries(puzzle86 PRIVATE ${GMP_LIBRARY})

# Link secp256k1
target_include_directories(puzzle86 PRIVATE ${SECP256K1_INCLUDE_DIR})
target_link_libraries(puzzle86 PRIVATE ${SECP256K1_LIB})

# Link the precomputed table if found (This fixes your LNK2001 errors)
if(SECP256K1_PRECOMP_LIB)
    message(STATUS "Found Precomputed Tables: ${SECP256K1_PRECOMP_LIB}")
    target_link_libraries(puzzle86 PRIVATE ${SECP256K1_PRECOMP_LIB})
else()
    message(WARNING "Could not find secp256k1_precomputed.lib. Linker errors might occur.")
endif()

option(USE_CUDA "Enable CUDA kernels" ON)
if(USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set_target_properties(puzzle86 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_sources(puzzle86 PRIVATE src/gpu_hash.cu src/gpu_ec.cu)
    target_link_libraries(puzzle86 PRIVATE CUDA::cudart)
    target_compile_definitions(puzzle86 PRIVATE HAVE_CUDA=1)
  else()
    message(FATAL_ERROR "CUDA Toolkit not found. Install NVIDIA CUDA Toolkit and re-run CMake, or set USE_CUDA=OFF to build CPU-only.")
  endif()
else()
  target_compile_definitions(puzzle86 PRIVATE HAVE_CUDA=0)
endif()
