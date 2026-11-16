cmake_minimum_required(VERSION 3.16)
project(haru_cpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4 /arch:AVX2)
else()
    add_compile_options(-O3 -Wall -Wextra -mavx2 -mfma)
endif()

# Include directories
include_directories(include)

# Optional ggml integration
find_package(ggml QUIET)
if(ggml_FOUND)
    message(STATUS "Found installed ggml library.")
    add_definitions(-DHARU_HAS_GGML)
else()
    message(STATUS "ggml not found in system paths. Building with built-in lightweight SIMD tensor backend.")
endif()

# Core library
add_library(haru_core STATIC
    src/haru_ggml.cpp
)

target_include_directories(haru_core PUBLIC include)

# CLI / Test Runner executable
add_executable(haru_runner
    src/main.cpp
)

target_link_libraries(haru_runner PRIVATE haru_core)
