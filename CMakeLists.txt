cmake_minimum_required(VERSION 3.18)
project(LunarGasParticle VERSION 1.0 LANGUAGES CXX CUDA)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # gnu++17

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust based on your GPU
# Common values: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (RTX 30xx), 89 (RTX 40xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

# Find CUDA package (optional, as CUDA language is enabled above)
find_package(CUDAToolkit REQUIRED)

# Find yaml-cpp package
find_package(yaml-cpp REQUIRED)

# Add executable
add_executable(dsmc_solver
    src/main.cu
    src/sim_config.cu
    src/simulation.cu
    src/kernels.cu
    src/sorting.cu
    src/visualize.cu
    src/geometry.cu
    src/source.cu
)

target_include_directories(dsmc_solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link CUDA libraries
target_link_libraries(dsmc_solver PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
    yaml-cpp::yaml-cpp
)

# Set C++ compiler flags
target_compile_options(dsmc_solver PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)

# Set CUDA compiler flags
set_target_properties(dsmc_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CUDA specific compile options
target_compile_options(dsmc_solver PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Optional: Set debug flags when CMAKE_BUILD_TYPE is Debug
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(dsmc_solver PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-g>
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # CUDA debug info
        $<$<COMPILE_LANGUAGE:CUDA>:-g>
    )
endif()

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA found: ${CUDAToolkit_FOUND}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
