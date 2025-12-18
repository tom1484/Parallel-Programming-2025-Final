cmake_minimum_required(VERSION 3.10)
project(DSMC_CPU VERSION 1.0 LANGUAGES CXX)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Add executable
add_executable(dsmc_solver_cpu
    main_cpu.cpp
    simulation_cpu.cpp
    kernels_cpu.cpp
    sorting_cpu.cpp
    source_cpu.cpp
    visualize_cpu.cpp
)

# Include directories
target_include_directories(dsmc_solver_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link libraries
target_link_libraries(dsmc_solver_cpu PRIVATE
    OpenMP::OpenMP_CXX
    yaml-cpp::yaml-cpp
)

# Compiler flags
target_compile_options(dsmc_solver_cpu PRIVATE
    -O3
    -march=native
    -fPIC
)

# Debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(dsmc_solver_cpu PRIVATE -g)
endif()

# Print configuration
message(STATUS "CPU Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP found: ${OpenMP_FOUND}")
