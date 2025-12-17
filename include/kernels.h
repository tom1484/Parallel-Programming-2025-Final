#ifndef DSMC_KERNELS_H
#define DSMC_KERNELS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "config.h"
#include "data_types.h"

// Helper for atomic adds on floats (required for sampling)
__device__ void atomicAddFloat(float* address, float val);

// Reset sampling accumulators to zero before each timestep
__global__ void reset_sampling_kernel(CellSystem c_sys);

// Compute final macroscopic quantities from accumulated sums
__global__ void finalize_sampling_kernel(CellSystem c_sys, SimParams params);

// Main physics kernel - processes one cell per block
__global__ void solve_cell_kernel(ParticleSystem p_sys, CellSystem c_sys, SimParams params);

#endif  // DSMC_KERNELS_H