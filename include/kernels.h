#ifndef DSMC_KERNELS_H
#define DSMC_KERNELS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "config.h"
#include "data_types.h"

// Helper for atomic adds on floats (required for sampling)
__device__ void atomicAddFloat(float* address, float val);
__global__ void solve_cell_kernel(ParticleSystem p_sys, CellSystem c_sys, float dt, int total_cells);

#endif  // DSMC_KERNELS_H