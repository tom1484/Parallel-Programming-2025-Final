#ifndef DSMC_SORTING_H
#define DSMC_SORTING_H

#include <cuda_runtime.h>

#include "config.h"
#include "data_types.h"
#include "utils.cuh"

// Kernel 1: Reset the particle counts for all cells to zero
__global__ void reset_counts_kernel(int* d_counts, int num_cells);
// Kernel 2: Histogram - Count how many particles are currently in each cell
__global__ void count_particles_kernel(const int* __restrict__ d_cell_id, int* __restrict__ d_counts,
                                       int num_particles);
// Kernel 3: Scatter - Move particles to their new sorted locations
// Uses a "running offset" array (d_write_offsets) to determine where to write
__global__ void reorder_particles_kernel(ParticleSystem sys, int* d_write_offsets, int num_particles);

// ------------------------------------------------------------------
// Host Function: The Sorting Pipeline
// ------------------------------------------------------------------
void sort_particles(ParticleSystem& p_sys, CellSystem& c_sys);

#endif  // DSMC_SORTING_H