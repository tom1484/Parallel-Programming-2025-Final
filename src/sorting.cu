#include <cub/cub.cuh>

#include "sorting.h"

// Kernel 1: Reset the particle counts for all cells to zero
__global__ void reset_counts_kernel(int* d_counts, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        d_counts[idx] = 0;
    }
}

// Kernel 2: Histogram - Count how many particles are currently in each cell
__global__ void count_particles_kernel(const int* __restrict__ d_cell_id, int* __restrict__ d_counts,
                                       int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        int cell = d_cell_id[idx];
        // Skip inactive particles (cell_id == INACTIVE_CELL_ID)
        if (cell == INACTIVE_CELL_ID) return;
        // Atomic increment is efficient here due to random access patterns
        // causing collisions mostly only within the same warp/block
        atomicAdd(&d_counts[cell], 1);
    }
}

// Kernel 3: Scatter - Move particles to their new sorted locations
// Uses a "running offset" array (d_write_offsets) to determine where to write
__global__ void reorder_particles_kernel(ParticleSystem sys, int* d_write_offsets, int* d_inactive_write_idx,
                                         int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // 1. Identify which cell this particle belongs to
    int cell = sys.d_cell_id[idx];

    // 2. Handle inactive particles - write them to the end of the sorted array
    if (cell == INACTIVE_CELL_ID) {
        // Get write index for inactive particles (counting backwards from end)
        int write_idx = atomicAdd(d_inactive_write_idx, 1);
        sys.d_pos_sorted[write_idx] = sys.d_pos[idx];
        sys.d_vel_sorted[write_idx] = sys.d_vel[idx];
        sys.d_species_sorted[write_idx] = sys.d_species[idx];
        return;
    }

    // 3. Get the unique write index for this particle
    // atomicAdd returns the OLD value, effectively giving us a slot index
    // and incrementing the counter for the next thread.
    int write_idx = atomicAdd(&d_write_offsets[cell], 1);

    // 4. Move data from Unsorted (Input) to Sorted (Output) arrays
    sys.d_pos_sorted[write_idx] = sys.d_pos[idx];
    sys.d_vel_sorted[write_idx] = sys.d_vel[idx];
    sys.d_species_sorted[write_idx] = sys.d_species[idx];
}

// ------------------------------------------------------------------
// Host Function: The Sorting Pipeline
// ------------------------------------------------------------------

void sort_particles(ParticleSystem& p_sys, CellSystem& c_sys) {
    int num_particles = p_sys.total_particles;
    int num_cells = c_sys.total_cells;

    // Grid configuration for particle-based kernels
    int threads = 256;
    int p_blocks = (num_particles + threads - 1) / threads;

    // Grid configuration for cell-based kernels
    int c_blocks = (num_cells + threads - 1) / threads;

    // ----------------------------------------------------------
    // Step 1: Count Particles per Cell (Histogram)
    // ----------------------------------------------------------

    // Reset counts to 0
    reset_counts_kernel<<<c_blocks, threads>>>(c_sys.d_cell_particle_count, num_cells);
    CHECK_CUDA(cudaGetLastError());

    // Count particles (skips inactive particles with cell_id == INACTIVE_CELL_ID)
    count_particles_kernel<<<p_blocks, threads>>>(p_sys.d_cell_id, c_sys.d_cell_particle_count, num_particles);
    CHECK_CUDA(cudaGetLastError());

    // ----------------------------------------------------------
    // Step 2: Prefix Sum (Scan) to calculate Offsets
    // ----------------------------------------------------------

    // Use CUB for device-side exclusive prefix sum with pre-allocated temp storage
    cub::DeviceScan::ExclusiveSum(c_sys.d_temp_storage, c_sys.temp_storage_bytes, c_sys.d_cell_particle_count,
                                  c_sys.d_cell_offset, num_cells);

    // ----------------------------------------------------------
    // Step 3: Reorder (Scatter) Particles
    // ----------------------------------------------------------

    // We need a mutable copy of offsets for the reorder kernel to use as "write heads".
    // We cannot use d_cell_offset directly because atomicAdd would destroy the
    // start indices needed for the physics kernel.
    // Use pre-allocated d_write_offsets buffer.

    // Initialize d_write_offsets with the clean offsets we just calculated
    CHECK_CUDA(
        cudaMemcpy(c_sys.d_write_offsets, c_sys.d_cell_offset, num_cells * sizeof(int), cudaMemcpyDeviceToDevice));

    // Calculate where inactive particles should start writing
    // They go after all active particles: offset[last_cell] + count[last_cell]
    // We can compute this as total_active = sum of all counts
    // For simplicity, we'll initialize inactive write index to the sum of prefix + last count
    // which equals the total number of active particles
    int total_active;
    int last_offset, last_count;
    CHECK_CUDA(cudaMemcpy(&last_offset, &c_sys.d_cell_offset[num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(&last_count, &c_sys.d_cell_particle_count[num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost));
    total_active = last_offset + last_count;

    // Set the inactive write index to start after all active particles
    CHECK_CUDA(cudaMemcpy(c_sys.d_inactive_write_idx, &total_active, sizeof(int), cudaMemcpyHostToDevice));

    // Scatter particles to d_pos_sorted, d_vel_sorted, etc.
    reorder_particles_kernel<<<p_blocks, threads>>>(p_sys, c_sys.d_write_offsets, c_sys.d_inactive_write_idx,
                                                    num_particles);
    CHECK_CUDA(cudaGetLastError());

    // Synchronization ensures sorting is done before physics starts
    CHECK_CUDA(cudaDeviceSynchronize());
}