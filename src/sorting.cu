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
        // Atomic increment is efficient here due to random access patterns
        // causing collisions mostly only within the same warp/block
        atomicAdd(&d_counts[cell], 1);
    }
}

// Kernel 3: Scatter - Move particles to their new sorted locations
// Uses a "running offset" array (d_write_offsets) to determine where to write
__global__ void reorder_particles_kernel(ParticleSystem sys, int* d_write_offsets, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // 1. Identify which cell this particle belongs to
    int cell = sys.d_cell_id[idx];

    // 2. Get the unique write index for this particle
    // atomicAdd returns the OLD value, effectively giving us a slot index
    // and incrementing the counter for the next thread.
    int write_idx = atomicAdd(&d_write_offsets[cell], 1);

    // 3. Move data from Unsorted (Input) to Sorted (Output) arrays
    sys.d_pos_sorted[write_idx] = sys.d_pos[idx];
    sys.d_vel_sorted[write_idx] = sys.d_vel[idx];
    sys.d_species_sorted[write_idx] = sys.d_species[idx];

    // Note: We don't necessarily need to sort d_cell_id itself,
    // because the cell logic implies the index. But if needed for debugging:
    // sys.d_cell_id_sorted[write_idx] = cell;
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

    // Count particles
    count_particles_kernel<<<p_blocks, threads>>>(p_sys.d_cell_id, c_sys.d_cell_particle_count, num_particles);
    CHECK_CUDA(cudaGetLastError());

    // ----------------------------------------------------------
    // Step 2: Prefix Sum (Scan) to calculate Offsets
    // ----------------------------------------------------------

    // NOTE: In a production environment (like CUB or Thrust), this would be
    // a device-side scan. For pure CUDA without external libraries,
    // a Host-side round-trip is the most robust implementation for a prototype.

    // A. Download counts to Host
    std::vector<int> h_counts(num_cells);
    CHECK_CUDA(
        cudaMemcpy(h_counts.data(), c_sys.d_cell_particle_count, num_cells * sizeof(int), cudaMemcpyDeviceToHost));

    // B. Perform Exclusive Scan on Host
    std::vector<int> h_offsets(num_cells);
    int sum = 0;
    for (int i = 0; i < num_cells; i++) {
        h_offsets[i] = sum;
        sum += h_counts[i];
    }

    // C. Upload offsets to Device (to d_cell_offset)
    // These are the "Start Indices" required by the Physics Kernel
    CHECK_CUDA(cudaMemcpy(c_sys.d_cell_offset, h_offsets.data(), num_cells * sizeof(int), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // Step 3: Reorder (Scatter) Particles
    // ----------------------------------------------------------

    // We need a mutable copy of offsets for the reorder kernel to use as "write heads".
    // We cannot use d_cell_offset directly because atomicAdd would destroy the
    // start indices needed for the physics kernel.

    int* d_write_offsets;
    CHECK_CUDA(cudaMalloc(&d_write_offsets, num_cells * sizeof(int)));

    // Initialize d_write_offsets with the clean offsets we just calculated
    CHECK_CUDA(cudaMemcpy(d_write_offsets, c_sys.d_cell_offset, num_cells * sizeof(int), cudaMemcpyDeviceToDevice));

    // Scatter particles to d_pos_sorted, d_vel_sorted, etc.
    reorder_particles_kernel<<<p_blocks, threads>>>(p_sys, d_write_offsets, num_particles);
    CHECK_CUDA(cudaGetLastError());

    // Clean up temporary buffer
    CHECK_CUDA(cudaFree(d_write_offsets));

    // Synchronization ensures sorting is done before physics starts
    CHECK_CUDA(cudaDeviceSynchronize());
}