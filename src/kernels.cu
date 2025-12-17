#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "config.h"
#include "data_types.h"
#include "kernels.h"

// Helper for atomic adds on floats (required for sampling)
__device__ void atomicAddFloat(float* address, float val) {
    int* address_as_ull = (int*)address;
    int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __float_as_int(__int_as_float(assumed) + val));
    } while (assumed != old);
}

__global__ void solve_cell_kernel(ParticleSystem p_sys, CellSystem c_sys, SimParams params) {
    // 1. Block/Thread Identity
    // Each block calculates ONE cell independently [cite: 62]
    int cell_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (cell_idx >= c_sys.total_cells) return;

    // Extract parameters
    float dt = params.dt;
    float dx = params.cell_dx;
    float dy = params.cell_dy;
    float domain_lx = params.domain_lx;
    float domain_ly = params.domain_ly;
    int grid_nx = params.grid_nx;
    int grid_ny = params.grid_ny;

    // --- Shared Memory Allocation [cite: 93] ---
    // Declaring arrays in shared memory to hold particle data for this cell
    __shared__ PositionType s_pos[MAX_PARTICLES_PER_CELL];
    __shared__ VelocityType s_vel[MAX_PARTICLES_PER_CELL];
    __shared__ int s_species[MAX_PARTICLES_PER_CELL];
    __shared__ int s_subcell[MAX_PARTICLES_PER_CELL];

    // Helper to count active particles in this cell
    __shared__ int s_num_particles;

    // Retrieve pre-calculated offset for this cell from the global sorted list
    int cell_start_idx = c_sys.d_cell_offset[cell_idx];
    int cell_count = c_sys.d_cell_particle_count[cell_idx];

    if (tid == 0) s_num_particles = cell_count;
    __syncthreads();

    // --- Step 1: Copy to Shared Memory [cite: 110] ---
    // Parallel copy using all threads. Coalesced because global input is sorted.
    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        int global_idx = cell_start_idx + i;
        s_pos[i] = p_sys.d_pos[global_idx];
        s_vel[i] = p_sys.d_vel[global_idx];
        s_species[i] = p_sys.d_species[global_idx];
    }
    __syncthreads();

    // --- Step 2: Sub-cell Indexing [cite: 113] ---
    // Index particles into sub-cells for collision selection
    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        // Simple logic: calculate sub-cell based on position within cell
        // (Pseudocode geometry logic)
        int sub_idx = 0;  // calculate_sub_cell(s_pos[i]);
        s_subcell[i] = sub_idx;
    }
    __syncthreads();

    // --- Step 3: Collision (NTC Method) [cite: 119] ---
    // Each thread handles ONE sub-cell or a specific collision pair
    // Paper: "Each thread performs collision calculation for one sub-cell" [cite: 121]
    if (tid < MAX_SUB_CELLS) {
        // 1. Calculate collision pairs (NTC)
        // 2. Select candidates from shared memory (s_subcell array)
        // 3. Perform collision (GSS/VHS model)
        // 4. Update s_vel[candidate_idx] directly in shared memory
    }
    __syncthreads();

    // --- Step 4: Sampling [cite: 122] ---
    // Accumulate macroscopic moments
    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        // Atomic add to global cell properties (or local shared reduction first)
        // atomicAddFloat(&c_sys.d_density[cell_idx], mass);
    }
    __syncthreads();

    // --- Step 7: Chemistry [cite: 133] ---
    // (Skipped State & Free-stream for brevity, logic is similar)
    // Threads pick particles to check for dissociation/recombination
    __syncthreads();

    // --- Step 8: Movement, Wall, Locating [cite: 137] ---
    // Each thread moves specific particles.
    // Critical: Uses Double Precision for position [cite: 173]
    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        PositionType p = s_pos[i];
        VelocityType v = s_vel[i];

        // 1. Move
        p.x += v.x * dt;
        p.y += v.y * dt;

        // 2. Wall Interaction (CLL Model) [cite: 201]
        // Reflective boundaries - keep particles inside domain
        if (p.x < 0) { p.x = -p.x; v.x = -v.x; }
        if (p.x >= domain_lx) { p.x = 2.0 * domain_lx - p.x; v.x = -v.x; }
        if (p.y < 0) { p.y = -p.y; v.y = -v.y; }
        if (p.y >= domain_ly) { p.y = 2.0 * domain_ly - p.y; v.y = -v.y; }

        // 3. Locate new cell based on updated position
        int cx = (int)(p.x / dx);
        int cy = (int)(p.y / dy);
        // Clamp to valid cell range
        cx = max(0, min(cx, grid_nx - 1));
        cy = max(0, min(cy, grid_ny - 1));
        int new_cell = cy * grid_nx + cx;

        int global_idx = cell_start_idx + i;

        // Write locally updated state
        s_pos[i] = p;
        s_vel[i] = v;

        // Write new cell ID to global (for sorting in next step)
        p_sys.d_cell_id[global_idx] = new_cell;
    }
    __syncthreads();

    // --- Step 9: Copy back to Global Memory [cite: 139] ---
    // Note: We write back to the "unsorted" slot derived from the start index.
    // The sorting pass will move these to their correct new density blocks later.
    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        int global_idx = cell_start_idx + i;
        p_sys.d_pos[global_idx] = s_pos[i];
        p_sys.d_vel[global_idx] = s_vel[i];
        p_sys.d_species[global_idx] = s_species[i];
    }
}