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

// ============================================================================
// Reset sampling accumulators to zero before each timestep
// ============================================================================
__global__ void reset_sampling_kernel(CellSystem c_sys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c_sys.total_cells) return;

    c_sys.d_density[idx] = 0.0f;
    c_sys.d_temperature[idx] = 0.0f;
    c_sys.d_vel_sum_x[idx] = 0.0f;
    c_sys.d_vel_sum_y[idx] = 0.0f;
    c_sys.d_vel_sq_sum[idx] = 0.0f;
}

// ============================================================================
// Finalize sampling: convert accumulated sums to physical quantities
// Called after solve_cell_kernel to compute density and temperature
// ============================================================================
__global__ void finalize_sampling_kernel(CellSystem c_sys, SimParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c_sys.total_cells) return;

    int n_particles = c_sys.d_cell_particle_count[idx];

    if (n_particles == 0) {
        c_sys.d_density[idx] = 0.0f;
        c_sys.d_temperature[idx] = 0.0f;
        return;
    }

    // Number density: n = (N_sim * Fnum) / V_cell
    // where N_sim = simulator particles, Fnum = particle weight, V_cell = cell volume
    float n_density = (float)n_particles * params.particle_weight / params.cell_volume;
    c_sys.d_density[idx] = n_density;

    // Temperature from kinetic theory:
    // T = (m / 2k_B) * <c^2> where c = v - <v> is the peculiar velocity (2D)
    // <c^2> = <v^2> - <v>^2
    // For 2D: T = m/(2*k_B) * (<vx^2 + vy^2> - <vx>^2 - <vy>^2)

    float inv_n = 1.0f / (float)n_particles;

    // Mean velocities
    float mean_vx = c_sys.d_vel_sum_x[idx] * inv_n;
    float mean_vy = c_sys.d_vel_sum_y[idx] * inv_n;

    // Mean of velocity squared
    float mean_v_sq = c_sys.d_vel_sq_sum[idx] * inv_n;

    // Peculiar velocity squared: <c^2> = <v^2> - |<v>|^2
    float mean_c_sq = mean_v_sq - (mean_vx * mean_vx + mean_vy * mean_vy);

    // Prevent negative values due to floating point errors
    mean_c_sq = fmaxf(mean_c_sq, 0.0f);

    // Temperature: T = m * <c^2> / (2 * k_B) for 2D
    // Boltzmann constant k_B = 1.380649e-23 J/K
    const float k_B = 1.380649e-23f;
    float temperature = params.particle_mass * mean_c_sq / (2.0f * k_B);

    c_sys.d_temperature[idx] = temperature;
}

// Check if a particle trajectory crosses a line segment
// Returns true if intersection occurs, and sets t to the parametric intersection point
__device__ bool segment_intersection(double p0x, double p0y,  // Old position
                                     double p1x, double p1y,  // New position
                                     float sx, float sy,      // Segment start
                                     float ex, float ey,      // Segment end
                                     float& t                 // Output: parametric t along particle trajectory [0,1]
) {
    // Direction of particle motion
    double dx = p1x - p0x;
    double dy = p1y - p0y;

    // Direction of segment
    float segx = ex - sx;
    float segy = ey - sy;

    // Cross product for denominator
    double denom = dx * segy - dy * segx;

    // Parallel check
    if (fabs(denom) < 1e-10) return false;

    // Vector from segment start to particle start
    double qpx = p0x - sx;
    double qpy = p0y - sy;

    // Parametric values
    t = (float)((segx * qpy - segy * qpx) / denom);    // Along particle trajectory
    float u = (float)((dx * qpy - dy * qpx) / denom);  // Along segment

    // Check if intersection is within both line segments
    return (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f);
}

// Reflect particle off a segment (specular reflection)
__device__ void reflect_particle(PositionType& p,     // Position (will be updated)
                                 VelocityType& v,     // Velocity (will be updated)
                                 const Segment& seg,  // Segment to reflect off
                                 float t              // Intersection parameter
) {
    // Normal vector
    float nx = seg.normal_x;
    float ny = seg.normal_y;

    // Reflect velocity: v' = v - 2(v·n)n
    float vdotn = v.x * nx + v.y * ny;
    v.x = v.x - 2.0f * vdotn * nx;
    v.y = v.y - 2.0f * vdotn * ny;

    // Calculate intersection point
    // We need the old position to compute this, but we only have new position
    // Since p was already updated, we need to work backwards
    // p_intersect = p_old + t * (p_new - p_old)
    // p_new = p_old + v * dt, so p_old = p_new - displacement
    // For simplicity, just flip the remaining distance past the wall

    // Reflect position: mirror across the segment
    // Distance from current position to segment (signed)
    float px = (float)p.x;
    float py = (float)p.y;

    // Vector from segment start to particle
    float dx = px - seg.start_x;
    float dy = py - seg.start_y;

    // Distance along normal (signed - positive if on normal side)
    float dist = dx * nx + dy * ny;

    // If particle is on wrong side of segment, reflect it
    if (dist < 0) {
        p.x = px - 2.0 * dist * nx;
        p.y = py - 2.0 * dist * ny;
    }
}

__global__ void solve_cell_kernel(ParticleSystem p_sys, CellSystem c_sys, SimParams params) {
    // 1. Block/Thread Identity
    // Each block calculates ONE cell independently [cite: 62]
    int cell_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (cell_idx >= c_sys.total_cells) return;

    // Skip cells that are entirely inside solid objects
    // (particles should not exist in these cells)
    if (c_sys.d_segments[cell_idx].inside) return;

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
    // TODO: Calculate Nsc based on the number of particles
    // Index particles into sub-cells for collision selection
    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        // Simple logic: calculate sub-cell based on position within cell
        // TODO: Implement actual sub-cell calculation
        int sub_idx = 0;  // calculate_sub_cell(s_pos[i]);
        s_subcell[i] = sub_idx;
    }
    __syncthreads();

    // --- Step 3: Collision (NTC Method) [cite: 119] ---
    // Each thread handles ONE sub-cell or a specific collision pair
    // Paper: "Each thread performs collision calculation for one sub-cell" [cite: 121]
    if (tid < MAX_SUB_CELLS) {
        // TODO: Implement collision logic here
        // 1. Calculate collision pairs (NTC)
        // 2. Select candidates from shared memory (s_subcell array)
        // 3. Perform collision (GSS/VHS model)
        // 4. Update s_vel[candidate_idx] directly in shared memory
    }
    __syncthreads();

    // --- Step 4: Sampling [cite: 122] ---
    // Accumulate velocity moments for macroscopic properties
    // Use shared memory reduction to minimize atomic operations
    __shared__ float s_vel_sum_x;
    __shared__ float s_vel_sum_y;
    __shared__ float s_vel_sq_sum;

    if (tid == 0) {
        s_vel_sum_x = 0.0f;
        s_vel_sum_y = 0.0f;
        s_vel_sq_sum = 0.0f;
    }
    __syncthreads();

    // Each thread accumulates its portion
    float local_vx_sum = 0.0f;
    float local_vy_sum = 0.0f;
    float local_vsq_sum = 0.0f;

    for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
        VelocityType v = s_vel[i];
        local_vx_sum += v.x;
        local_vy_sum += v.y;
        local_vsq_sum += v.x * v.x + v.y * v.y;
    }

    // Atomic add to shared memory accumulators
    atomicAdd(&s_vel_sum_x, local_vx_sum);
    atomicAdd(&s_vel_sum_y, local_vy_sum);
    atomicAdd(&s_vel_sq_sum, local_vsq_sum);
    __syncthreads();

    // Thread 0 writes final sums to global memory
    if (tid == 0) {
        c_sys.d_vel_sum_x[cell_idx] = s_vel_sum_x;
        c_sys.d_vel_sum_y[cell_idx] = s_vel_sum_y;
        c_sys.d_vel_sq_sum[cell_idx] = s_vel_sq_sum;
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

        // Store old position for segment intersection test
        double old_x = p.x;
        double old_y = p.y;

        // 1. Move
        p.x += v.x * dt;
        p.y += v.y * dt;

        // 2. Segment collision check (solid objects)
        Segment seg = c_sys.d_segments[cell_idx];
        if (seg.exists) {
            float t;
            if (segment_intersection(old_x, old_y, p.x, p.y, seg.start_x, seg.start_y, seg.end_x, seg.end_y, t)) {
                reflect_particle(p, v, seg, t);
            }
        }

        // 3. Domain boundary interaction (reflective walls)
        if (p.x < 0) {
            p.x = -p.x;
            v.x = -v.x;
        }
        if (p.x >= domain_lx) {
            p.x = 2.0 * domain_lx - p.x;
            v.x = -v.x;
        }
        if (p.y < 0) {
            p.y = -p.y;
            v.y = -v.y;
        }
        if (p.y >= domain_ly) {
            p.y = 2.0 * domain_ly - p.y;
            v.y = -v.y;
        }

        // 4. Locate new cell based on updated position
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