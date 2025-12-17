#include "simulation.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "utils.cuh"

using namespace std;

// --- Allocation ---

void allocate_system(ParticleSystem& p_sys, CellSystem& c_sys, const SimConfig& cfg) {
    // Calculate totals
    c_sys.total_cells = cfg.grid_nx * cfg.grid_ny;

    // Estimate total particles based on density and volume
    // (In practice, allocate extra buffer for inflow)
    double volume = (cfg.domain_lx * cfg.domain_ly);
    int est_particles = (int)((cfg.init_density * volume) / cfg.particle_weight);
    int buffer_size = est_particles * 1.5;  // 50% buffer for fluctuation

    p_sys.total_particles = est_particles;

    // --- GPU Allocations (Particle System) ---
    // Note: We use Double Precision for Position [cite: 173]
    CHECK_CUDA(cudaMalloc(&p_sys.d_pos, buffer_size * sizeof(PositionType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_vel, buffer_size * sizeof(VelocityType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_species, buffer_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_cell_id, buffer_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_sub_id, buffer_size * sizeof(int)));

    // Sorted Arrays (Double Buffering)
    CHECK_CUDA(cudaMalloc(&p_sys.d_pos_sorted, buffer_size * sizeof(PositionType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_vel_sorted, buffer_size * sizeof(VelocityType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_species_sorted, buffer_size * sizeof(int)));

    // --- GPU Allocations (Cell System) ---
    CHECK_CUDA(cudaMalloc(&c_sys.d_density, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_temperature, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_cell_particle_count, c_sys.total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_cell_offset, c_sys.total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_write_offsets, c_sys.total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_inactive_write_idx, sizeof(int)));  // Single int for inactive particle write index

    // Velocity accumulator arrays for macroscopic sampling
    CHECK_CUDA(cudaMalloc(&c_sys.d_vel_sum_x, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_vel_sum_y, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_vel_sum_z, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_vel_sq_sum, c_sys.total_cells * sizeof(float)));

    // Pre-allocate CUB temp storage (query size first)
    c_sys.d_temp_storage = nullptr;
    c_sys.temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(c_sys.d_temp_storage, c_sys.temp_storage_bytes, c_sys.d_cell_particle_count,
                                  c_sys.d_cell_offset, c_sys.total_cells);
    CHECK_CUDA(cudaMalloc(&c_sys.d_temp_storage, c_sys.temp_storage_bytes));

    // Allocate segment array for solid objects
    CHECK_CUDA(cudaMalloc(&c_sys.d_segments, c_sys.total_cells * sizeof(Segment)));

    printf("Allocated System: %d cells, capacity for %d particles.\n", c_sys.total_cells, buffer_size);
}

// --- Initialization ---

// Check if a point is inside a solid object based on segment normal
// Returns true if the point is on the "inside" (opposite to normal) side of the segment
static bool is_inside_segment(double px, double py, const Segment& seg) {
    if (!seg.exists) return false;
    
    // Vector from segment start to point
    float dx = (float)px - seg.start_x;
    float dy = (float)py - seg.start_y;
    
    // Dot product with outward normal
    // If negative, point is on the inside (opposite to normal direction)
    float dot = dx * seg.normal_x + dy * seg.normal_y;
    
    return dot < 0.0f;
}

void init_simulation(ParticleSystem& p_sys, const CellSystem& c_sys, const SimConfig& cfg) {
    // Copy segment data from GPU to check for inside cells
    vector<Segment> h_segments(c_sys.total_cells);
    CHECK_CUDA(cudaMemcpy(h_segments.data(), c_sys.d_segments,
                          c_sys.total_cells * sizeof(Segment), cudaMemcpyDeviceToHost));

    // Count inside cells and segment cells for info
    int inside_count = 0;
    int segment_count = 0;
    for (int i = 0; i < c_sys.total_cells; i++) {
        if (h_segments[i].inside) inside_count++;
        if (h_segments[i].exists) segment_count++;
    }
    if (inside_count > 0 || segment_count > 0) {
        printf("Initialization: Avoiding %d inside cells and checking %d segment cells\n", 
               inside_count, segment_count);
    }

    vector<PositionType> h_pos(p_sys.total_particles);
    vector<VelocityType> h_vel(p_sys.total_particles);
    vector<int> h_cell_id(p_sys.total_particles);

    mt19937 gen(1234);
    uniform_real_distribution<double> dist_x(0.0, cfg.domain_lx);
    uniform_real_distribution<double> dist_y(0.0, cfg.domain_ly);
    normal_distribution<float> dist_v(0.0f, 300.0f);  // Approx thermal velocity

    float dx = cfg.domain_lx / cfg.grid_nx;
    float dy = cfg.domain_ly / cfg.grid_ny;

    for (int i = 0; i < p_sys.total_particles; i++) {
        // Generate random position, rejecting positions inside solid objects
        int cell_id;
        double px, py;
        int attempts = 0;
        const int max_attempts = 1000;
        bool is_valid;

        do {
            px = dist_x(gen);
            py = dist_y(gen);
            int cx = (int)(px / dx);
            int cy = (int)(py / dy);
            // Clamp to valid range
            cx = max(0, min(cx, cfg.grid_nx - 1));
            cy = max(0, min(cy, cfg.grid_ny - 1));
            cell_id = cy * cfg.grid_nx + cx;
            
            // Check if position is valid:
            // 1. Not in a cell marked as completely inside
            // 2. Not on the inside of a segment (if cell has one)
            is_valid = !h_segments[cell_id].inside && 
                       !is_inside_segment(px, py, h_segments[cell_id]);
            
            attempts++;
        } while (!is_valid && attempts < max_attempts);

        if (attempts >= max_attempts) {
            // Fallback: place at domain corner (should not happen in practice)
            px = 0.0;
            py = 0.0;
            cell_id = 0;
        }

        h_pos[i] = make_double2(px, py);

        // Maxwellian Velocity
        h_vel[i] = make_float3(dist_v(gen), dist_v(gen), 0.0f);

        // Store cell ID
        h_cell_id[i] = cell_id;
    }

    // Copy to GPU
    CHECK_CUDA(
        cudaMemcpy(p_sys.d_pos, h_pos.data(), p_sys.total_particles * sizeof(PositionType), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(p_sys.d_vel, h_vel.data(), p_sys.total_particles * sizeof(VelocityType), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(p_sys.d_cell_id, h_cell_id.data(), p_sys.total_particles * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize species to 0 (single species simulation)
    CHECK_CUDA(cudaMemset(p_sys.d_species, 0, p_sys.total_particles * sizeof(int)));
}

// --- Cleanup ---

void free_system(ParticleSystem& p_sys, CellSystem& c_sys) {
    // Free particle system
    cudaFree(p_sys.d_pos);
    cudaFree(p_sys.d_pos_sorted);
    cudaFree(p_sys.d_vel);
    cudaFree(p_sys.d_vel_sorted);
    cudaFree(p_sys.d_species);
    cudaFree(p_sys.d_species_sorted);
    cudaFree(p_sys.d_cell_id);
    cudaFree(p_sys.d_sub_id);

    // Free cell system
    cudaFree(c_sys.d_density);
    cudaFree(c_sys.d_temperature);
    cudaFree(c_sys.d_vel_sum_x);
    cudaFree(c_sys.d_vel_sum_y);
    cudaFree(c_sys.d_vel_sum_z);
    cudaFree(c_sys.d_vel_sq_sum);
    cudaFree(c_sys.d_cell_particle_count);
    cudaFree(c_sys.d_cell_offset);
    cudaFree(c_sys.d_write_offsets);
    cudaFree(c_sys.d_inactive_write_idx);
    cudaFree(c_sys.d_temp_storage);
    cudaFree(c_sys.d_segments);
}
