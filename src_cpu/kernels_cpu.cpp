#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "kernels_cpu.h"

// ============================================================================
// Reset sampling accumulators to zero before each timestep
// ============================================================================
void reset_sampling_cpu(CellSystemCPU& c_sys) {
    #pragma omp parallel for
    for (int idx = 0; idx < c_sys.total_cells; idx++) {
        c_sys.density[idx] = 0.0f;
        c_sys.temperature[idx] = 0.0f;
        c_sys.vel_sum_x[idx] = 0.0f;
        c_sys.vel_sum_y[idx] = 0.0f;
        c_sys.vel_sum_z[idx] = 0.0f;
        c_sys.vel_sq_sum[idx] = 0.0f;
    }
}

// ============================================================================
// Finalize sampling: convert accumulated sums to physical quantities
// ============================================================================
void finalize_sampling_cpu(CellSystemCPU& c_sys, const SimParams& params) {
    #pragma omp parallel for
    for (int idx = 0; idx < c_sys.total_cells; idx++) {
        int n_particles = c_sys.cell_particle_count[idx];

        if (n_particles == 0) {
            c_sys.density[idx] = 0.0f;
            c_sys.temperature[idx] = 0.0f;
            continue;
        }

        // Number density
        float n_density = (float)n_particles * params.particle_weight / params.cell_volume;
        c_sys.density[idx] = n_density;

        // Temperature calculation
        float inv_n = 1.0f / (float)n_particles;
        float mean_vx = c_sys.vel_sum_x[idx] * inv_n;
        float mean_vy = c_sys.vel_sum_y[idx] * inv_n;
        float mean_vz = c_sys.vel_sum_z[idx] * inv_n;
        float mean_v_sq = c_sys.vel_sq_sum[idx] * inv_n;

        float mean_c_sq = mean_v_sq - (mean_vx * mean_vx + mean_vy * mean_vy + mean_vz * mean_vz);
        mean_c_sq = std::max(mean_c_sq, 0.0f);

        const float k_B = 1.380649e-23f;
        float temperature = params.particle_mass * mean_c_sq / (3.0f * k_B);
        c_sys.temperature[idx] = temperature;
    }
}

// ============================================================================
// Helper: Check segment intersection
// ============================================================================
static bool segment_intersection(double p0x, double p0y, double p1x, double p1y,
                                float sx, float sy, float ex, float ey, float& t) {
    double dx = p1x - p0x;
    double dy = p1y - p0y;
    float segx = ex - sx;
    float segy = ey - sy;
    double denom = dx * segy - dy * segx;

    if (std::abs(denom) < 1e-12) return false;

    double t_line = ((sx - p0x) * segy - (sy - p0y) * segx) / denom;
    if (t_line < 0.0 || t_line > 1.0) return false;

    double t_seg = ((sx - p0x) * dy - (sy - p0y) * dx) / denom;
    if (t_seg < 0.0 || t_seg > 1.0) return false;

    t = (float)t_line;
    return true;
}

// ============================================================================
// Helper: Reflect particle off segment
// ============================================================================
static void reflect_particle(PositionType& p, VelocityType& v, const Segment& seg, float t) {
    float nx = seg.normal_x;
    float ny = seg.normal_y;

    // Velocity reflection
    float v_dot_n = v.x * nx + v.y * ny;
    v.x -= 2.0f * v_dot_n * nx;
    v.y -= 2.0f * v_dot_n * ny;

    // Position correction
    double px = p.x;
    double py = p.y;
    float dist = (float)((px - seg.start_x) * nx + (py - seg.start_y) * ny);
    
    if (dist < 0) {
        p.x = px - 2.0 * dist * nx;
        p.y = py - 2.0 * dist * ny;
    }
}

// ============================================================================
// Main physics kernel - CPU version with OpenMP
// Processes all cells in parallel, each cell independently
// ============================================================================
void solve_cell_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys, const SimParams& params) {
    // Process each cell independently (mimics GPU block-per-cell pattern)
    #pragma omp parallel for schedule(dynamic)
    for (int cell_idx = 0; cell_idx < c_sys.total_cells; cell_idx++) {
        // Skip cells inside solid objects
        if (c_sys.segments[cell_idx].inside) continue;

        // Get particles for this cell
        int cell_start_idx = c_sys.cell_offset[cell_idx];
        int cell_count = c_sys.cell_particle_count[cell_idx];
        
        if (cell_count == 0) continue;

        // Extract parameters
        float dt = params.dt;
        float dx = params.cell_dx;
        float dy = params.cell_dy;
        float domain_lx = params.domain_lx;
        float domain_ly = params.domain_ly;
        int grid_nx = params.grid_nx;
        int grid_ny = params.grid_ny;

        // Local accumulators for sampling
        float local_vel_sum_x = 0.0f;
        float local_vel_sum_y = 0.0f;
        float local_vel_sum_z = 0.0f;
        float local_vel_sq_sum = 0.0f;

        // Process particles in this cell
        for (int i = 0; i < cell_count; i++) {
            int global_idx = cell_start_idx + i;
            
            // Read particle data
            PositionType p = p_sys.pos[global_idx];
            VelocityType v = p_sys.vel[global_idx];

            // Sampling - accumulate velocity moments
            local_vel_sum_x += v.x;
            local_vel_sum_y += v.y;
            local_vel_sum_z += v.z;
            local_vel_sq_sum += v.x * v.x + v.y * v.y + v.z * v.z;

            // Store old position for collision detection
            double old_x = p.x;
            double old_y = p.y;

            // Movement
            p.x += v.x * dt;
            p.y += v.y * dt;

            // Segment collision (solid objects)
            const Segment& seg = c_sys.segments[cell_idx];
            if (seg.exists) {
                float t;
                if (segment_intersection(old_x, old_y, p.x, p.y,
                                       seg.start_x, seg.start_y, seg.end_x, seg.end_y, t)) {
                    reflect_particle(p, v, seg, t);
                }
            }

            // Domain boundary reflections
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

            // Locate new cell
            int cx = (int)(p.x / dx);
            int cy = (int)(p.y / dy);
            cx = std::max(0, std::min(cx, grid_nx - 1));
            cy = std::max(0, std::min(cy, grid_ny - 1));
            int new_cell = cy * grid_nx + cx;

            // Write back updated particle data
            p_sys.pos[global_idx] = p;
            p_sys.vel[global_idx] = v;
            p_sys.cell_id[global_idx] = new_cell;
        }

        // Write accumulated sampling data
        c_sys.vel_sum_x[cell_idx] = local_vel_sum_x;
        c_sys.vel_sum_y[cell_idx] = local_vel_sum_y;
        c_sys.vel_sum_z[cell_idx] = local_vel_sum_z;
        c_sys.vel_sq_sum[cell_idx] = local_vel_sq_sum;
    }
}
