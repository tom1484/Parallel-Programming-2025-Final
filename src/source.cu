#include "source.h"
#include "utils.cuh"
#include "config.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

// Boltzmann constant
#define K_BOLTZMANN 1.380649e-23f

// ============================================================================
// Device Kernels
// ============================================================================

__global__ void init_rng_kernel(curandState* states, int num_states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    // Initialize each state with unique sequence
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void emit_particles_kernel(
    ParticleSystem p_sys,
    int first_particle_idx,     // Starting index for this source's particles
    int emit_offset,            // Offset within source's allocation (particles already generated)
    int num_to_emit,            // Number of particles to emit this timestep
    float start_x, float start_y,
    float end_x, float end_y,
    float dir_x, float dir_y,
    float bulk_velocity,
    float temperature,
    float particle_mass,
    float cell_dx, float cell_dy,
    int grid_nx, int grid_ny,
    curandState* rng_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_emit) return;
    
    // Global particle index
    int particle_idx = first_particle_idx + emit_offset + idx;
    
    // Load RNG state
    curandState local_state = rng_states[idx];
    
    // ========================================================================
    // Position: Uniform random along emission segment
    // ========================================================================
    float t = curand_uniform(&local_state);  // [0, 1)
    float pos_x = start_x + t * (end_x - start_x);
    float pos_y = start_y + t * (end_y - start_y);
    
    // ========================================================================
    // Velocity: Maxwellian flux distribution
    // ========================================================================
    // Thermal velocity scale: v_th = sqrt(k_B * T / m)
    float v_th = sqrtf(K_BOLTZMANN * temperature / particle_mass);
    
    // Tangent direction (perpendicular to emission direction)
    float tan_x = -dir_y;
    float tan_y = dir_x;
    
    // Normal component (into domain): Rayleigh distribution for flux
    // v_n = sqrt(-2 * v_th^2 * ln(R)) where R is uniform (0,1]
    // This gives positive velocities biased toward higher speeds (flux distribution)
    float u1 = curand_uniform(&local_state);
    // Avoid log(0) by clamping
    u1 = fmaxf(u1, 1e-10f);
    float v_normal = sqrtf(-2.0f * v_th * v_th * logf(u1));
    
    // Add bulk velocity
    v_normal += bulk_velocity;
    
    // Tangential component: Standard Maxwellian (normal distribution)
    // Use Box-Muller transform
    float u2 = curand_uniform(&local_state);
    float u3 = curand_uniform(&local_state);
    u2 = fmaxf(u2, 1e-10f);
    float v_tangent = v_th * sqrtf(-2.0f * logf(u2)) * cosf(2.0f * M_PI * u3);
    
    // Z component: Standard Maxwellian
    float u4 = curand_uniform(&local_state);
    float u5 = curand_uniform(&local_state);
    u4 = fmaxf(u4, 1e-10f);
    float v_z = v_th * sqrtf(-2.0f * logf(u4)) * cosf(2.0f * M_PI * u5);
    
    // Convert to Cartesian velocities
    float vel_x = v_normal * dir_x + v_tangent * tan_x;
    float vel_y = v_normal * dir_y + v_tangent * tan_y;
    float vel_z = v_z;
    
    // ========================================================================
    // Calculate cell ID
    // ========================================================================
    int cx = (int)(pos_x / cell_dx);
    int cy = (int)(pos_y / cell_dy);
    cx = max(0, min(cx, grid_nx - 1));
    cy = max(0, min(cy, grid_ny - 1));
    int cell_id = cy * grid_nx + cx;
    
    // ========================================================================
    // Write particle data (activate the particle)
    // ========================================================================
    p_sys.d_pos[particle_idx] = make_double2((double)pos_x, (double)pos_y);
    p_sys.d_vel[particle_idx] = make_float3(vel_x, vel_y, vel_z);
    p_sys.d_species[particle_idx] = 0;  // Single species
    p_sys.d_cell_id[particle_idx] = cell_id;  // Activate by setting valid cell ID
    
    // Save RNG state
    rng_states[idx] = local_state;
}

// ============================================================================
// Host Functions
// ============================================================================

bool load_source(const std::string& path, ParticleSource& source) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open source file: %s\n", path.c_str());
        return false;
    }
    
    std::string line;
    std::vector<int> timesteps;
    std::vector<int> counts;
    
    // Initialize with defaults
    source.start_x = 0; source.start_y = 0;
    source.end_x = 0; source.end_y = 0;
    source.dir_x = 1; source.dir_y = 0;
    source.bulk_velocity = 0;
    source.temperature = 300;
    source.total_particles = 0;
    
    float thermal_vel = 300.0f;
    float stream_vel_x = 0, stream_vel_y = 0, stream_vel_z = 0;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        // Parse key-value pairs
        if (key == "total_particles") {
            iss >> source.total_particles;
        } else if (key == "start_x") {
            iss >> source.start_x;
        } else if (key == "start_y") {
            iss >> source.start_y;
        } else if (key == "end_x") {
            iss >> source.end_x;
        } else if (key == "end_y") {
            iss >> source.end_y;
        } else if (key == "dir_x") {
            iss >> source.dir_x;
        } else if (key == "dir_y") {
            iss >> source.dir_y;
        } else if (key == "thermal_vel") {
            iss >> thermal_vel;
        } else if (key == "stream_vel_x") {
            iss >> stream_vel_x;
        } else if (key == "stream_vel_y") {
            iss >> stream_vel_y;
        } else if (key == "stream_vel_z") {
            iss >> stream_vel_z;
        } else if (key == "temperature") {
            iss >> source.temperature;
        } else if (key == "bulk_velocity") {
            iss >> source.bulk_velocity;
        } else {
            // Try to parse as schedule entry: timestep count
            // Key might be a number (timestep)
            try {
                int ts = std::stoi(key);
                int count;
                iss >> count;
                timesteps.push_back(ts);
                counts.push_back(count);
            } catch (...) {
                // Unknown key, skip
                fprintf(stderr, "Warning: Unknown key in source file: %s\n", key.c_str());
            }
        }
    }
    file.close();
    
    // Compute bulk_velocity from stream velocity if not explicitly set
    if (source.bulk_velocity == 0) {
        // Project stream velocity onto emission direction
        source.bulk_velocity = stream_vel_x * source.dir_x + stream_vel_y * source.dir_y;
    }
    
    // Convert thermal_vel to temperature if temperature wasn't set
    // thermal_vel = sqrt(k_B * T / m), so T = m * thermal_vel^2 / k_B
    // For Argon: m = 6.6335e-26 kg, k_B = 1.380649e-23
    if (source.temperature == 300 && thermal_vel != 300.0f) {
        float mass = 6.6335e-26f;
        source.temperature = mass * thermal_vel * thermal_vel / K_BOLTZMANN;
    }
    
    // Normalize direction
    float len = sqrtf(source.dir_x * source.dir_x + source.dir_y * source.dir_y);
    if (len > 0) {
        source.dir_x /= len;
        source.dir_y /= len;
    }
    
    if (timesteps.empty()) {
        fprintf(stderr, "Error: No schedule entries in source file: %s\n", path.c_str());
        return false;
    }
    
    // Verify total matches schedule sum
    int schedule_sum = 0;
    for (int c : counts) schedule_sum += c;
    if (source.total_particles == 0) {
        source.total_particles = schedule_sum;
    } else if (schedule_sum != source.total_particles) {
        fprintf(stderr, "Warning: Source %s: header says %d particles, schedule sums to %d\n",
                path.c_str(), source.total_particles, schedule_sum);
        // Use the schedule sum as the actual total
        source.total_particles = schedule_sum;
    }
    
    // Upload schedule to device
    source.schedule_size = (int)timesteps.size();
    CHECK_CUDA(cudaMalloc(&source.d_schedule_timesteps, source.schedule_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&source.d_schedule_counts, source.schedule_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(source.d_schedule_timesteps, timesteps.data(), 
                          source.schedule_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(source.d_schedule_counts, counts.data(),
                          source.schedule_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize tracking
    source.current_schedule_idx = 0;
    source.particles_generated = 0;
    source.first_particle_idx = 0;  // Set later by setup_source_rng
    
    printf("Loaded source from %s: %d total particles, %d schedule entries\n",
           path.c_str(), source.total_particles, source.schedule_size);
    printf("  Segment: (%.4f, %.4f) -> (%.4f, %.4f)\n",
           source.start_x, source.start_y, source.end_x, source.end_y);
    printf("  Direction: (%.4f, %.4f), bulk_vel=%.1f m/s, temp=%.1f K\n",
           source.dir_x, source.dir_y, source.bulk_velocity, source.temperature);
    
    return true;
}

void init_source_system(SourceSystem& src_sys) {
    src_sys.num_sources = 0;
    src_sys.total_source_particles = 0;
    src_sys.base_particle_idx = 0;
    src_sys.d_rng_states = nullptr;
    src_sys.max_rng_states = 0;
    
    // Initialize all source slots
    for (int i = 0; i < MAX_SOURCES; i++) {
        src_sys.sources[i].d_schedule_timesteps = nullptr;
        src_sys.sources[i].d_schedule_counts = nullptr;
        src_sys.sources[i].schedule_size = 0;
        src_sys.sources[i].total_particles = 0;
    }
}

void add_source(SourceSystem& src_sys, ParticleSource& source) {
    if (src_sys.num_sources >= MAX_SOURCES) {
        fprintf(stderr, "Error: Maximum sources (%d) reached!\n", MAX_SOURCES);
        return;
    }
    
    // first_particle_idx is relative to base_particle_idx (set in setup_source_rng)
    // For now, just track the offset from base
    source.first_particle_idx = src_sys.total_source_particles;
    
    // Copy source to system
    src_sys.sources[src_sys.num_sources] = source;
    src_sys.total_source_particles += source.total_particles;
    
    src_sys.num_sources++;
}

void setup_source_rng(SourceSystem& src_sys, int total_source_particles, int base_particle_idx) {
    if (src_sys.num_sources == 0 || total_source_particles == 0) {
        return;
    }
    
    // Store base index and update all source first_particle_idx to be absolute
    src_sys.base_particle_idx = base_particle_idx;
    for (int i = 0; i < src_sys.num_sources; i++) {
        src_sys.sources[i].first_particle_idx += base_particle_idx;
        printf("Source %d: particles [%d, %d)\n", i,
               src_sys.sources[i].first_particle_idx,
               src_sys.sources[i].first_particle_idx + src_sys.sources[i].total_particles);
    }
    
    // Find maximum single-timestep emission count
    int max_emit_batch = 0;
    for (int i = 0; i < src_sys.num_sources; i++) {
        ParticleSource& source = src_sys.sources[i];
        
        // Copy counts from device to find max
        std::vector<int> counts(source.schedule_size);
        CHECK_CUDA(cudaMemcpy(counts.data(), source.d_schedule_counts,
                              source.schedule_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int c : counts) {
            if (c > max_emit_batch) max_emit_batch = c;
        }
    }
    
    if (max_emit_batch <= 0) return;
    
    // Allocate RNG states
    src_sys.max_rng_states = max_emit_batch;
    CHECK_CUDA(cudaMalloc(&src_sys.d_rng_states, max_emit_batch * sizeof(curandState)));
    
    // Initialize RNG states
    int threads = 256;
    int blocks = (max_emit_batch + threads - 1) / threads;
    init_rng_kernel<<<blocks, threads>>>(src_sys.d_rng_states, max_emit_batch, 12345UL);
    CHECK_CUDA(cudaGetLastError());
    
    printf("Initialized %d RNG states for particle emission.\n", max_emit_batch);
}

void emit_particles(SourceSystem& src_sys, ParticleSystem& p_sys,
                    const CellSystem& c_sys, const SimParams& params, int current_timestep) {
    if (src_sys.num_sources == 0) return;
    
    for (int i = 0; i < src_sys.num_sources; i++) {
        ParticleSource& source = src_sys.sources[i];
        
        // Check if we've exhausted the schedule
        if (source.current_schedule_idx >= source.schedule_size) continue;
        
        // Get current schedule entry (copy from device)
        int schedule_timestep, num_to_emit;
        CHECK_CUDA(cudaMemcpy(&schedule_timestep, 
                              source.d_schedule_timesteps + source.current_schedule_idx,
                              sizeof(int), cudaMemcpyDeviceToHost));
        
        // Check if it's time to emit
        if (current_timestep != schedule_timestep) continue;
        
        CHECK_CUDA(cudaMemcpy(&num_to_emit,
                              source.d_schedule_counts + source.current_schedule_idx,
                              sizeof(int), cudaMemcpyDeviceToHost));
        
        if (num_to_emit <= 0) {
            source.current_schedule_idx++;
            continue;
        }
        
        // Safety check
        if (source.particles_generated + num_to_emit > source.total_particles) {
            fprintf(stderr, "Warning: Source %d trying to emit more particles than allocated!\n", i);
            num_to_emit = source.total_particles - source.particles_generated;
        }
        
        if (num_to_emit <= 0) {
            source.current_schedule_idx++;
            continue;
        }
        
        // Launch emission kernel
        int threads = 256;
        int blocks = (num_to_emit + threads - 1) / threads;
        
        emit_particles_kernel<<<blocks, threads>>>(
            p_sys,
            source.first_particle_idx,
            source.particles_generated,
            num_to_emit,
            source.start_x, source.start_y,
            source.end_x, source.end_y,
            source.dir_x, source.dir_y,
            source.bulk_velocity,
            source.temperature,
            params.particle_mass,
            params.cell_dx, params.cell_dy,
            params.grid_nx, params.grid_ny,
            src_sys.d_rng_states
        );
        CHECK_CUDA(cudaGetLastError());
        
        printf("Timestep %d: Source %d emitted %d particles (total: %d/%d)\n",
               current_timestep, i, num_to_emit, 
               source.particles_generated + num_to_emit, source.total_particles);
        
        // Update tracking
        source.particles_generated += num_to_emit;
        source.current_schedule_idx++;
    }
}

void free_source_system(SourceSystem& src_sys) {
    for (int i = 0; i < src_sys.num_sources; i++) {
        if (src_sys.sources[i].d_schedule_timesteps) {
            cudaFree(src_sys.sources[i].d_schedule_timesteps);
        }
        if (src_sys.sources[i].d_schedule_counts) {
            cudaFree(src_sys.sources[i].d_schedule_counts);
        }
    }
    if (src_sys.d_rng_states) {
        cudaFree(src_sys.d_rng_states);
    }
    src_sys.num_sources = 0;
    src_sys.total_source_particles = 0;
}
