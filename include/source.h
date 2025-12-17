#ifndef DSMC_SOURCE_H
#define DSMC_SOURCE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <string>
#include <vector>

#include "data_types.h"

// Maximum number of particle sources
#define MAX_SOURCES 16

// Schedule entry: when and how many particles to emit
struct ScheduleEntry {
    int timestep;       // Timestep to emit
    int num_particles;  // Number of particles to emit
};

// Single particle source (emitter) definition
struct ParticleSource {
    // Geometry: line segment where particles are emitted
    float start_x, start_y;  // Segment start point
    float end_x, end_y;      // Segment end point

    // Emission direction (normalized, pointing into domain)
    float dir_x, dir_y;

    // Velocity distribution parameters
    float bulk_velocity;  // Mean velocity magnitude in emission direction (m/s)
    float temperature;    // Temperature for thermal velocity sampling (K)

    // Schedule data (device pointers)
    int* d_schedule_timesteps;  // Array of timesteps
    int* d_schedule_counts;     // Array of particle counts
    int schedule_size;          // Number of schedule entries
    int current_schedule_idx;   // Current position in schedule (host-tracked)

    // Particle allocation tracking
    int total_particles;      // Total particles this source will generate
    int particles_generated;  // Particles generated so far
    int first_particle_idx;   // Starting index in global particle array
};

// Source system containing all emitters
struct SourceSystem {
    ParticleSource sources[MAX_SOURCES];
    int num_sources;

    // Base index for source particles in global particle array
    int base_particle_idx;

    // Total particles from all sources (for pre-allocation)
    int total_source_particles;

    // RNG states for particle generation (one per max emission batch)
    curandState* d_rng_states;
    int max_rng_states;
};

// ============================================================================
// Host Functions
// ============================================================================

// Load source configuration from YAML file (without schedule)
// File format:
//   total_particles: <int>       # Must match schedule sum
//   geometry:
//     start_x: <float>
//     start_y: <float>
//     end_x: <float>
//     end_y: <float>
//   direction:
//     x: <float>
//     y: <float>
//   velocity:
//     thermal_vel: <float>       # OR temperature: <float>
//     stream_x: <float>          # OR bulk_velocity: <float>
//     stream_y: <float>
//     stream_z: <float>
bool load_source_config(const std::string& path, ParticleSource& source);

// Load schedule from .dat file
// File format (one entry per line):
//   <timestep> <count>
//   <timestep> <count>
//   ...
// Lines starting with # are comments
bool load_schedule(const std::string& path, ParticleSource& source);

// Load source with embedded schedule (backward compatibility)
// Supports both:
//   1. YAML with schedule section
//   2. Separate config + schedule files via load_source_config + load_schedule
bool load_source(const std::string& path, ParticleSource& source);

// Initialize an empty source system
void init_source_system(SourceSystem& src_sys);

// Add a loaded source to the system (assigns particle index range)
void add_source(SourceSystem& src_sys, ParticleSource& source);

// Setup RNG states for particle emission (call after sources are loaded)
// base_particle_idx is the index where source particles start (usually config.num_particles)
void setup_source_rng(SourceSystem& src_sys, int total_source_particles, int base_particle_idx);

// Emit particles for current timestep (call in simulation loop before physics)
void emit_particles(SourceSystem& src_sys, ParticleSystem& p_sys, const CellSystem& c_sys, const SimParams& params,
                    int current_timestep);

// Free source system resources
void free_source_system(SourceSystem& src_sys);

// ============================================================================
// Device Kernels
// ============================================================================

// Initialize cuRAND states
__global__ void init_rng_kernel(curandState* states, int num_states, unsigned long seed);

// Generate particles from a source
__global__ void emit_particles_kernel(
    ParticleSystem p_sys,
    int first_particle_idx,  // Starting index for this source's particles
    int emit_offset,         // Offset within source's allocation (particles already generated)
    int num_to_emit,         // Number of particles to emit this timestep
    float start_x, float start_y, float end_x, float end_y, float dir_x, float dir_y, float bulk_velocity,
    float temperature, float particle_mass, float cell_dx, float cell_dy, int grid_nx, int grid_ny,
    curandState* rng_states);

#endif  // DSMC_SOURCE_H
