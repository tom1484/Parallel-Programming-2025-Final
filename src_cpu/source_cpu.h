#ifndef SOURCE_CPU_H
#define SOURCE_CPU_H

#include <string>
#include <vector>
#include "data_types_cpu.h"

#define MAX_SOURCES 16

struct ParticleSource {
    // Geometry
    float start_x, start_y;
    float end_x, end_y;
    float dir_x, dir_y;
    
    // Velocity parameters
    float bulk_velocity;
    float temperature;
    
    // Schedule
    std::vector<int> schedule_timesteps;
    std::vector<int> schedule_counts;
    int current_schedule_idx;
    
    // Particle tracking
    int total_particles;
    int particles_generated;
    int first_particle_idx;
};

struct SourceSystem {
    ParticleSource sources[MAX_SOURCES];
    int num_sources;
    int base_particle_idx;
    int total_source_particles;
};

// Initialize source system
void init_source_system_cpu(SourceSystem& src_sys);

// Load source configuration and schedule
bool load_source_cpu(const std::string& config_path, const std::string& schedule_path, ParticleSource& source);

// Add source to system
void add_source_cpu(SourceSystem& src_sys, ParticleSource& source);

// Setup source system with particle indices
void setup_sources_cpu(SourceSystem& src_sys, int base_particle_idx);

// Emit particles at current timestep
void emit_particles_cpu(SourceSystem& src_sys, ParticleSystemCPU& p_sys, CellSystemCPU& c_sys, const SimParams& params, int timestep);

// Initialize source particles as inactive
void init_source_particles_inactive_cpu(ParticleSystemCPU& p_sys, int start_idx, int end_idx);

#endif // SOURCE_CPU_H
