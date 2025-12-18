#include <cmath>
#include <cstdio>
#include <random>
#include <yaml-cpp/yaml.h>
#include <fstream>

#include "source_cpu.h"

using namespace std;

void init_source_system_cpu(SourceSystem& src_sys) {
    src_sys.num_sources = 0;
    src_sys.base_particle_idx = 0;
    src_sys.total_source_particles = 0;
}

bool load_source_cpu(const string& config_path, const string& schedule_path, ParticleSource& source) {
    try {
        // Load config YAML
        YAML::Node config = YAML::LoadFile(config_path);
        
        source.total_particles = config["total_particles"].as<int>();
        
        // Geometry
        source.start_x = config["geometry"]["start_x"].as<float>();
        source.start_y = config["geometry"]["start_y"].as<float>();
        source.end_x = config["geometry"]["end_x"].as<float>();
        source.end_y = config["geometry"]["end_y"].as<float>();
        
        // Direction
        source.dir_x = config["direction"]["x"].as<float>();
        source.dir_y = config["direction"]["y"].as<float>();
        
        // Normalize direction
        float dir_len = sqrtf(source.dir_x * source.dir_x + source.dir_y * source.dir_y);
        if (dir_len > 0) {
            source.dir_x /= dir_len;
            source.dir_y /= dir_len;
        }
        
        // Velocity parameters
        float thermal_vel = config["velocity"]["thermal_vel"].as<float>();
        float stream_x = config["velocity"]["stream_x"].as<float>();
        float stream_y = config["velocity"]["stream_y"].as<float>();
        
        // Calculate bulk velocity from stream velocity
        source.bulk_velocity = sqrtf(stream_x * stream_x + stream_y * stream_y);
        
        // Convert thermal velocity to temperature
        // v_thermal = sqrt(k_B * T / m) => T = m * v_thermal^2 / k_B
        const float k_B = 1.380649e-23f;
        const float particle_mass = 6.63e-26f; // Argon
        source.temperature = particle_mass * thermal_vel * thermal_vel / k_B;
        
        printf("Loaded source config from %s: %d total particles\n", 
               config_path.c_str(), source.total_particles);
        printf("  Segment: (%.4f, %.4f) -> (%.4f, %.4f)\n",
               source.start_x, source.start_y, source.end_x, source.end_y);
        printf("  Direction: (%.4f, %.4f), bulk_vel=%.1f m/s, temp=%.1f K\n",
               source.dir_x, source.dir_y, source.bulk_velocity, source.temperature);
        
        // Load schedule
        ifstream sched_file(schedule_path);
        if (!sched_file.is_open()) {
            fprintf(stderr, "Error: Cannot open schedule file: %s\n", schedule_path.c_str());
            return false;
        }
        
        source.schedule_timesteps.clear();
        source.schedule_counts.clear();
        
        string line;
        int total_scheduled = 0;
        while (getline(sched_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            int timestep, count;
            if (sscanf(line.c_str(), "%d %d", &timestep, &count) == 2) {
                source.schedule_timesteps.push_back(timestep);
                source.schedule_counts.push_back(count);
                total_scheduled += count;
            }
        }
        
        printf("Loaded schedule from %s: %lu entries, %d total particles\n",
               schedule_path.c_str(), source.schedule_timesteps.size(), total_scheduled);
        
        if (total_scheduled != source.total_particles) {
            fprintf(stderr, "Warning: Schedule sum (%d) doesn't match total_particles (%d)\n",
                    total_scheduled, source.total_particles);
        }
        
        source.current_schedule_idx = 0;
        source.particles_generated = 0;
        
        return true;
        
    } catch (const exception& e) {
        fprintf(stderr, "Error loading source: %s\n", e.what());
        return false;
    }
}

void add_source_cpu(SourceSystem& src_sys, ParticleSource& source) {
    if (src_sys.num_sources >= MAX_SOURCES) {
        fprintf(stderr, "Error: Maximum number of sources (%d) exceeded\n", MAX_SOURCES);
        return;
    }
    
    src_sys.sources[src_sys.num_sources] = source;
    src_sys.total_source_particles += source.total_particles;
    src_sys.num_sources++;
}

void setup_sources_cpu(SourceSystem& src_sys, int base_particle_idx) {
    src_sys.base_particle_idx = base_particle_idx;
    
    int current_idx = base_particle_idx;
    for (int i = 0; i < src_sys.num_sources; i++) {
        src_sys.sources[i].first_particle_idx = current_idx;
        printf("Source %d: particles [%d, %d)\n", 
               i, current_idx, current_idx + src_sys.sources[i].total_particles);
        current_idx += src_sys.sources[i].total_particles;
    }
}

void emit_particles_cpu(SourceSystem& src_sys, ParticleSystemCPU& p_sys, 
                       CellSystemCPU& c_sys, const SimParams& params, int timestep) {
    // Use fixed seed to match GPU version (seed 12345)
    // For deterministic results, we use timestep to make emissions consistent
    static mt19937 gen(12345);  // Same seed as GPU version
    
    for (int src_idx = 0; src_idx < src_sys.num_sources; src_idx++) {
        ParticleSource& src = src_sys.sources[src_idx];
        
        // Check if we should emit at this timestep
        if (src.current_schedule_idx >= (int)src.schedule_timesteps.size()) {
            continue; // No more scheduled emissions
        }
        
        if (src.schedule_timesteps[src.current_schedule_idx] != timestep) {
            continue; // Not time yet
        }
        
        int num_emit = src.schedule_counts[src.current_schedule_idx];
        src.current_schedule_idx++;
        
        // Emit particles
        uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
        normal_distribution<float> vel_dist(0.0f, 1.0f);
        
        // Thermal velocity
        const float k_B = 1.380649e-23f;
        const float particle_mass = 6.63e-26f;
        float v_thermal = sqrtf(k_B * src.temperature / particle_mass);
        
        for (int i = 0; i < num_emit; i++) {
            if (src.particles_generated >= src.total_particles) break;
            
            int global_idx = src.first_particle_idx + src.particles_generated;
            
            // Position along segment
            float t = pos_dist(gen);
            float px = src.start_x + t * (src.end_x - src.start_x);
            float py = src.start_y + t * (src.end_y - src.start_y);
            
            // Velocity: bulk + thermal
            float vx = src.dir_x * src.bulk_velocity + vel_dist(gen) * v_thermal;
            float vy = src.dir_y * src.bulk_velocity + vel_dist(gen) * v_thermal;
            float vz = vel_dist(gen) * v_thermal;
            
            // Determine cell
            int cx = (int)(px / params.cell_dx);
            int cy = (int)(py / params.cell_dy);
            cx = max(0, min(cx, params.grid_nx - 1));
            cy = max(0, min(cy, params.grid_ny - 1));
            int cell_idx = cy * params.grid_nx + cx;
            
            // Initialize particle
            p_sys.pos[global_idx] = {px, py};
            p_sys.vel[global_idx] = {vx, vy, vz};
            p_sys.species[global_idx] = 0;
            p_sys.cell_id[global_idx] = cell_idx;
            p_sys.sub_id[global_idx] = 0;
            
            src.particles_generated++;
        }
    }
}

void init_source_particles_inactive_cpu(ParticleSystemCPU& p_sys, int start_idx, int end_idx) {
    for (int i = start_idx; i < end_idx; i++) {
        p_sys.cell_id[i] = INACTIVE_CELL_ID;
    }
}
