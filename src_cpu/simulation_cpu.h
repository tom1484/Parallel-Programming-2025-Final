#ifndef SIMULATION_CPU_H
#define SIMULATION_CPU_H

#include <string>
#include "data_types_cpu.h"

// Load config from YAML (wrapper around GPU version)
SimConfig load_config_cpu(const std::string& path);

// Allocate memory for particle and cell systems
void allocate_system_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys, const SimConfig& cfg, int extra_particles = 0);

// Initialize particle positions and velocities
void init_simulation_cpu(ParticleSystemCPU& p_sys, const CellSystemCPU& c_sys, const SimConfig& cfg, int num_initial_particles);

// Free allocated memory
void free_system_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys);

// Initialize empty geometry
void init_empty_geometry_cpu(CellSystemCPU& c_sys);

#endif // SIMULATION_CPU_H
