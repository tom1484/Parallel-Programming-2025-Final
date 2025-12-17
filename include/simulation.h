#ifndef DSMC_SIMULATION_H
#define DSMC_SIMULATION_H

#include "data_types.h"
#include "sim_config.h"

// Allocate GPU memory for particle and cell systems
// extra_particles: additional particles to allocate (e.g., from sources)
void allocate_system(ParticleSystem& p_sys, CellSystem& c_sys, const SimConfig& cfg, int extra_particles = 0);

// Initialize particle positions and velocities for initial particles only
// Source particles start as inactive (cell_id = -1)
// Must be called after geometry is loaded
void init_simulation(ParticleSystem& p_sys, const CellSystem& c_sys, const SimConfig& cfg, int num_initial_particles);

// Initialize source particles as inactive (call after init_simulation)
void init_source_particles_inactive(ParticleSystem& p_sys, int init_particles, int total_particles);

// Free all GPU memory allocated by allocate_system
void free_system(ParticleSystem& p_sys, CellSystem& c_sys);

#endif  // DSMC_SIMULATION_H
