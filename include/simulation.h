#ifndef DSMC_SIMULATION_H
#define DSMC_SIMULATION_H

#include "data_types.h"
#include "sim_config.h"

// Allocate GPU memory for particle and cell systems
void allocate_system(ParticleSystem& p_sys, CellSystem& c_sys, const SimConfig& cfg);

// Initialize particle positions and velocities
// Must be called after geometry is loaded
void init_simulation(ParticleSystem& p_sys, const CellSystem& c_sys, const SimConfig& cfg);

// Free all GPU memory allocated by allocate_system
void free_system(ParticleSystem& p_sys, CellSystem& c_sys);

#endif  // DSMC_SIMULATION_H
