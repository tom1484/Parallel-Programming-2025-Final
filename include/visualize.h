#ifndef DSMC_VISUALIZE_H
#define DSMC_VISUALIZE_H

#include <string>

#include "data_types.h"

// Dump simulation state to files for visualization/debugging
// Creates two files in output_dir:
//   - {timestep}-cell.dat: Cell data (particle counts, offsets, density, temperature)
//   - {timestep}-particle.dat: Particle data (positions, velocities, species, cell IDs)
void dump_simulation(const std::string& output_dir, int timestep, const ParticleSystem& p_sys, const CellSystem& c_sys);

#endif  // DSMC_VISUALIZE_H
