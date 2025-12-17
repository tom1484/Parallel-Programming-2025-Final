#ifndef DSMC_VISUALIZE_H
#define DSMC_VISUALIZE_H

#include <string>

#include "data_types.h"

// Dump cell data to file for visualization/debugging
// Creates file in output_dir:
//   - {timestep}-cell.dat: Cell data (particle counts, offsets, density, temperature)
void dump_cells(const std::string& output_dir, int timestep, const CellSystem& c_sys);

// Dump particle data to file for visualization/debugging
// Creates file in output_dir:
//   - {timestep}-particle.dat: Particle data (positions, velocities, species, cell IDs)
void dump_particles(const std::string& output_dir, int timestep, const ParticleSystem& p_sys);

// Dump both cells and particles (convenience function)
void dump_simulation(const std::string& output_dir, int timestep, const ParticleSystem& p_sys, const CellSystem& c_sys);

// Dump final cell data (mandatory output)
// Creates file in output_dir:
//   - cell.dat: Final cell data
void dump_final_cells(const std::string& output_dir, const CellSystem& c_sys);

#endif  // DSMC_VISUALIZE_H
