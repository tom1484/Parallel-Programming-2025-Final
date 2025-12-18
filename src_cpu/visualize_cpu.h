#ifndef VISUALIZE_CPU_H
#define VISUALIZE_CPU_H

#include <string>
#include "data_types_cpu.h"

// Dump final cell data
void dump_final_cells_cpu(const std::string& output_dir, const CellSystemCPU& c_sys);

// Dump cell data at a specific timestep
void dump_cells_cpu(const std::string& vis_dir, int timestep, const CellSystemCPU& c_sys);

// Dump particle data at a specific timestep
void dump_particles_cpu(const std::string& vis_dir, int timestep, const ParticleSystemCPU& p_sys);

#endif // VISUALIZE_CPU_H
