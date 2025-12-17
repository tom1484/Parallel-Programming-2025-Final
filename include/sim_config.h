#ifndef DSMC_SIM_CONFIG_H
#define DSMC_SIM_CONFIG_H

#include <string>
#include "data_types.h"

// --- Simulation Settings Container ---
struct SimConfig {
    // Simulation
    float dt;           // Time step
    int total_steps;    // Total number of simulation steps

    // Grid
    int grid_nx;        // Number of cells X
    int grid_ny;        // Number of cells Y
    float domain_lx;    // Domain width (meters)
    float domain_ly;    // Domain height (meters)

    // Initialization
    float init_temp;        // Initial temperature (Kelvin)
    float init_density;     // Number density (particles/m³)
    float particle_weight;  // Real atoms per simulator particle
};

// Load simulation configuration from YAML file
SimConfig load_config(const std::string& path);

// Create SimParams (for kernel) from SimConfig
SimParams make_sim_params(const SimConfig& cfg);

#endif  // DSMC_SIM_CONFIG_H
