#ifndef DSMC_DATA_TYPES_H
#define DSMC_DATA_TYPES_H

#include "config.h"

// Simulation parameters (passed to kernels)
struct SimParams {
    // Grid dimensions
    int grid_nx;      // Number of cells in X
    int grid_ny;      // Number of cells in Y
    float domain_lx;  // Domain width (meters)
    float domain_ly;  // Domain height (meters)

    // Derived (computed once)
    float cell_dx;    // Cell width
    float cell_dy;    // Cell height

    // Physics
    float dt;         // Time step
};

struct ParticleSystem {
    // Current State (SoA)
    PositionType* d_pos;
    VelocityType* d_vel;
    int* d_species;
    int* d_cell_id;  // Current cell index
    int* d_sub_id;   // Sub-cell index

    // Double Buffering for Sorting
    PositionType* d_pos_sorted;
    VelocityType* d_vel_sorted;
    int* d_species_sorted;

    int total_particles;
};

struct CellSystem {
    float* d_density;
    float* d_temperature;

    // Sorting helpers
    int* d_cell_particle_count;
    int* d_cell_offset;
    int total_cells;

    int* d_write_offsets;        // Pre-allocated for scatter kernel
    void* d_temp_storage;        // Pre-allocated for CUB scan
    size_t temp_storage_bytes;   // Size of CUB temp storage
};

#endif  // DSMC_DATA_TYPES_H