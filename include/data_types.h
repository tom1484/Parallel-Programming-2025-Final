#ifndef DSMC_DATA_TYPES_H
#define DSMC_DATA_TYPES_H

#include "config.h"

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
};

#endif  // DSMC_DATA_TYPES_H