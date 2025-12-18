#ifndef DATA_TYPES_CPU_H
#define DATA_TYPES_CPU_H

// CPU version type definitions (no CUDA vector types)
#define MAX_PARTICLES_PER_CELL 128
#define MAX_SUB_CELLS 36
#define INACTIVE_CELL_ID (-1)

// CPU equivalents of CUDA vector types
struct double2 {
    double x, y;
};

struct float3 {
    float x, y, z;
};

typedef double2 PositionType;
typedef float3 VelocityType;

struct SimConfig {
    float dt;
    int total_steps;
    int grid_nx, grid_ny;
    float domain_lx, domain_ly;
    float init_temp;
    float init_density;
    float particle_weight;
};

struct SimParams {
    int grid_nx, grid_ny;
    float domain_lx, domain_ly;
    float cell_dx, cell_dy;
    float dt;
    float particle_weight;
    float cell_volume;
    float particle_mass;
};

struct Segment {
    float start_x, start_y;
    float end_x, end_y;
    float normal_x, normal_y;
    int exists;
    int inside;
};

struct ParticleSystemCPU {
    // Current state
    PositionType* pos;
    VelocityType* vel;
    int* species;
    int* cell_id;
    int* sub_id;

    // Sorted arrays (for sorting)
    PositionType* pos_sorted;
    VelocityType* vel_sorted;
    int* species_sorted;

    int total_particles;
};

struct CellSystemCPU {
    float* density;
    float* temperature;

    // Velocity accumulators
    float* vel_sum_x;
    float* vel_sum_y;
    float* vel_sum_z;
    float* vel_sq_sum;

    // Sorting helpers
    int* cell_particle_count;
    int* cell_offset;
    int total_cells;

    // Geometry
    Segment* segments;
};

#endif // DATA_TYPES_CPU_H
