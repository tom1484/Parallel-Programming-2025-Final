#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "simulation_cpu.h"
#include <yaml-cpp/yaml.h>

using namespace std;

// Load configuration from YAML file
SimConfig load_config_cpu(const string& path) {
    SimConfig cfg;
    YAML::Node config = YAML::LoadFile(path);

    cfg.grid_nx = config["grid"]["nx"].as<int>();
    cfg.grid_ny = config["grid"]["ny"].as<int>();
    cfg.domain_lx = config["grid"]["lx"].as<float>();
    cfg.domain_ly = config["grid"]["ly"].as<float>();
    cfg.dt = config["physics"]["dt"].as<float>();
    cfg.total_steps = config["physics"]["total_steps"].as<int>();
    cfg.init_temp = config["init"]["temp"].as<float>();
    cfg.init_density = config["init"]["density"].as<float>();
    cfg.particle_weight = config["init"]["particle_weight"].as<float>();

    printf("Loaded Config from %s\n", path.c_str());
    printf("  Grid: %dx%d\n", cfg.grid_nx, cfg.grid_ny);
    printf("  Steps: %d\n", cfg.total_steps);

    return cfg;
}

// Allocate CPU memory for particle and cell systems
void allocate_system_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys, const SimConfig& cfg, int extra_particles) {
    c_sys.total_cells = cfg.grid_nx * cfg.grid_ny;

    // Calculate particle count
    double volume = cfg.domain_lx * cfg.domain_ly;
    int init_particles = (int)((cfg.init_density * volume) / cfg.particle_weight);
    int total_particles = init_particles + extra_particles;
    int buffer_size = (int)(total_particles * 1.1);

    p_sys.total_particles = total_particles;

    // Allocate particle arrays
    p_sys.pos = new PositionType[buffer_size];
    p_sys.vel = new VelocityType[buffer_size];
    p_sys.species = new int[buffer_size];
    p_sys.cell_id = new int[buffer_size];
    p_sys.sub_id = new int[buffer_size];

    // Sorted arrays (double buffering)
    p_sys.pos_sorted = new PositionType[buffer_size];
    p_sys.vel_sorted = new VelocityType[buffer_size];
    p_sys.species_sorted = new int[buffer_size];

    // Allocate cell arrays
    c_sys.density = new float[c_sys.total_cells];
    c_sys.temperature = new float[c_sys.total_cells];
    c_sys.cell_particle_count = new int[c_sys.total_cells];
    c_sys.cell_offset = new int[c_sys.total_cells];
    c_sys.vel_sum_x = new float[c_sys.total_cells];
    c_sys.vel_sum_y = new float[c_sys.total_cells];
    c_sys.vel_sum_z = new float[c_sys.total_cells];
    c_sys.vel_sq_sum = new float[c_sys.total_cells];
    c_sys.segments = new Segment[c_sys.total_cells];

    printf("Allocated CPU System: %d cells, capacity for %d particles.\n", c_sys.total_cells, buffer_size);
}

// Check if a point is inside a solid object
static bool is_inside_segment(double px, double py, const Segment& seg) {
    if (!seg.exists) return false;
    float dx = (float)px - seg.start_x;
    float dy = (float)py - seg.start_y;
    float dot = dx * seg.normal_x + dy * seg.normal_y;
    return dot < 0.0f;
}

// Initialize particles
void init_simulation_cpu(ParticleSystemCPU& p_sys, const CellSystemCPU& c_sys, const SimConfig& cfg, int num_initial_particles) {
    // Count inside/segment cells
    int inside_count = 0;
    int segment_count = 0;
    for (int i = 0; i < c_sys.total_cells; i++) {
        if (c_sys.segments[i].inside) inside_count++;
        if (c_sys.segments[i].exists) segment_count++;
    }
    if (inside_count > 0 || segment_count > 0) {
        printf("Geometry: %d segment cells, %d inside cells\n", segment_count, inside_count);
    }

    // Grid parameters
    float dx = cfg.domain_lx / cfg.grid_nx;
    float dy = cfg.domain_ly / cfg.grid_ny;

    // Random number generation with fixed seed to match GPU version
    mt19937 gen(1234);  // Same seed as GPU version
    uniform_real_distribution<double> pos_x(0.0, cfg.domain_lx);
    uniform_real_distribution<double> pos_y(0.0, cfg.domain_ly);
    normal_distribution<float> vel_dist(0.0f, 1.0f);

    // Thermal velocity from temperature
    const float k_B = 1.380649e-23f;
    const float particle_mass = 6.63e-26f;  // Argon
    float v_thermal = sqrtf(k_B * cfg.init_temp / particle_mass);

    int particle_count = 0;
    int max_attempts = num_initial_particles * 10;
    int attempts = 0;

    while (particle_count < num_initial_particles && attempts < max_attempts) {
        attempts++;

        // Generate random position
        double px = pos_x(gen);
        double py = pos_y(gen);

        // Determine cell
        int cx = (int)(px / dx);
        int cy = (int)(py / dy);
        if (cx < 0 || cx >= cfg.grid_nx || cy < 0 || cy >= cfg.grid_ny) continue;
        int cell_idx = cy * cfg.grid_nx + cx;

        // Skip if inside solid
        if (c_sys.segments[cell_idx].inside) continue;
        if (c_sys.segments[cell_idx].exists && is_inside_segment(px, py, c_sys.segments[cell_idx])) continue;

        // Initialize particle
        p_sys.pos[particle_count] = {px, py};
        p_sys.vel[particle_count] = {
            vel_dist(gen) * v_thermal,
            vel_dist(gen) * v_thermal,
            vel_dist(gen) * v_thermal
        };
        p_sys.species[particle_count] = 0;
        p_sys.cell_id[particle_count] = cell_idx;
        p_sys.sub_id[particle_count] = 0;

        particle_count++;
    }

    printf("Initialized %d particles (%d attempts)\n", particle_count, attempts);
}

// Free memory
void free_system_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys) {
    delete[] p_sys.pos;
    delete[] p_sys.vel;
    delete[] p_sys.species;
    delete[] p_sys.cell_id;
    delete[] p_sys.sub_id;
    delete[] p_sys.pos_sorted;
    delete[] p_sys.vel_sorted;
    delete[] p_sys.species_sorted;

    delete[] c_sys.density;
    delete[] c_sys.temperature;
    delete[] c_sys.cell_particle_count;
    delete[] c_sys.cell_offset;
    delete[] c_sys.vel_sum_x;
    delete[] c_sys.vel_sum_y;
    delete[] c_sys.vel_sum_z;
    delete[] c_sys.vel_sq_sum;
    delete[] c_sys.segments;
}

// Initialize empty geometry
void init_empty_geometry_cpu(CellSystemCPU& c_sys) {
    for (int i = 0; i < c_sys.total_cells; i++) {
        c_sys.segments[i].exists = 0;
        c_sys.segments[i].inside = 0;
    }
}
