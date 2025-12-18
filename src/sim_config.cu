#include <yaml-cpp/yaml.h>

#include <iostream>

#include "sim_config.h"

using namespace std;

SimConfig load_config(const string& path) {
    SimConfig cfg;
    try {
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

        cout << "Loaded Config from " << path << "\n";
        cout << "  Grid: " << cfg.grid_nx << "x" << cfg.grid_ny << "\n";
        cout << "  Steps: " << cfg.total_steps << "\n";

    } catch (const YAML::Exception& e) {
        cerr << "Error loading config: " << e.what() << "\n";
        exit(1);
    }
    return cfg;
}

SimParams make_sim_params(const SimConfig& cfg) {
    SimParams params;
    params.grid_nx = cfg.grid_nx;
    params.grid_ny = cfg.grid_ny;
    params.domain_lx = cfg.domain_lx;
    params.domain_ly = cfg.domain_ly;
    params.cell_dx = cfg.domain_lx / cfg.grid_nx;
    params.cell_dy = cfg.domain_ly / cfg.grid_ny;
    params.dt = cfg.dt;

    // Sampling parameters
    params.particle_weight = cfg.particle_weight;
    // Cell volume (2D simulation assumes unit depth = 1.0 m)
    params.cell_volume = params.cell_dx * params.cell_dy * 1.0f;
    // Molecular mass (using Argon as default gas species)
    params.particle_mass = ARGON_MASS;

    // Collision parameters (Hard Sphere model for Argon)
    params.sigma_ref = ARGON_SIGMA_REF;

    return params;
}
