#include <yaml-cpp/yaml.h>

#include <cub/cub.cuh>
#include <iostream>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "config.h"
#include "data_types.h"
#include "kernels.h"
#include "sorting.h"
#include "utils.cuh"
#include "visualize.h"

using namespace std;

// --- Simulation Settings Container ---
struct SimConfig {
    // Grid
    int grid_nx;      // Number of cells X
    int grid_ny;      // Number of cells Y
    float domain_lx;  // Domain width (meters)
    float domain_ly;  // Domain height (meters)

    // Physics
    float dt;  // dt
    int total_steps;

    // Initialization
    float init_temp;        // Kelvin
    float init_density;     // Number density
    float particle_weight;  // Real atoms per simulator particle
};

// --- Config Loader ---
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

// --- Create SimParams from SimConfig ---
SimParams make_sim_params(const SimConfig& cfg) {
    SimParams params;
    params.grid_nx = cfg.grid_nx;
    params.grid_ny = cfg.grid_ny;
    params.domain_lx = cfg.domain_lx;
    params.domain_ly = cfg.domain_ly;
    params.cell_dx = cfg.domain_lx / cfg.grid_nx;
    params.cell_dy = cfg.domain_ly / cfg.grid_ny;
    params.dt = cfg.dt;
    return params;
}

// --- 2. Allocation Helper ---
void allocate_system(ParticleSystem& p_sys, CellSystem& c_sys, const SimConfig& cfg) {
    // Calculate totals
    c_sys.total_cells = cfg.grid_nx * cfg.grid_ny;

    // Estimate total particles based on density and volume
    // (In practice, allocate extra buffer for inflow)
    double volume = (cfg.domain_lx * cfg.domain_ly);
    int est_particles = (int)((cfg.init_density * volume) / cfg.particle_weight);
    int buffer_size = est_particles * 1.5;  // 50% buffer for fluctuation

    p_sys.total_particles = est_particles;

    // --- GPU Allocations (Particle System) ---
    // Note: We use Double Precision for Position [cite: 173]
    CHECK_CUDA(cudaMalloc(&p_sys.d_pos, buffer_size * sizeof(PositionType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_vel, buffer_size * sizeof(VelocityType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_species, buffer_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_cell_id, buffer_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_sub_id, buffer_size * sizeof(int)));

    // Sorted Arrays (Double Buffering)
    CHECK_CUDA(cudaMalloc(&p_sys.d_pos_sorted, buffer_size * sizeof(PositionType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_vel_sorted, buffer_size * sizeof(VelocityType)));
    CHECK_CUDA(cudaMalloc(&p_sys.d_species_sorted, buffer_size * sizeof(int)));

    // --- GPU Allocations (Cell System) ---
    CHECK_CUDA(cudaMalloc(&c_sys.d_density, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_temperature, c_sys.total_cells * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_cell_particle_count, c_sys.total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_cell_offset, c_sys.total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_sys.d_write_offsets, c_sys.total_cells * sizeof(int)));

    // Pre-allocate CUB temp storage (query size first)
    c_sys.d_temp_storage = nullptr;
    c_sys.temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(c_sys.d_temp_storage, c_sys.temp_storage_bytes, c_sys.d_cell_particle_count,
                                  c_sys.d_cell_offset, c_sys.total_cells);
    CHECK_CUDA(cudaMalloc(&c_sys.d_temp_storage, c_sys.temp_storage_bytes));

    printf("Allocated System: %d cells, capacity for %d particles.\n", c_sys.total_cells, buffer_size);
}

// --- 3. Initialization Helper (Host Side) ---
void init_simulation(ParticleSystem& p_sys, const SimConfig& cfg) {
    vector<PositionType> h_pos(p_sys.total_particles);
    vector<VelocityType> h_vel(p_sys.total_particles);
    vector<int> h_cell_id(p_sys.total_particles);

    mt19937 gen(1234);
    uniform_real_distribution<double> dist_x(0.0, cfg.domain_lx);
    uniform_real_distribution<double> dist_y(0.0, cfg.domain_ly);
    normal_distribution<float> dist_v(0.0f, 300.0f);  // Approx thermal velocity

    float dx = cfg.domain_lx / cfg.grid_nx;
    float dy = cfg.domain_ly / cfg.grid_ny;

    for (int i = 0; i < p_sys.total_particles; i++) {
        // Random Position
        h_pos[i] = make_double2(dist_x(gen), dist_y(gen));

        // Maxwellian Velocity
        h_vel[i] = make_float3(dist_v(gen), dist_v(gen), 0.0f);

        // Calculate Initial Cell ID
        int cx = (int)(h_pos[i].x / dx);
        int cy = (int)(h_pos[i].y / dy);
        h_cell_id[i] = cy * cfg.grid_nx + cx;
    }

    // Copy to GPU
    CHECK_CUDA(
        cudaMemcpy(p_sys.d_pos, h_pos.data(), p_sys.total_particles * sizeof(PositionType), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(p_sys.d_vel, h_vel.data(), p_sys.total_particles * sizeof(VelocityType), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(p_sys.d_cell_id, h_cell_id.data(), p_sys.total_particles * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize species to 0 (single species simulation)
    CHECK_CUDA(cudaMemset(p_sys.d_species, 0, p_sys.total_particles * sizeof(int)));
}

int main(int argc, char** argv) {
    // --- Argument Parser (argparse) ---
    argparse::ArgumentParser program("dsmc_solver", "1.0");
    program.add_argument("-c", "--config").default_value(std::string("config.yaml")).help("Path to config YAML file");
    program.add_argument("-o", "--output").default_value(std::string("outputs")).help("Output directory for dumps");
    program.add_argument("-d", "--dump").flag().help("Enable dumping simulation state each timestep");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        return 1;
    }

    string config_path = program.get<string>("--config");
    string output_dir = program.get<string>("--output");
    bool dump_enabled = program.get<bool>("--dump");

    cout << "Config: " << config_path << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Dump:   " << (dump_enabled ? "enabled" : "disabled") << "\n";

    // --- Load Config ---
    SimConfig config = load_config(config_path);
    printf("Simulation Configured: %dx%d grid, dt=%.2e, steps=%d\n", config.grid_nx, config.grid_ny, config.dt,
           config.total_steps);

    ParticleSystem p_sys;
    CellSystem c_sys;

    // --- Setup ---
    allocate_system(p_sys, c_sys, config);
    init_simulation(p_sys, config);

    // Initial Sort to ensure memory Coalescing before first step
    // (Particles generated randomly on CPU are not sorted by cell)
    sort_particles(p_sys, c_sys);
    swap(p_sys.d_pos, p_sys.d_pos_sorted);
    swap(p_sys.d_vel, p_sys.d_vel_sorted);
    swap(p_sys.d_species, p_sys.d_species_sorted);

    printf("Total cells: %d\n", c_sys.total_cells);
    printf("Total particles: %d\n", p_sys.total_particles);

    // Create simulation parameters for kernel
    SimParams sim_params = make_sim_params(config);

    // Dump initial state
    if (dump_enabled) {
        dump_simulation(output_dir, 0, p_sys, c_sys);
    }

    // --- Time Loop ---
    printf("Starting Simulation for %d steps...\n", config.total_steps);

    for (int step = 0; step < config.total_steps; step++) {
        // Threads per block fixed at 64 [cite: 107]
        // Grid size = Total Cells (One block per cell) [cite: 95]
        solve_cell_kernel<<<c_sys.total_cells, THREADS_PER_BLOCK>>>(p_sys, c_sys, sim_params);
        CHECK_CUDA(cudaGetLastError());  // Catch launch errors

        // --- Sorting / Indexing Pipeline ---
        sort_particles(p_sys, c_sys);

        // --- Ping-Pong Buffers ---
        swap(p_sys.d_pos, p_sys.d_pos_sorted);
        swap(p_sys.d_vel, p_sys.d_vel_sorted);
        swap(p_sys.d_species, p_sys.d_species_sorted);

        // Dump state after this timestep
        if (dump_enabled) {
            dump_simulation(output_dir, step + 1, p_sys, c_sys);
        }
    }

    // --- Cleanup ---
    cudaFree(p_sys.d_pos);
    cudaFree(p_sys.d_pos_sorted);
    cudaFree(p_sys.d_vel);
    cudaFree(p_sys.d_vel_sorted);
    cudaFree(p_sys.d_species);
    cudaFree(p_sys.d_species_sorted);
    cudaFree(p_sys.d_cell_id);
    cudaFree(c_sys.d_cell_particle_count);
    cudaFree(c_sys.d_cell_offset);
    cudaFree(c_sys.d_write_offsets);
    cudaFree(c_sys.d_temp_storage);

    return 0;
}