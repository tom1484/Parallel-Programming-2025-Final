#include <yaml-cpp/yaml.h>

#include <sys/stat.h>
#include <cub/cub.cuh>
#include <iostream>
#include <random>
#include <vector>

#include "argparse.hpp"
#include "config.h"
#include "data_types.h"
#include "geometry.h"
#include "kernels.h"
#include "sorting.h"
#include "utils.cuh"
#include "visualize.h"

using namespace std;

// --- Simulation Settings Container ---
struct SimConfig {
    // Simulation
    float dt;  // dt
    int total_steps;

    // Grid
    int grid_nx;      // Number of cells X
    int grid_ny;      // Number of cells Y
    float domain_lx;  // Domain width (meters)
    float domain_ly;  // Domain height (meters)

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

    // Allocate segment array for solid objects
    CHECK_CUDA(cudaMalloc(&c_sys.d_segments, c_sys.total_cells * sizeof(Segment)));

    printf("Allocated System: %d cells, capacity for %d particles.\n", c_sys.total_cells, buffer_size);
}

// --- 3. Initialization Helper (Host Side) ---

// Check if a point is inside a solid object based on segment normal
// Returns true if the point is on the "inside" (opposite to normal) side of the segment
static bool is_inside_segment(double px, double py, const Segment& seg) {
    if (!seg.exists) return false;
    
    // Vector from segment start to point
    float dx = (float)px - seg.start_x;
    float dy = (float)py - seg.start_y;
    
    // Dot product with outward normal
    // If negative, point is on the inside (opposite to normal direction)
    float dot = dx * seg.normal_x + dy * seg.normal_y;
    
    return dot < 0.0f;
}

void init_simulation(ParticleSystem& p_sys, const CellSystem& c_sys, const SimConfig& cfg) {
    // Copy segment data from GPU to check for inside cells
    vector<Segment> h_segments(c_sys.total_cells);
    CHECK_CUDA(cudaMemcpy(h_segments.data(), c_sys.d_segments,
                          c_sys.total_cells * sizeof(Segment), cudaMemcpyDeviceToHost));

    // Count inside cells and segment cells for info
    int inside_count = 0;
    int segment_count = 0;
    for (int i = 0; i < c_sys.total_cells; i++) {
        if (h_segments[i].inside) inside_count++;
        if (h_segments[i].exists) segment_count++;
    }
    if (inside_count > 0 || segment_count > 0) {
        printf("Initialization: Avoiding %d inside cells and checking %d segment cells\n", 
               inside_count, segment_count);
    }

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
        // Generate random position, rejecting positions inside solid objects
        int cell_id;
        double px, py;
        int attempts = 0;
        const int max_attempts = 1000;
        bool is_valid;

        do {
            px = dist_x(gen);
            py = dist_y(gen);
            int cx = (int)(px / dx);
            int cy = (int)(py / dy);
            // Clamp to valid range
            cx = max(0, min(cx, cfg.grid_nx - 1));
            cy = max(0, min(cy, cfg.grid_ny - 1));
            cell_id = cy * cfg.grid_nx + cx;
            
            // Check if position is valid:
            // 1. Not in a cell marked as completely inside
            // 2. Not on the inside of a segment (if cell has one)
            is_valid = !h_segments[cell_id].inside && 
                       !is_inside_segment(px, py, h_segments[cell_id]);
            
            attempts++;
        } while (!is_valid && attempts < max_attempts);

        if (attempts >= max_attempts) {
            // Fallback: place at domain corner (should not happen in practice)
            px = 0.0;
            py = 0.0;
            cell_id = 0;
        }

        h_pos[i] = make_double2(px, py);

        // Maxwellian Velocity
        h_vel[i] = make_float3(dist_v(gen), dist_v(gen), 0.0f);

        // Store cell ID
        h_cell_id[i] = cell_id;
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
    program.add_argument("-g", "--geometry").default_value(std::string("")).help("Path to geometry file (optional)");
    program.add_argument("-d", "--dump").flag().help("Enable dumping simulation state each timestep");
    program.add_argument("--dump-start").default_value(0).scan<'i', int>().help("First timestep to dump (inclusive, default: 0)");
    program.add_argument("--dump-end").default_value(100).scan<'i', int>().help("Last timestep to dump (exclusive, default: 100)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        return 1;
    }

    string config_path = program.get<string>("--config");
    string output_dir = program.get<string>("--output");
    string geometry_path = program.get<string>("--geometry");
    bool dump_enabled = program.get<bool>("--dump");
    int dump_start = program.get<int>("--dump-start");
    int dump_end = program.get<int>("--dump-end");

    cout << "Config: " << config_path << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Geometry: " << (geometry_path.empty() ? "(none)" : geometry_path) << "\n";
    cout << "Dump:   " << (dump_enabled ? "enabled" : "disabled");
    if (dump_enabled) {
        cout << " [" << dump_start << ", " << dump_end << ")";
    }
    cout << "\n";

    // --- Load Config ---
    SimConfig config = load_config(config_path);
    printf("Simulation Configured: %dx%d grid, dt=%.2e, steps=%d\n", config.grid_nx, config.grid_ny, config.dt,
           config.total_steps);

    ParticleSystem p_sys;
    CellSystem c_sys;

    // --- Setup ---
    allocate_system(p_sys, c_sys, config);

    // Create simulation parameters for kernel (needed for geometry loading)
    SimParams sim_params = make_sim_params(config);

    // Load geometry (or initialize empty)
    if (!geometry_path.empty()) {
        load_geometry(geometry_path, c_sys, sim_params);
    } else {
        init_empty_geometry(c_sys);
    }

    init_simulation(p_sys, c_sys, config);

    // Initial Sort to ensure memory Coalescing before first step
    // (Particles generated randomly on CPU are not sorted by cell)
    sort_particles(p_sys, c_sys);
    swap(p_sys.d_pos, p_sys.d_pos_sorted);
    swap(p_sys.d_vel, p_sys.d_vel_sorted);
    swap(p_sys.d_species, p_sys.d_species_sorted);

    printf("Total cells: %d\n", c_sys.total_cells);
    printf("Total particles: %d\n", p_sys.total_particles);

    // Build visualization output directory path
    string vis_dir = output_dir + "/visualization";

    // Check if visualization directory exists (if dumping is enabled)
    if (dump_enabled) {
        struct stat st;
        if (stat(vis_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            cerr << "Error: Visualization directory does not exist: " << vis_dir << "\n";
            cerr << "Please create it before running with --dump enabled.\n";
            return 1;
        }
    }

    // Dump initial state (if within range)
    if (dump_enabled && 0 >= dump_start && 0 < dump_end) {
        dump_simulation(vis_dir, 0, p_sys, c_sys);
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

        // Dump state after this timestep (if within range)
        if (dump_enabled && (step + 1) >= dump_start && (step + 1) < dump_end) {
            dump_simulation(vis_dir, step + 1, p_sys, c_sys);
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
    cudaFree(c_sys.d_segments);

    return 0;
}