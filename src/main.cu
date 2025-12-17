#include <sys/stat.h>
#include <iostream>

#include "argparse.hpp"
#include "config.h"
#include "data_types.h"
#include "geometry.h"
#include "kernels.h"
#include "sim_config.h"
#include "simulation.h"
#include "sorting.h"
#include "utils.cuh"
#include "visualize.h"

using namespace std;

int main(int argc, char** argv) {
    // =========================================================================
    // Argument Parsing
    // =========================================================================
    argparse::ArgumentParser program("dsmc_solver", "1.0");
    program.add_argument("-c", "--config")
        .default_value(std::string("config.yaml"))
        .help("Path to config YAML file");
    program.add_argument("-o", "--output")
        .default_value(std::string("outputs"))
        .help("Output directory for dumps");
    program.add_argument("-g", "--geometry")
        .default_value(std::string(""))
        .help("Path to geometry file (optional)");
    program.add_argument("-d", "--dump")
        .flag()
        .help("Enable dumping simulation state");
    program.add_argument("--dump-start")
        .default_value(0)
        .scan<'i', int>()
        .help("First timestep to dump (default: 0)");
    program.add_argument("--dump-max")
        .default_value(100)
        .scan<'i', int>()
        .help("Maximum number of timesteps to dump (default: 100)");
    program.add_argument("--dump-skip")
        .default_value(1)
        .scan<'i', int>()
        .help("Dump every N timesteps (default: 1)");

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
    int dump_max = program.get<int>("--dump-max");
    int dump_skip = program.get<int>("--dump-skip");

    cout << "Config: " << config_path << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Geometry: " << (geometry_path.empty() ? "(none)" : geometry_path) << "\n";
    cout << "Dump:   " << (dump_enabled ? "enabled" : "disabled");
    if (dump_enabled) {
        cout << " (start=" << dump_start << ", max=" << dump_max << ", skip=" << dump_skip << ")";
    }
    cout << "\n";

    // =========================================================================
    // Configuration Loading
    // =========================================================================
    SimConfig config = load_config(config_path);
    SimParams sim_params = make_sim_params(config);

    printf("Simulation: %dx%d grid, dt=%.2e, steps=%d\n", 
           config.grid_nx, config.grid_ny, config.dt, config.total_steps);

    // =========================================================================
    // System Setup
    // =========================================================================
    ParticleSystem p_sys;
    CellSystem c_sys;

    // Allocate GPU memory
    allocate_system(p_sys, c_sys, config);

    // Load geometry (or initialize empty)
    if (!geometry_path.empty()) {
        load_geometry(geometry_path, c_sys, sim_params);
    } else {
        init_empty_geometry(c_sys);
    }

    // Initialize particles
    init_simulation(p_sys, c_sys, config);

    // Initial sort to ensure memory coalescing before first step
    sort_particles(p_sys, c_sys);
    swap(p_sys.d_pos, p_sys.d_pos_sorted);
    swap(p_sys.d_vel, p_sys.d_vel_sorted);
    swap(p_sys.d_species, p_sys.d_species_sorted);

    printf("Total cells: %d\n", c_sys.total_cells);
    printf("Total particles: %d\n", p_sys.total_particles);

    // =========================================================================
    // Visualization Setup
    // =========================================================================
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

    // Dump counter for tracking how many dumps have been made
    int dump_count = 0;

    // Dump initial state (timestep 0)
    if (dump_enabled && dump_start == 0 && dump_count < dump_max) {
        dump_simulation(vis_dir, 0, p_sys, c_sys);
        dump_count++;
    }

    // =========================================================================
    // Simulation Loop
    // =========================================================================
    printf("Starting Simulation for %d steps...\n", config.total_steps);

    // Calculate grid dimensions for reset/finalize kernels
    int reset_threads = 256;
    int reset_blocks = (c_sys.total_cells + reset_threads - 1) / reset_threads;

    for (int step = 0; step < config.total_steps; step++) {
        // --- Reset Sampling Accumulators ---
        reset_sampling_kernel<<<reset_blocks, reset_threads>>>(c_sys);
        CHECK_CUDA(cudaGetLastError());

        // --- Physics Kernel ---
        // Each block processes one cell independently [cite: 62]
        // 64 threads per block (thread team) [cite: 107]
        solve_cell_kernel<<<c_sys.total_cells, THREADS_PER_BLOCK>>>(p_sys, c_sys, sim_params);
        CHECK_CUDA(cudaGetLastError());

        // --- Finalize Sampling (compute density & temperature) ---
        finalize_sampling_kernel<<<reset_blocks, reset_threads>>>(c_sys, sim_params);
        CHECK_CUDA(cudaGetLastError());

        // --- Sorting Pipeline ---
        sort_particles(p_sys, c_sys);

        // --- Buffer Swap ---
        swap(p_sys.d_pos, p_sys.d_pos_sorted);
        swap(p_sys.d_vel, p_sys.d_vel_sorted);
        swap(p_sys.d_species, p_sys.d_species_sorted);

        // --- Visualization Dump ---
        int current_step = step + 1;
        if (dump_enabled && dump_count < dump_max && current_step >= dump_start) {
            if ((current_step - dump_start) % dump_skip == 0) {
                dump_simulation(vis_dir, current_step, p_sys, c_sys);
                dump_count++;
            }
        }
    }

    // =========================================================================
    // Final Result Dump (mandatory for evaluation)
    // =========================================================================
    dump_final_result(output_dir, p_sys);

    // =========================================================================
    // Cleanup
    // =========================================================================
    free_system(p_sys, c_sys);

    printf("Simulation complete.\n");
    return 0;
}