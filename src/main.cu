#include <sys/stat.h>

#include <iostream>
#include <vector>

#include "argparse.hpp"
#include "config.h"
#include "data_types.h"
#include "geometry.h"
#include "kernels.h"
#include "sim_config.h"
#include "simulation.h"
#include "sorting.h"
#include "source.h"
#include "utils.cuh"
#include "visualize.h"

using namespace std;

int main(int argc, char** argv) {
    // =========================================================================
    // Argument Parsing
    // =========================================================================
    argparse::ArgumentParser program("dsmc_solver", "1.0");
    program.add_argument("-c", "--config").default_value(std::string("config.yaml")).help("Path to config YAML file");
    program.add_argument("-o", "--output").default_value(std::string("outputs")).help("Output directory for dumps");
    program.add_argument("-g", "--geometry").default_value(std::string("")).help("Path to geometry file (optional)");
    program.add_argument("-s", "--source")
        .append()
        .default_value(std::vector<std::string>{})
        .help("Path to source config YAML file (can be specified multiple times)");
    program.add_argument("-S", "--schedule")
        .append()
        .default_value(std::vector<std::string>{})
        .help("Path to schedule .dat file (must match -s count, or use embedded schedule in source YAML)");
    program.add_argument("--vis").flag().help("Enable visualization dumps (cells only)");
    program.add_argument("--vis-particle").flag().help("Also dump particles (requires --vis)");
    program.add_argument("--vis-start").default_value(0).scan<'i', int>().help("First timestep to dump (default: 0)");
    program.add_argument("--vis-max")
        .default_value(100)
        .scan<'i', int>()
        .help("Maximum number of timesteps to dump (default: 100)");
    program.add_argument("--vis-skip").default_value(1).scan<'i', int>().help("Dump every N timesteps (default: 1)");

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
    vector<string> source_paths = program.get<vector<string>>("--source");
    vector<string> schedule_paths = program.get<vector<string>>("--schedule");
    bool vis_enabled = program.get<bool>("--vis");
    bool vis_particle = program.get<bool>("--vis-particle");
    int vis_start = program.get<int>("--vis-start");
    int vis_max = program.get<int>("--vis-max");
    int vis_skip = program.get<int>("--vis-skip");

    // Validate source/schedule pairing
    if (!schedule_paths.empty() && schedule_paths.size() != source_paths.size()) {
        cerr << "Error: Number of --schedule (-S) arguments (" << schedule_paths.size() 
             << ") must match --source (-s) arguments (" << source_paths.size() << ")\n";
        return 1;
    }

    cout << "Config: " << config_path << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Geometry: " << (geometry_path.empty() ? "(none)" : geometry_path) << "\n";
    cout << "Sources: " << source_paths.size() << " file(s)\n";
    for (size_t i = 0; i < source_paths.size(); i++) {
        cout << "  - " << source_paths[i];
        if (i < schedule_paths.size()) {
            cout << " + " << schedule_paths[i];
        }
        cout << "\n";
    }
    cout << "Vis:    " << (vis_enabled ? "enabled" : "disabled");
    if (vis_enabled) {
        cout << " (start=" << vis_start << ", max=" << vis_max << ", skip=" << vis_skip;
        cout << ", particles=" << (vis_particle ? "yes" : "no") << ")";
    }
    cout << "\n";

    // =========================================================================
    // Configuration Loading
    // =========================================================================
    SimConfig config = load_config(config_path);
    SimParams sim_params = make_sim_params(config);

    printf("Simulation: %dx%d grid, dt=%.2e, steps=%d\n", config.grid_nx, config.grid_ny, config.dt,
           config.total_steps);

    // =========================================================================
    // Source Loading
    // =========================================================================
    SourceSystem source_sys;
    init_source_system(source_sys);

    int total_source_particles = 0;
    bool use_separate_schedules = !schedule_paths.empty();
    
    for (size_t i = 0; i < source_paths.size(); i++) {
        ParticleSource src;
        bool loaded = false;
        
        if (use_separate_schedules) {
            // Load source config and schedule separately
            if (load_source_config(source_paths[i], src)) {
                if (load_schedule(schedule_paths[i], src)) {
                    loaded = true;
                } else {
                    cerr << "Warning: Failed to load schedule: " << schedule_paths[i] << "\n";
                }
            } else {
                cerr << "Warning: Failed to load source config: " << source_paths[i] << "\n";
            }
        } else {
            // Load source with embedded schedule
            loaded = load_source(source_paths[i], src);
            if (!loaded) {
                cerr << "Warning: Failed to load source: " << source_paths[i] << "\n";
            }
        }
        
        if (loaded) {
            add_source(source_sys, src);
            total_source_particles += src.total_particles;
            printf("Added source %zu: %d total particles\n", i, src.total_particles);
        }
    }

    // =========================================================================
    // System Setup
    // =========================================================================
    ParticleSystem p_sys;
    CellSystem c_sys;

    // Calculate initial particle count (same formula as in allocate_system)
    double volume = config.domain_lx * config.domain_ly;
    int init_particles = (int)((config.init_density * volume) / config.particle_weight);

    // Allocate GPU memory (with extra space for source particles)
    allocate_system(p_sys, c_sys, config, total_source_particles);

    // Load geometry (or initialize empty)
    if (!geometry_path.empty()) {
        load_geometry(geometry_path, c_sys, sim_params);
    } else {
        init_empty_geometry(c_sys);
    }

    // Initialize particles (only initial particles from config)
    init_simulation(p_sys, c_sys, config, init_particles);

    // Initialize source particle slots as inactive
    if (total_source_particles > 0) {
        init_source_particles_inactive(p_sys, init_particles, init_particles + total_source_particles);
    }

    // Setup RNG states for sources (also assigns absolute particle indices)
    setup_source_rng(source_sys, total_source_particles, init_particles);

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

    // Check if visualization directory exists (if vis is enabled)
    if (vis_enabled) {
        struct stat st;
        if (stat(vis_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            cerr << "Error: Visualization directory does not exist: " << vis_dir << "\n";
            cerr << "Please create it before running with --vis enabled.\n";
            return 1;
        }
    }

    // Vis counter for tracking how many dumps have been made
    int vis_count = 0;

    // Dump initial state (timestep 0)
    if (vis_enabled && vis_start == 0 && vis_count < vis_max) {
        dump_cells(vis_dir, 0, c_sys);
        if (vis_particle) {
            dump_particles(vis_dir, 0, p_sys);
        }
        vis_count++;
    }

    // =========================================================================
    // Simulation Loop
    // =========================================================================
    printf("Starting Simulation for %d steps...\n", config.total_steps);

    // Calculate grid dimensions for reset/finalize kernels
    int reset_threads = 256;
    int reset_blocks = (c_sys.total_cells + reset_threads - 1) / reset_threads;

    for (int step = 0; step < config.total_steps; step++) {
        // --- Emit Particles from Sources ---
        emit_particles(source_sys, p_sys, c_sys, sim_params, step);

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
        if (vis_enabled && vis_count < vis_max && current_step >= vis_start) {
            if ((current_step - vis_start) % vis_skip == 0) {
                dump_cells(vis_dir, current_step, c_sys);
                if (vis_particle) {
                    dump_particles(vis_dir, current_step, p_sys);
                }
                vis_count++;
            }
        }
    }

    // =========================================================================
    // Final Result Dump (mandatory - cells only)
    // =========================================================================
    dump_final_cells(output_dir, c_sys);

    // =========================================================================
    // Cleanup
    // =========================================================================
    free_source_system(source_sys);
    free_system(p_sys, c_sys);

    printf("Simulation complete.\n");
    return 0;
}