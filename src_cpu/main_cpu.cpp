#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "../include/argparse.hpp"
#include "data_types_cpu.h"
#include "kernels_cpu.h"
#include "simulation_cpu.h"
#include "sorting_cpu.h"
#include "source_cpu.h"
#include "visualize_cpu.h"

using namespace std;

int main(int argc, char** argv) {
    // =========================================================================
    // Argument Parsing
    // =========================================================================
    argparse::ArgumentParser program("dsmc_solver_cpu", "1.0");
    program.add_argument("-c", "--config").default_value(string("config.yaml")).help("Path to config YAML file");
    program.add_argument("-o", "--output").default_value(string("outputs-cpu")).help("Output directory");
    program.add_argument("-s", "--source")
        .append()
        .default_value(vector<string>{})
        .help("Path to source config YAML file (can be specified multiple times)");
    program.add_argument("-S", "--schedule")
        .append()
        .default_value(vector<string>{})
        .help("Path to schedule .dat file (must match -s count)");
    program.add_argument("--vis").flag().help("Enable visualization dumps");
    program.add_argument("--vis-particle").flag().help("Also dump particles (requires --vis)");
    program.add_argument("--vis-start").default_value(0).scan<'i', int>().help("First timestep to dump");
    program.add_argument("--vis-max").default_value(100).scan<'i', int>().help("Max timesteps to dump");
    program.add_argument("--vis-skip").default_value(1).scan<'i', int>().help("Dump every N timesteps");
    program.add_argument("-t", "--threads").default_value(0).scan<'i', int>().help("Number of OpenMP threads (0=auto)");

    try {
        program.parse_args(argc, argv);
    } catch (const exception& err) {
        cerr << err.what() << endl;
        cerr << program;
        return 1;
    }

    string config_path = program.get<string>("--config");
    string output_dir = program.get<string>("--output");
    vector<string> source_paths = program.get<vector<string>>("--source");
    vector<string> schedule_paths = program.get<vector<string>>("--schedule");
    bool vis_enabled = program.get<bool>("--vis");
    bool vis_particle = program.get<bool>("--vis-particle");
    int vis_start = program.get<int>("--vis-start");
    int vis_max = program.get<int>("--vis-max");
    int vis_skip = program.get<int>("--vis-skip");
    int num_threads = program.get<int>("--threads");
    
    // Validate source/schedule pairing
    if (schedule_paths.size() != source_paths.size()) {
        cerr << "Error: Number of --schedule (-S) arguments must match --source (-s) arguments\n";
        return 1;
    }

    // Set OpenMP threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("Running with %d OpenMP threads\n", omp_get_num_threads());
        }
    }

    cout << "Config: " << config_path << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Sources: " << source_paths.size() << " file(s)\n";
    for (size_t i = 0; i < source_paths.size(); i++) {
        cout << "  - " << source_paths[i] << " + " << schedule_paths[i] << "\n";
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
    SimConfig config = load_config_cpu(config_path);
    SimParams sim_params;
    sim_params.grid_nx = config.grid_nx;
    sim_params.grid_ny = config.grid_ny;
    sim_params.domain_lx = config.domain_lx;
    sim_params.domain_ly = config.domain_ly;
    sim_params.cell_dx = config.domain_lx / config.grid_nx;
    sim_params.cell_dy = config.domain_ly / config.grid_ny;
    sim_params.dt = config.dt;
    sim_params.particle_weight = config.particle_weight;
    sim_params.cell_volume = sim_params.cell_dx * sim_params.cell_dy * 1.0f;
    sim_params.particle_mass = 6.63e-26f;  // Argon

    printf("Simulation: %dx%d grid, dt=%.2e, steps=%d\n", 
           config.grid_nx, config.grid_ny, config.dt, config.total_steps);

    // =========================================================================
    // Source Loading
    // =========================================================================
    SourceSystem source_sys;
    init_source_system_cpu(source_sys);
    
    int total_source_particles = 0;
    for (size_t i = 0; i < source_paths.size(); i++) {
        ParticleSource src;
        if (load_source_cpu(source_paths[i], schedule_paths[i], src)) {
            add_source_cpu(source_sys, src);
            total_source_particles += src.total_particles;
            printf("Added source %zu: %d total particles\n", i, src.total_particles);
        } else {
            cerr << "Warning: Failed to load source " << i << "\n";
        }
    }

    // =========================================================================
    // System Setup
    // =========================================================================
    ParticleSystemCPU p_sys;
    CellSystemCPU c_sys;

    // Calculate initial particle count
    double volume = config.domain_lx * config.domain_ly;
    int init_particles = (int)((config.init_density * volume) / config.particle_weight);

    // Allocate memory (with space for source particles)
    allocate_system_cpu(p_sys, c_sys, config, total_source_particles);

    // Initialize empty geometry (no solid objects for now)
    init_empty_geometry_cpu(c_sys);

    // Initialize particles
    init_simulation_cpu(p_sys, c_sys, config, init_particles);
    
    // Initialize source particle slots as inactive
    if (total_source_particles > 0) {
        init_source_particles_inactive_cpu(p_sys, init_particles, init_particles + total_source_particles);
    }
    
    // Setup source system
    setup_sources_cpu(source_sys, init_particles);

    // Initial sort
    sort_particles_cpu(p_sys, c_sys);

    printf("Total cells: %d\n", c_sys.total_cells);
    printf("Total particles: %d\n", p_sys.total_particles);

    // =========================================================================
    // Visualization Setup
    // =========================================================================
    string vis_dir = output_dir + "/visualization";
    if (vis_enabled) {
        struct stat st;
        if (stat(vis_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            cerr << "Error: Visualization directory does not exist: " << vis_dir << "\n";
            cerr << "Please create it before running with --vis enabled.\n";
            return 1;
        }
    }

    int vis_count = 0;

    // =========================================================================
    // Simulation Loop
    // =========================================================================
    printf("Starting CPU Simulation for %d steps...\n", config.total_steps);
    
    double start_time = omp_get_wtime();

    for (int step = 0; step < config.total_steps; step++) {
        // Emit particles from sources
        emit_particles_cpu(source_sys, p_sys, c_sys, sim_params, step);
        
        // Reset sampling
        reset_sampling_cpu(c_sys);

        // Physics kernel (cell-based parallelization)
        solve_cell_cpu(p_sys, c_sys, sim_params);

        // Finalize sampling
        finalize_sampling_cpu(c_sys, sim_params);

        // Sorting
        sort_particles_cpu(p_sys, c_sys);

        // Visualization dump
        if (vis_enabled && vis_count < vis_max && step >= vis_start) {
            if ((step - vis_start) % vis_skip == 0) {
                dump_cells_cpu(vis_dir, step, c_sys);
                if (vis_particle) {
                    dump_particles_cpu(vis_dir, step, p_sys);
                }
                vis_count++;
            }
        }

        // Progress report
        if (step % 100 == 0) {
            printf("Step %d/%d\n", step, config.total_steps);
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    printf("Simulation complete in %.3f seconds\n", elapsed);
    printf("Performance: %.2f ns per particle per timestep\n",
           (elapsed * 1e9) / (p_sys.total_particles * config.total_steps));

    // =========================================================================
    // Final Result Dump
    // =========================================================================
    dump_final_cells_cpu(output_dir, c_sys);

    // =========================================================================
    // Cleanup
    // =========================================================================
    free_system_cpu(p_sys, c_sys);

    printf("CPU simulation complete.\n");
    return 0;
}
