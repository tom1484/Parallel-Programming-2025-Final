#include <algorithm>
#include <cstring>
#include <omp.h>

#include "sorting_cpu.h"

// CPU version of particle sorting using OpenMP
// Implements counting sort (same algorithm as GPU version)

void sort_particles_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys) {
    int num_particles = p_sys.total_particles;
    int num_cells = c_sys.total_cells;

    // Step 1: Reset counts
    #pragma omp parallel for
    for (int i = 0; i < num_cells; i++) {
        c_sys.cell_particle_count[i] = 0;
    }

    // Step 2: Count particles per cell (histogram)
    // Need to handle race conditions - use atomic updates or reduction
    #pragma omp parallel
    {
        // Thread-local histogram
        int* local_counts = new int[num_cells];
        std::memset(local_counts, 0, num_cells * sizeof(int));

        #pragma omp for nowait
        for (int i = 0; i < num_particles; i++) {
            int cell = p_sys.cell_id[i];
            if (cell != INACTIVE_CELL_ID) {
                local_counts[cell]++;
            }
        }

        // Reduce local histograms to global
        #pragma omp critical
        {
            for (int i = 0; i < num_cells; i++) {
                c_sys.cell_particle_count[i] += local_counts[i];
            }
        }

        delete[] local_counts;
    }

    // Step 3: Prefix sum (exclusive scan)
    c_sys.cell_offset[0] = 0;
    for (int i = 1; i < num_cells; i++) {
        c_sys.cell_offset[i] = c_sys.cell_offset[i-1] + c_sys.cell_particle_count[i-1];
    }

    // Step 4: Create working copy of offsets for scatter
    int* write_offsets = new int[num_cells];
    std::memcpy(write_offsets, c_sys.cell_offset, num_cells * sizeof(int));

    // Calculate where inactive particles should go (at the end, after all active particles)
    int inactive_start = 0;
    if (num_cells > 0) {
        // Find the last non-empty cell
        for (int i = num_cells - 1; i >= 0; i--) {
            if (c_sys.cell_particle_count[i] > 0) {
                inactive_start = c_sys.cell_offset[i] + c_sys.cell_particle_count[i];
                break;
            }
        }
    }
    int inactive_write_idx = inactive_start;

    // Step 5: Scatter particles to sorted positions
    // This needs to be serial or use atomic updates
    for (int i = 0; i < num_particles; i++) {
        int cell = p_sys.cell_id[i];
        
        int write_idx;
        if (cell == INACTIVE_CELL_ID) {
            write_idx = inactive_write_idx++;
        } else {
            write_idx = write_offsets[cell]++;
        }

        p_sys.pos_sorted[write_idx] = p_sys.pos[i];
        p_sys.vel_sorted[write_idx] = p_sys.vel[i];
        p_sys.species_sorted[write_idx] = p_sys.species[i];
    }

    // Step 6: Swap pointers (double buffering)
    std::swap(p_sys.pos, p_sys.pos_sorted);
    std::swap(p_sys.vel, p_sys.vel_sorted);
    std::swap(p_sys.species, p_sys.species_sorted);

    delete[] write_offsets;
}
