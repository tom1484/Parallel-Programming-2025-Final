#include <cstdio>
#include <string>

#include "visualize_cpu.h"

using namespace std;

// Dump final cell statistics to file
void dump_final_cells_cpu(const string& output_dir, const CellSystemCPU& c_sys) {
    string path = output_dir + "/cell.dat";
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", path.c_str());
        return;
    }

    fprintf(fp, "# cell_id particle_count offset density temperature\n");
    for (int i = 0; i < c_sys.total_cells; i++) {
        fprintf(fp, "%d %d %d %.6e %.6e\n", 
                i, c_sys.cell_particle_count[i], c_sys.cell_offset[i],
                c_sys.density[i], c_sys.temperature[i]);
    }

    fclose(fp);
    printf("Dumped final cells: %s\n", path.c_str());
}

// Dump cell data for visualization
void dump_cells_cpu(const string& vis_dir, int timestep, const CellSystemCPU& c_sys) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%08d-cell.dat", vis_dir.c_str(), timestep);
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "# cell_id particle_count offset density temperature\n");
    for (int i = 0; i < c_sys.total_cells; i++) {
        fprintf(fp, "%d %d %d %.6e %.6e\n",
                i, c_sys.cell_particle_count[i], c_sys.cell_offset[i],
                c_sys.density[i], c_sys.temperature[i]);
    }

    fclose(fp);
}

// Dump particle data for visualization
void dump_particles_cpu(const string& vis_dir, int timestep, const ParticleSystemCPU& p_sys) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%08d-particle.dat", vis_dir.c_str(), timestep);
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "# particle_id x y vx vy vz species cell_id\n");
    for (int i = 0; i < p_sys.total_particles; i++) {
        fprintf(fp, "%d %.8e %.8e %.6e %.6e %.6e %d %d\n",
                i, p_sys.pos[i].x, p_sys.pos[i].y,
                p_sys.vel[i].x, p_sys.vel[i].y, p_sys.vel[i].z,
                p_sys.species[i], p_sys.cell_id[i]);
    }

    fclose(fp);
    printf("Dumped particles: %s\n", filename);
}
