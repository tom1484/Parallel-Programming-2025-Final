#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "utils.cuh"
#include "visualize.h"

// Internal helper function that does the actual dump work
static void dump_impl(const std::string& cell_path, const std::string& particle_path,
                      const std::string& header_info, const ParticleSystem& p_sys, const CellSystem& c_sys) {
    // ----------------------------------------------------------
    // Copy cell data from device to host
    // ----------------------------------------------------------
    std::vector<int> h_cell_particle_count(c_sys.total_cells);
    std::vector<int> h_cell_offset(c_sys.total_cells);
    std::vector<float> h_density(c_sys.total_cells);
    std::vector<float> h_temperature(c_sys.total_cells);

    CHECK_CUDA(cudaMemcpy(h_cell_particle_count.data(), c_sys.d_cell_particle_count, c_sys.total_cells * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_cell_offset.data(), c_sys.d_cell_offset, c_sys.total_cells * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_density.data(), c_sys.d_density, c_sys.total_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_temperature.data(), c_sys.d_temperature, c_sys.total_cells * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------
    // Write cell data
    // ----------------------------------------------------------
    std::ofstream cell_file(cell_path);
    if (!cell_file.is_open()) {
        fprintf(stderr, "Error: Could not open %s for writing\n", cell_path.c_str());
        return;
    }

    cell_file << "# Cell data" << header_info << "\n";
    cell_file << "# cell_id particle_count offset density temperature\n";
    cell_file << std::fixed << std::setprecision(6);

    for (int i = 0; i < c_sys.total_cells; i++) {
        cell_file << i << " " << h_cell_particle_count[i] << " " << h_cell_offset[i] << " " << h_density[i] << " "
                  << h_temperature[i] << "\n";
    }
    cell_file.close();

    // ----------------------------------------------------------
    // Copy particle data from device to host
    // ----------------------------------------------------------
    std::vector<PositionType> h_pos(p_sys.total_particles);
    std::vector<VelocityType> h_vel(p_sys.total_particles);
    std::vector<int> h_species(p_sys.total_particles);
    std::vector<int> h_cell_id(p_sys.total_particles);

    CHECK_CUDA(
        cudaMemcpy(h_pos.data(), p_sys.d_pos, p_sys.total_particles * sizeof(PositionType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_vel.data(), p_sys.d_vel, p_sys.total_particles * sizeof(VelocityType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_species.data(), p_sys.d_species, p_sys.total_particles * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(h_cell_id.data(), p_sys.d_cell_id, p_sys.total_particles * sizeof(int), cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------
    // Write particle data
    // ----------------------------------------------------------
    std::ofstream particle_file(particle_path);
    if (!particle_file.is_open()) {
        fprintf(stderr, "Error: Could not open %s for writing\n", particle_path.c_str());
        return;
    }

    particle_file << "# Particle data" << header_info << "\n";
    particle_file << "# particle_id pos_x pos_y vel_x vel_y vel_z species cell_id\n";
    particle_file << std::fixed << std::setprecision(8);

    for (int i = 0; i < p_sys.total_particles; i++) {
        particle_file << i << " " << h_pos[i].x << " " << h_pos[i].y << " " << h_vel[i].x << " " << h_vel[i].y << " "
                      << h_vel[i].z << " " << h_species[i] << " " << h_cell_id[i] << "\n";
    }
    particle_file.close();
}

void dump_simulation(const std::string& output_dir, int timestep, const ParticleSystem& p_sys,
                     const CellSystem& c_sys) {
    std::ostringstream cell_path, particle_path, header_info;
    cell_path << output_dir << "/" << std::setw(8) << std::setfill('0') << timestep << "-cell.dat";
    particle_path << output_dir << "/" << std::setw(8) << std::setfill('0') << timestep << "-particle.dat";
    header_info << " at timestep " << timestep;

    dump_impl(cell_path.str(), particle_path.str(), header_info.str(), p_sys, c_sys);
    printf("Dumped timestep %d: %s, %s\n", timestep, cell_path.str().c_str(), particle_path.str().c_str());
}

void dump_final_result(const std::string& output_dir, const ParticleSystem& p_sys, const CellSystem& c_sys) {
    std::string cell_path = output_dir + "/cell.dat";
    std::string particle_path = output_dir + "/particle.dat";

    dump_impl(cell_path, particle_path, " (final)", p_sys, c_sys);
    printf("Dumped final result: %s, %s\n", cell_path.c_str(), particle_path.c_str());
}
