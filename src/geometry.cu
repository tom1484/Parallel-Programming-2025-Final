#include "geometry.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "utils.cuh"

void init_empty_geometry(CellSystem& c_sys) {
    // Initialize all segments as non-existent on host, then upload
    std::vector<Segment> h_segments(c_sys.total_cells);
    for (int i = 0; i < c_sys.total_cells; i++) {
        h_segments[i].exists = 0;
        h_segments[i].start_x = 0.0f;
        h_segments[i].start_y = 0.0f;
        h_segments[i].end_x = 0.0f;
        h_segments[i].end_y = 0.0f;
        h_segments[i].normal_x = 0.0f;
        h_segments[i].normal_y = 0.0f;
    }
    CHECK_CUDA(cudaMemcpy(c_sys.d_segments, h_segments.data(),
                          c_sys.total_cells * sizeof(Segment), cudaMemcpyHostToDevice));
}

bool load_geometry(const std::string& path, CellSystem& c_sys, const SimParams& params) {
    std::ifstream file(path);
    if (!file.is_open()) {
        printf("Geometry file not found: %s (using empty geometry)\n", path.c_str());
        init_empty_geometry(c_sys);
        return false;
    }

    // Initialize host segments (all non-existent by default)
    std::vector<Segment> h_segments(c_sys.total_cells);
    for (int i = 0; i < c_sys.total_cells; i++) {
        h_segments[i].exists = 0;
    }

    std::string line;
    int line_num = 0;
    bool header_read = false;

    while (std::getline(file, line)) {
        line_num++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);

        if (!header_read) {
            // First non-comment line is the header: nx ny lx ly
            int nx, ny;
            float lx, ly;
            if (!(iss >> nx >> ny >> lx >> ly)) {
                fprintf(stderr, "Error: Invalid header format at line %d\n", line_num);
                init_empty_geometry(c_sys);
                return false;
            }

            // Validate grid dimensions match
            if (nx != params.grid_nx || ny != params.grid_ny) {
                fprintf(stderr, "Error: Geometry grid (%d x %d) doesn't match simulation grid (%d x %d)\n",
                        nx, ny, params.grid_nx, params.grid_ny);
                init_empty_geometry(c_sys);
                return false;
            }

            // Check domain size (with small tolerance)
            float tol = 1e-4f;
            if (std::abs(lx - params.domain_lx) > tol || std::abs(ly - params.domain_ly) > tol) {
                fprintf(stderr, "Warning: Geometry domain (%.4f x %.4f) doesn't match simulation domain (%.4f x %.4f)\n",
                        lx, ly, params.domain_lx, params.domain_ly);
            }

            header_read = true;
            continue;
        }

        // Parse segment line: cell_id start_x start_y end_x end_y normal_x normal_y
        int cell_id;
        float start_x, start_y, end_x, end_y, normal_x, normal_y;

        if (!(iss >> cell_id >> start_x >> start_y >> end_x >> end_y >> normal_x >> normal_y)) {
            fprintf(stderr, "Error: Invalid segment format at line %d\n", line_num);
            continue;
        }

        // Validate cell_id
        if (cell_id < 0 || cell_id >= c_sys.total_cells) {
            fprintf(stderr, "Error: Invalid cell_id %d at line %d (max: %d)\n",
                    cell_id, line_num, c_sys.total_cells - 1);
            continue;
        }

        // Normalize the normal vector
        float norm_len = std::sqrt(normal_x * normal_x + normal_y * normal_y);
        if (norm_len > 1e-6f) {
            normal_x /= norm_len;
            normal_y /= norm_len;
        }

        // Store segment
        h_segments[cell_id].exists = 1;
        h_segments[cell_id].start_x = start_x;
        h_segments[cell_id].start_y = start_y;
        h_segments[cell_id].end_x = end_x;
        h_segments[cell_id].end_y = end_y;
        h_segments[cell_id].normal_x = normal_x;
        h_segments[cell_id].normal_y = normal_y;
    }

    file.close();

    // Count segments for info
    int seg_count = 0;
    for (int i = 0; i < c_sys.total_cells; i++) {
        if (h_segments[i].exists) seg_count++;
    }
    printf("Loaded geometry from %s: %d segments\n", path.c_str(), seg_count);

    // Upload to GPU
    CHECK_CUDA(cudaMemcpy(c_sys.d_segments, h_segments.data(),
                          c_sys.total_cells * sizeof(Segment), cudaMemcpyHostToDevice));

    return true;
}
