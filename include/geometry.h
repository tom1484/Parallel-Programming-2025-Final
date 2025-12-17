#ifndef DSMC_GEOMETRY_H
#define DSMC_GEOMETRY_H

#include <string>

#include "data_types.h"

// Load geometry from a .dat file
// File format:
//   Line 1: nx ny lx ly (grid dimensions, must match simulation config)
//   Remaining lines: cell_id start_x start_y end_x end_y normal_x normal_y
//
// Returns true if geometry was loaded, false if file doesn't exist or is empty
bool load_geometry(const std::string& path, CellSystem& c_sys, const SimParams& params);

// Initialize all segments to non-existent (no geometry)
void init_empty_geometry(CellSystem& c_sys);

#endif  // DSMC_GEOMETRY_H
