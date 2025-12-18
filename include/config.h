#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#include <vector_types.h>  // For double2, float2

// --- Hardware Abstraction ---
// Paper uses 64 threads per block (one cell) [cite: 107]
#define THREADS_PER_BLOCK 64

// Shared memory limit (e.g., 48kB or 64kB depending on architecture) [cite: 96]
#define SHARED_MEM_PER_BLOCK 6144

// --- Simulation Constants ---
// Derived max particles based on shared memory limit (approx 120 in paper) [cite: 100]
#define MAX_PARTICLES_PER_CELL 128
#define MAX_SUB_CELLS 36

// Special cell ID to mark inactive particles (not processed by physics)
#define INACTIVE_CELL_ID (-1)

// Precision abstraction
typedef double2 PositionType;  // Paper uses double for position [cite: 172]
typedef float2 VelocityType;   // 2D simulation: only vx, vy needed

#endif  // DSMC_CONFIG_H