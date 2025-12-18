#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#include <vector_types.h>  // For double2, float2

// --- Hardware Abstraction ---
// Paper uses 64 threads per block (one cell) [cite: 107]
#define THREADS_PER_BLOCK 64

// Shared memory limit (e.g., 24kB or 48kB depending on architecture) [cite: 96]
#define SHARED_MEM_PER_BLOCK 24576

// --- Simulation Constants ---
// Derived max particles based on shared memory limit (approx 120 in paper) [cite: 100]
#define MAX_PARTICLES_PER_CELL 512
#define MAX_SUB_CELLS 64

// Special cell ID to mark inactive particles (not processed by physics)
#define INACTIVE_CELL_ID (-1)

// --- Gas Species Constants (Argon) ---
// Argon: 39.948 amu = 39.948 * 1.66054e-27 kg
#define ARGON_MASS 6.6335e-26f  // kg

// Argon hard-sphere diameter (from Bird's DSMC book)
#define ARGON_DIAMETER 4.17e-10f  // meters

// Reference cross-section for Argon (π * d²)
// σ = π * (4.17e-10)² ≈ 5.465e-19 m²
#define ARGON_SIGMA_REF 5.465e-19f  // m²

// Initial (σ × c_r)_max estimate
// For T~300K Argon: c_r_mean ≈ 560 m/s, so (σ × c_r)_max ≈ 3e-16 m³/s
#define INITIAL_SIGMA_CR_MAX 3.0e-16f  // m³/s

// Precision abstraction
typedef double2 PositionType;  // Paper uses double for position [cite: 172]
typedef float2 VelocityType;   // 2D simulation: only vx, vy needed

#endif  // DSMC_CONFIG_H