#ifndef KERNELS_CPU_H
#define KERNELS_CPU_H

#include "data_types_cpu.h"

// Reset sampling accumulators
void reset_sampling_cpu(CellSystemCPU& c_sys);

// Finalize sampling computations
void finalize_sampling_cpu(CellSystemCPU& c_sys, const SimParams& params);

// Main physics kernel (cell-based parallelization)
void solve_cell_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys, const SimParams& params);

#endif // KERNELS_CPU_H
