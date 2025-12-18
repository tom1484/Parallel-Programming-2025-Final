#ifndef SORTING_CPU_H
#define SORTING_CPU_H

#include "data_types_cpu.h"

// Sort particles by cell ID using counting sort
void sort_particles_cpu(ParticleSystemCPU& p_sys, CellSystemCPU& c_sys);

#endif // SORTING_CPU_H
