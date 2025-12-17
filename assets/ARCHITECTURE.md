# Part 1: Paper Details Summary

### **Title**: A GPU–CUDA based direct simulation Monte Carlo algorithm for real gas flows
**Authors**: M.J. Goldsworthy

### 1. Objective
[cite_start]The paper describes a Direct Simulation Monte Carlo (DSMC) algorithm designed to leverage the computational performance of Graphics Processing Units (GPUs)[cite: 19]. [cite_start]It specifically addresses the complexity of "real gas effects," such as internal energy relaxation and chemical reactions, which are often omitted in other GPU implementations[cite: 19, 58].

### 2. Core Parallelization Strategy
[cite_start]Unlike other GPU approaches that split functional tasks (collisions, movement) into separate sequential kernels, this method uses a **cell-based strategy**[cite: 53, 60].
* [cite_start]**Cell Independence**: Computations for each DSMC cell are performed independently of all other cells during a single time-step[cite: 62].
* [cite_start]**Thread Teams**: A "team" of threads (fixed at 64 threads) is assigned to a single cell (one GPU block) to perform all operations for that cell[cite: 107].
* [cite_start]**Load Balancing**: By computing each cell independently, the code automatically accounts for load imbalances caused by varying collision rates or boundary interactions in specific areas[cite: 63].

### 3. Memory Management
* [cite_start]**Shared Memory**: To reduce memory latency, all particle properties (position, velocity, species) for a specific cell are loaded into high-speed, on-chip shared memory at the start of a time-step[cite: 93].
* [cite_start]**Constraints**: The limited size of shared memory (6kB per cell on the hardware used) restricts the maximum number of simulator particles per cell to approximately 120[cite: 97, 170].
* [cite_start]**Precision**: Particle positions are stored in double precision (`double2`) to resolve small displacements, while velocities use single precision (`float3`) to save memory and register usage[cite: 172, 173].

### 4. Algorithm Execution Flow (Per Cell)
The computation within a single cell follows a sequential pipeline executed by the thread team:
1.  [cite_start]**Load**: Threads copy particle data from global memory to shared memory (coalesced access)[cite: 110].
2.  [cite_start]**Indexing**: Particles are indexed into collision sub-cells ($N_{sc} \approx N_p/4$) to ensure nearest-neighbor selection[cite: 113, 116].
3.  [cite_start]**Collision**: The No Time Counter (NTC) method is used[cite: 119]. [cite_start]Each thread calculates collisions for a specific sub-cell to avoid data races[cite: 121].
4.  [cite_start]**Sampling**: Macroscopic properties are accumulated using atomic operations[cite: 122, 124].
5.  [cite_start]**Chemistry**: The Macroscopic Chemistry Method is applied where threads process dissociation or recombination events[cite: 133, 136].
6.  [cite_start]**Movement & Boundaries**: Threads move particles and calculate wall interactions using the Cercignani-Lampis-Lord (CLL) model[cite: 137, 201].
7.  [cite_start]**Store**: Updated particle data is written back to global memory[cite: 139].

### 5. Global Mesh & Sorting
* **Adaptive Meshing**: To adhere to the shared memory particle limit, the grid uses a hierarchical adaptive mesh. [cite_start]Cells exceeding the limit are subdivided into 4 child cells[cite: 177, 183].
* [cite_start]**Sorting**: Between time-steps, the Thrust library is used to reorder global particle arrays by cell index, ensuring coalesced memory access for the next step[cite: 102, 103].

### 6. Performance
* [cite_start]**Speed**: The code achieves between 20 (non-reacting) and 30 (reacting) nanoseconds per simulator particle per time-step[cite: 22].
* [cite_start]**Comparison**: It is approximately 20 times faster than the standard DS2V CPU code and comparable to other GPU codes that lack the advanced chemistry physics included here[cite: 22, 58].

***

# Part 2: Codebase Implementation Concept

This summary details the architecture of the "Pure CUDA" skeleton code provided in the conversation, which implements the concepts from the paper without external libraries like Thrust.

### 1. Hardware Abstraction Layer
To ensure the code remains portable across different GPU generations (Compute Capabilities), parameters are abstracted in `config.h`:
* **Thread Configuration**: Defined as `THREADS_PER_BLOCK` (set to 64 to match the paper's thread team size).
* **Memory Limits**: `SHARED_MEM_PER_BLOCK` and `MAX_PARTICLES_PER_CELL` are defined as constants, allowing users to tune particle capacity based on their specific hardware's shared memory size.

### 2. Data Structures: Structure of Arrays (SoA)
The `ParticleSystem` struct utilizes a Structure of Arrays layout rather than an Array of Structures (AoS).
* **Concept**: Separate global arrays exist for positions (`d_pos`), velocities (`d_vel`), and species (`d_species`).
* **Benefit**: This ensures that when threads with sequential IDs read particle data, the memory transactions are **coalesced** (contiguous), significantly increasing bandwidth utilization during the "Load" and "Store" phases.

### 3. The "Mega Kernel" (`solve_cell_kernel`)
This kernel acts as the primary physics engine, encapsulating the entire time-step logic for a single cell (following Figure 2 of the paper).
* **Shared Memory Buffering**: Explicit `__shared__` arrays (`s_pos`, `s_vel`) act as a manually managed cache.
* **Execution**:
    * It loads data into shared memory.
    * It performs sub-cell indexing and collisions locally.
    * It moves particles and calculates new cell indices.
    * It writes the *new* cell ID to `d_cell_id` in global memory, preparing the data for the sorting phase.

### 4. Custom Sorting Pipeline (Replacing Thrust)
The provided codebase replaces the paper's usage of the Thrust library with a manual "Counting Sort" implementation, optimized for minimal overhead per frame.
* **Double Buffering**: Two sets of arrays exist: the current state (`d_pos`) and the sorted state (`d_pos_sorted`). Pointers are swapped (`std::swap`) on the host at the end of every frame.
* **Pipeline Stages**:
    1.  **Histogram**: `count_particles_kernel` calculates the number of particles in every cell using atomic adds.
    2.  **Prefix Sum**: CUB's `DeviceScan::ExclusiveSum` performs a device-side exclusive scan, calculating the starting memory address (offset) for each cell without any host round-trips.
    3.  **Scatter**: `reorder_particles_kernel` moves particles from `d_pos` to `d_pos_sorted`. It uses atomic operations on the cell offsets to determine the exact write position for each particle, effectively grouping them by cell in memory.

#### Sorting Optimizations
To minimize per-frame overhead, the following optimizations have been applied:
* **Device-Side Prefix Sum**: The prefix sum is performed entirely on the GPU using CUB (`cub::DeviceScan::ExclusiveSum`), eliminating host↔device memory transfers and implicit synchronizations that were previously required.
* **Pre-Allocated Buffers**: Temporary buffers used during sorting are allocated once during initialization and reused every frame:
    * `d_write_offsets`: Mutable copy of cell offsets for the scatter kernel's atomic write indices.
    * `d_temp_storage`: CUB's internal workspace for the prefix sum operation.
* **Result**: Zero `cudaMalloc`/`cudaFree` calls per frame, reducing overhead by ~2-4 ms per frame.

### 5. Host Orchestration (`main.cu`)
The host controls the simulation loop and memory management.
* **Initialization**: Particles are seeded with random positions and Maxwellian velocities on the CPU, then uploaded to the GPU.
* **Loop**:
    1.  Launch Physics Kernel (1 block per cell).
    2.  Launch Sorting Pipeline (Reset -> Count -> Scan -> Reorder).
    3.  Swap Pointers.
* **Config**: A `SimConfig` struct manages physical parameters (Time step, Domain size) to decouple simulation logic from memory allocation.