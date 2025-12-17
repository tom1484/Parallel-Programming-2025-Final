# Paper Summary: GPU-CUDA DSMC Algorithm

### **Title**: A GPU–CUDA based direct simulation Monte Carlo algorithm for real gas flows
**Authors**: M.J. Goldsworthy

---

## 1. Objective
[cite_start]The paper describes a Direct Simulation Monte Carlo (DSMC) algorithm designed to leverage the computational performance of Graphics Processing Units (GPUs)[cite: 19]. [cite_start]It specifically addresses the complexity of "real gas effects," such as internal energy relaxation and chemical reactions, which are often omitted in other GPU implementations[cite: 19, 58].

---

## 2. Core Parallelization Strategy
[cite_start]Unlike other GPU approaches that split functional tasks (collisions, movement) into separate sequential kernels, this method uses a **cell-based strategy**[cite: 53, 60].
* [cite_start]**Cell Independence**: Computations for each DSMC cell are performed independently of all other cells during a single time-step[cite: 62].
* [cite_start]**Thread Teams**: A "team" of threads (fixed at 64 threads) is assigned to a single cell (one GPU block) to perform all operations for that cell[cite: 107].
* [cite_start]**Load Balancing**: By computing each cell independently, the code automatically accounts for load imbalances caused by varying collision rates or boundary interactions in specific areas[cite: 63].

---

## 3. Memory Management
* [cite_start]**Shared Memory**: To reduce memory latency, all particle properties (position, velocity, species) for a specific cell are loaded into high-speed, on-chip shared memory at the start of a time-step[cite: 93].
* [cite_start]**Constraints**: The limited size of shared memory (6kB per cell on the hardware used) restricts the maximum number of simulator particles per cell to approximately 120[cite: 97, 170].
* [cite_start]**Precision**: Particle positions are stored in double precision (`double2`) to resolve small displacements, while velocities use single precision (`float3`) to save memory and register usage[cite: 172, 173].

---

## 4. Algorithm Execution Flow (Per Cell)
The computation within a single cell follows a sequential pipeline executed by the thread team:
1.  [cite_start]**Load**: Threads copy particle data from global memory to shared memory (coalesced access)[cite: 110].
2.  [cite_start]**Indexing**: Particles are indexed into collision sub-cells ($N_{sc} \approx N_p/4$) to ensure nearest-neighbor selection[cite: 113, 116].
3.  [cite_start]**Collision**: The No Time Counter (NTC) method is used[cite: 119]. [cite_start]Each thread calculates collisions for a specific sub-cell to avoid data races[cite: 121].
4.  [cite_start]**Sampling**: Macroscopic properties are accumulated using atomic operations[cite: 122, 124].
5.  [cite_start]**Chemistry**: The Macroscopic Chemistry Method is applied where threads process dissociation or recombination events[cite: 133, 136].
6.  [cite_start]**Movement & Boundaries**: Threads move particles and calculate wall interactions using the Cercignani-Lampis-Lord (CLL) model[cite: 137, 201].
7.  [cite_start]**Store**: Updated particle data is written back to global memory[cite: 139].

---

## 5. Global Mesh & Sorting
* **Adaptive Meshing**: To adhere to the shared memory particle limit, the grid uses a hierarchical adaptive mesh. [cite_start]Cells exceeding the limit are subdivided into 4 child cells[cite: 177, 183].
* [cite_start]**Sorting**: Between time-steps, the Thrust library is used to reorder global particle arrays by cell index, ensuring coalesced memory access for the next step[cite: 102, 103].

---

## 6. Performance
* [cite_start]**Speed**: The code achieves between 20 (non-reacting) and 30 (reacting) nanoseconds per simulator particle per time-step[cite: 22].
* [cite_start]**Comparison**: It is approximately 20 times faster than the standard DS2V CPU code and comparable to other GPU codes that lack the advanced chemistry physics included here[cite: 22, 58].

---

> **Note**: For implementation details of this codebase, see [CODEBASE.md](CODEBASE.md).