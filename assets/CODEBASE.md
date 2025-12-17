# Codebase Documentation

This document explains the structure and implementation details of the DSMC (Direct Simulation Monte Carlo) GPU solver.

---

## Project Structure

```
final-dsmc/
├── CMakeLists.txt          # Build configuration
├── README.md               # Project overview
├── assets/
│   ├── ALGORITHM.md        # Paper summary and algorithm explanation
│   ├── CODEBASE.md         # This file - code structure documentation
│   └── testcases/          # YAML configuration files for test cases
│       └── case-00/
├── include/                # Header files
│   ├── argparse.hpp        # Third-party argument parser (header-only)
│   ├── config.h            # Hardware constants and type definitions
│   ├── data_types.h        # Core data structures
│   ├── sim_config.h        # Simulation configuration struct and loader
│   ├── simulation.h        # System allocation and initialization
│   ├── kernels.h           # Physics kernel declarations
│   ├── geometry.h          # Geometry loading declarations
│   ├── sorting.h           # Sorting pipeline declarations
│   ├── utils.cuh           # CUDA utility macros
│   └── visualize.h         # Visualization/dump interface
├── src/                    # Source files
│   ├── main.cu             # Entry point and orchestration
│   ├── sim_config.cu       # YAML config loading
│   ├── simulation.cu       # GPU allocation, particle initialization, cleanup
│   ├── kernels.cu          # Physics kernel implementation
│   ├── sorting.cu          # Counting sort implementation
│   ├── geometry.cu         # Solid object geometry loading
│   └── visualize.cu        # Data dump implementation
├── scripts/                # Utility scripts
│   ├── configure           # CMake configuration script
│   ├── run_release         # Run solver with test case
│   ├── visualize.py        # Python visualization and GIF generator
│   └── geometry/           # Geometry generation scripts
│       ├── base.py         # Core geometry data structures
│       └── circle.py       # Circle geometry generator
└── outputs/                # Simulation output directory
```

---

## Header Files (`include/`)

### `config.h`
Hardware abstraction layer with compile-time constants:

| Constant | Value | Description |
|----------|-------|-------------|
| `THREADS_PER_BLOCK` | 64 | Thread team size per cell (from paper) |
| `SHARED_MEM_PER_BLOCK` | 6144 | Shared memory budget per block (bytes) |
| `MAX_PARTICLES_PER_CELL` | 128 | Maximum particles in shared memory |
| `MAX_SUB_CELLS` | 36 | Maximum collision sub-cells |

**Type Definitions:**
- `PositionType` → `double2` (high precision for small displacements)
- `VelocityType` → `float3` (single precision to save memory)

---

### `data_types.h`
Core data structures passed between host and device:

#### `SimParams`
Simulation parameters passed to kernels (lightweight, copyable):
```cpp
struct SimParams {
    int grid_nx, grid_ny;       // Grid dimensions (cells)
    float domain_lx, domain_ly; // Physical domain size (meters)
    float cell_dx, cell_dy;     // Cell dimensions (derived)
    float dt;                   // Time step
};
```

#### `ParticleSystem`
Structure of Arrays (SoA) layout for coalesced memory access:
```cpp
struct ParticleSystem {
    // Current state
    PositionType* d_pos;        // Particle positions
    VelocityType* d_vel;        // Particle velocities
    int* d_species;             // Species index
    int* d_cell_id;             // Current cell assignment
    int* d_sub_id;              // Sub-cell index (for collisions)
    
    // Double buffering (for sorting)
    PositionType* d_pos_sorted;
    VelocityType* d_vel_sorted;
    int* d_species_sorted;
    
    int total_particles;
};
```

#### `Segment`
Line segment for solid object boundaries (particle-wall collisions):
```cpp
struct Segment {
    float start_x, start_y;   // Segment start point
    float end_x, end_y;       // Segment end point
    float normal_x, normal_y; // Outward normal (normalized)
    int exists;               // Whether this cell has a segment (0 or 1)
    int inside;               // Whether this cell is inside a solid object (0 or 1)
};
```

#### `CellSystem`
Per-cell data and sorting workspace:
```cpp
struct CellSystem {
    float* d_density;           // Sampled density
    float* d_temperature;       // Sampled temperature
    
    // Sorting infrastructure
    int* d_cell_particle_count; // Histogram of particles per cell
    int* d_cell_offset;         // Prefix sum (start index per cell)
    int* d_write_offsets;       // Mutable offsets for scatter (pre-allocated)
    void* d_temp_storage;       // CUB workspace (pre-allocated)
    size_t temp_storage_bytes;
    
    // Solid object geometry
    Segment* d_segments;        // Array of segments, one per cell
    
    int total_cells;
};
```

---

### `sim_config.h`
Simulation configuration loading:
```cpp
struct SimConfig {
    float dt;               // Time step
    int total_steps;        // Total simulation steps
    int grid_nx, grid_ny;   // Grid dimensions
    float domain_lx, domain_ly;  // Domain size (meters)
    float init_temp;        // Initial temperature (Kelvin)
    float init_density;     // Number density (particles/m³)
    float particle_weight;  // Real atoms per simulator particle
};

// Load configuration from YAML file
SimConfig load_config(const std::string& path);

// Create SimParams (kernel-compatible) from SimConfig
SimParams make_sim_params(const SimConfig& cfg);
```

---

### `simulation.h`
System allocation and initialization:
```cpp
// Allocate GPU memory for particle and cell systems
void allocate_system(ParticleSystem& p_sys, CellSystem& c_sys, const SimConfig& cfg);

// Initialize particle positions and velocities (avoids solid objects)
void init_simulation(ParticleSystem& p_sys, const CellSystem& c_sys, const SimConfig& cfg);

// Free all GPU memory
void free_system(ParticleSystem& p_sys, CellSystem& c_sys);
```

---

### `geometry.h`
Solid object geometry loading interface:
```cpp
// Load geometry from a .dat file
bool load_geometry(const std::string& path, CellSystem& c_sys, const SimParams& params);

// Initialize all segments to non-existent (no geometry)
void init_empty_geometry(CellSystem& c_sys);
```

---

### `kernels.h`
Physics kernel interface:
```cpp
__global__ void solve_cell_kernel(ParticleSystem p_sys, CellSystem c_sys, SimParams params);
```

---

### `sorting.h`
Counting sort pipeline:
```cpp
// Kernels
__global__ void reset_counts_kernel(int* d_counts, int num_cells);
__global__ void count_particles_kernel(const int* d_cell_id, int* d_counts, int num_particles);
__global__ void reorder_particles_kernel(ParticleSystem sys, int* d_write_offsets, int num_particles);

// Host orchestration
void sort_particles(ParticleSystem& p_sys, CellSystem& c_sys);
```

---

### `visualize.h`
Debug output interface:
```cpp
void dump_simulation(const std::string& output_dir, int timestep,
                     const ParticleSystem& p_sys, const CellSystem& c_sys);
```

---

### `utils.cuh`
CUDA error checking macro:
```cpp
#define CHECK_CUDA(call) { ... }  // Exits on error with file/line info
```

---

## Source Files (`src/`)

### `main.cu`
**Entry point and orchestration.** Clean, minimal main file with clear sections:

1. **Argument Parsing** (using `argparse.hpp`):
   - `-c, --config`: Path to YAML config file
   - `-o, --output`: Output directory for dumps
   - `-g, --geometry`: Path to geometry file (optional)
   - `-d, --dump`: Enable visualization dumps
   - `--dump-start`: First timestep to dump (default: 0)
   - `--dump-max`: Maximum dumps (default: 100)
   - `--dump-skip`: Dump every N timesteps (default: 1)

2. **Configuration Loading**: Calls `load_config()` and `make_sim_params()`

3. **System Setup**: Calls `allocate_system()`, `load_geometry()`, `init_simulation()`

4. **Simulation Loop**:
   ```
   for each timestep:
       1. Launch solve_cell_kernel (physics)
       2. Call sort_particles (reorder by cell)
       3. Swap double buffers
       4. Optionally dump state (based on dump settings)
   ```

5. **Cleanup**: Calls `free_system()`

---

### `sim_config.cu`
**YAML configuration loading.** Parses config files using yaml-cpp library.

---

### `simulation.cu`
**System allocation and initialization.**

- `allocate_system()`: Allocates all GPU memory for particles and cells
- `init_simulation()`: Initializes particle positions (avoiding solid objects) and velocities
- `free_system()`: Frees all GPU memory
- `is_inside_segment()`: Helper to check if a point is inside a solid object

---

### `kernels.cu`
**Physics kernel implementation.** Each CUDA block processes one cell independently.

**Execution Flow (per cell):**
1. **Load**: Copy particles from global → shared memory (coalesced)
2. **Sub-cell Indexing**: Assign particles to collision sub-cells
3. **Collision**: NTC method (placeholder - to be implemented)
4. **Sampling**: Accumulate macroscopic properties (placeholder)
5. **Movement**: Update positions using `p += v * dt`
6. **Segment Collision**: If cell has a segment, check ray-segment intersection and apply specular reflection
7. **Boundary**: Reflective walls (bounce particles back into domain)
8. **Re-locate**: Calculate new cell ID based on updated position
9. **Store**: Write back to global memory

**Key Implementation Details:**
- Uses `__shared__` arrays for particle data (reduces global memory latency)
- Calculates new `cell_id` from position: `cell = cy * grid_nx + cx`
- Clamps positions and cell indices to valid domain
- Segment collision uses parametric ray-segment intersection with specular reflection: `v' = v - 2(v·n)n`

---

### `geometry.cu`
**Solid object geometry loading.** Parses `.dat` files and uploads segment data to GPU.

**File Format:**
```
# Header line (must match simulation config)
nx ny lx ly
# Record format: cell_id type [segment_params...]
# type 0 = segment: cell_id 0 start_x start_y end_x end_y normal_x normal_y
# type 1 = inside:  cell_id 1 (cell is inside solid object, no particles allowed)
15 0 0.10 0.15 0.15 0.15 0.0 -1.0
16 0 0.15 0.15 0.15 0.10 1.0 0.0
25 1
26 1
```

**Record Types:**
- `type 0` (segment): Defines a boundary segment with start/end points and outward normal
- `type 1` (inside): Marks the cell as being inside a solid object (particles should not exist here)

**Functions:**
- `load_geometry()`: Parses file, validates grid dimensions, uploads to `d_segments`
- `init_empty_geometry()`: Sets all segments to `exists = 0` and `inside = 0` (no geometry)

---

### `sorting.cu`
**Counting sort implementation** to reorder particles by cell ID.

**Pipeline (3 stages):**

| Stage | Kernel | Description |
|-------|--------|-------------|
| 1. Histogram | `reset_counts_kernel` | Zero the count array |
|              | `count_particles_kernel` | Atomic increment counts per cell |
| 2. Prefix Sum | CUB `DeviceScan::ExclusiveSum` | Calculate start offset per cell |
| 3. Scatter | `reorder_particles_kernel` | Move particles to sorted positions |

**Optimizations Applied:**
- **Device-side prefix sum**: No host↔device transfers (uses CUB)
- **Pre-allocated buffers**: `d_temp_storage` and `d_write_offsets` allocated once at startup
- **Zero per-frame allocations**: Eliminates `cudaMalloc`/`cudaFree` overhead

---

### `visualize.cu`
**Debug dump implementation.** Copies GPU data to host and writes ASCII files:

- `{timestep}-cell.dat`: Cell ID, particle count, offset, density, temperature
- `{timestep}-particle.dat`: Particle ID, position (x,y), velocity (x,y,z), species, cell ID

---

## Scripts (`scripts/`)

### `visualize.py`
Python script to create animated GIFs from dump files:

```bash
python scripts/visualize.py -i outputs/test -o animation.gif [options]
```

**Options:**
| Flag | Description |
|------|-------------|
| `-i, --input` | Input directory with dump files |
| `-o, --output` | Output GIF filename |
| `-c, --config` | YAML config for domain dimensions |
| `-g, --geometry` | Geometry file (.dat) for solid object segments |
| `--fps` | Frames per second |
| `--show-grid` | Draw cell grid lines |
| `--show-velocity` | Draw velocity vectors |
| `--color-by` | Color by: `speed`, `cell`, or `species` |

---

## Build System

**CMake** with CUDA language support:

```bash
# Configure
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build/Release

# Run
./build/Release/dsmc_solver -c assets/testcases/case-00.yaml -d -o outputs/test

# Run with geometry
./build/Release/dsmc_solver -c assets/testcases/case-00.yaml -g assets/testcases/segment-test.dat -d -o outputs/test
```

**Dependencies:**
- CUDA Toolkit ≥11 (includes CUB)
- yaml-cpp (for configuration loading)

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Simulation Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  d_pos       │    │  d_pos       │    │  d_pos_sorted│       │
│  │  d_vel       │───▶│  (updated)   │───▶│  d_vel_sorted│       │
│  │  d_cell_id   │    │  d_cell_id   │    │  (reordered) │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         │            solve_cell_kernel    sort_particles        │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                        std::swap()                              │
│                    (pointer swap on host)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration File Format (YAML)

```yaml
grid:
  nx: 10          # Cells in X direction
  ny: 10          # Cells in Y direction
  lx: 0.3         # Domain width (meters)
  ly: 0.3         # Domain height (meters)

physics:
  dt: 1.0e-6      # Time step (seconds)
  total_steps: 20 # Number of simulation steps

init:
  temp: 200.0           # Initial temperature (Kelvin)
  density: 1.0e13       # Number density (particles/m³)
  particle_weight: 1.0e10  # Real atoms per simulator particle
```
