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
│           ├── config.yaml
│           ├── geometry/   # Geometry files
│           └── source/     # Particle source schedule files
├── include/                # Header files
│   ├── argparse.hpp        # Third-party argument parser (header-only)
│   ├── config.h            # Hardware constants and type definitions
│   ├── data_types.h        # Core data structures
│   ├── sim_config.h        # Simulation configuration struct and loader
│   ├── simulation.h        # System allocation and initialization
│   ├── kernels.h           # Physics kernel declarations
│   ├── geometry.h          # Geometry loading declarations
│   ├── source.h            # Particle source/emitter declarations
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
│   ├── source.cu           # Particle source emission implementation
│   └── visualize.cu        # Data dump implementation
├── scripts/                # Utility scripts
│   ├── configure           # CMake configuration script
│   ├── run_release         # Run solver with test case
│   ├── visualize.py        # Python visualization and GIF generator
│   ├── draw_result.py      # Python heatmap plotting for cell stats
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
| `SHARED_MEM_PER_BLOCK` | 24576 | Shared memory budget per block (bytes) |
| `MAX_PARTICLES_PER_CELL` | 512 | Maximum particles in shared memory |
| `MAX_SUB_CELLS` | 64 | Maximum collision sub-cells |
| `INACTIVE_CELL_ID` | -1 | Cell ID for inactive/unborn particles |

**Gas Species Constants (Argon):**

| Constant | Value | Description |
|----------|-------|-------------|
| `ARGON_MASS` | 6.6335e-26 | Molecular mass (kg) |
| `ARGON_DIAMETER` | 4.17e-10 | Hard-sphere diameter (m) |
| `ARGON_SIGMA_REF` | 5.465e-19 | Reference cross-section π×d² (m²) |
| `INITIAL_SIGMA_CR_MAX` | 3.0e-16 | Initial (σ×c_r)_max estimate (m³/s) |

**Type Definitions:**
- `PositionType` → `double2` (high precision for small displacements)
- `VelocityType` → `float2` (2D simulation: only vx, vy needed)

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
    
    // Sampling parameters
    float particle_weight;      // Real atoms per simulator particle (Fnum)
    float cell_volume;          // Cell volume (dx * dy * 1.0 for 2D)
    float particle_mass;        // Molecular mass (kg), default: Argon
    
    // Collision parameters (NTC method)
    float sigma_ref;            // Reference cross-section (m²) for hard-sphere model
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
    float* d_density;           // Sampled number density (m⁻³)
    float* d_temperature;       // Sampled temperature (K)
    
    // Velocity accumulators for macroscopic sampling
    float* d_vel_sum_x;         // Sum of vx per cell
    float* d_vel_sum_y;         // Sum of vy per cell
    float* d_vel_sq_sum;        // Sum of |v|² per cell
    
    // Sorting infrastructure
    int* d_cell_particle_count; // Histogram of particles per cell
    int* d_cell_offset;         // Prefix sum (start index per cell)
    int* d_write_offsets;       // Mutable offsets for scatter (pre-allocated)
    void* d_temp_storage;       // CUB workspace (pre-allocated)
    size_t temp_storage_bytes;
    
    // Solid object geometry
    Segment* d_segments;        // Array of segments, one per cell
    
    // Collision tracking (NTC method)
    float* d_sigma_cr_max;        // Maximum (σ × c_r) per cell
    curandState* d_rng_collision; // RNG states for collision (one per cell)
    
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
// Reset sampling accumulators to zero before each timestep
__global__ void reset_sampling_kernel(CellSystem c_sys);

// Compute final density & temperature from accumulated sums
__global__ void finalize_sampling_kernel(CellSystem c_sys, SimParams params);

// Main physics kernel - processes one cell per block
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
// Dump cell data to file
void dump_cells(const std::string& output_dir, int timestep, const CellSystem& c_sys);

// Dump particle data to file
void dump_particles(const std::string& output_dir, int timestep, const ParticleSystem& p_sys);

// Dump both cells and particles (convenience function)
void dump_simulation(const std::string& output_dir, int timestep,
                     const ParticleSystem& p_sys, const CellSystem& c_sys);

// Dump final cell data (mandatory output)
void dump_final_cells(const std::string& output_dir, const CellSystem& c_sys);
```

---

### `source.h`
Particle source/emitter system for injecting particles during simulation:

#### `ParticleSource`
Single particle emitter definition:
```cpp
struct ParticleSource {
    // Geometry: line segment where particles are emitted
    float start_x, start_y;     // Segment start point
    float end_x, end_y;         // Segment end point
    
    // Emission direction (normalized, pointing into domain)
    float dir_x, dir_y;
    
    // Velocity distribution parameters
    float bulk_velocity;        // Mean velocity in emission direction (m/s)
    float temperature;          // Temperature for thermal velocity (K)
    
    // Schedule data (device pointers)
    int* d_schedule_timesteps;  // Array of timesteps to emit
    int* d_schedule_counts;     // Array of particle counts per timestep
    int schedule_size;          // Number of schedule entries
    int current_schedule_idx;   // Current position in schedule
    
    // Particle allocation tracking
    int total_particles;        // Total particles this source will generate
    int particles_generated;    // Particles generated so far
    int first_particle_idx;     // Starting index in global particle array
};
```

#### `SourceSystem`
Container for all particle sources:
```cpp
struct SourceSystem {
    ParticleSource sources[MAX_SOURCES];  // Up to 16 sources
    int num_sources;
    int base_particle_idx;        // Base index for source particles
    int total_source_particles;   // Sum of all sources' particles
    curandState* d_rng_states;    // GPU random states for emission
    int max_rng_states;
};
```

#### Functions:
```cpp
// Load source configuration from YAML file (without schedule)
bool load_source_config(const std::string& path, ParticleSource& source);

// Load schedule from .dat file
bool load_schedule(const std::string& path, ParticleSource& source);

// Load source with embedded schedule (backward compatibility)
bool load_source(const std::string& path, ParticleSource& source);

// Initialize empty source system
void init_source_system(SourceSystem& src_sys);

// Add a loaded source to the system
void add_source(SourceSystem& src_sys, ParticleSource& source);

// Setup RNG states and assign particle indices
void setup_source_rng(SourceSystem& src_sys, int total_source_particles, int base_particle_idx);

// Emit particles for current timestep (call in simulation loop)
void emit_particles(SourceSystem& src_sys, ParticleSystem& p_sys,
                    const CellSystem& c_sys, const SimParams& params, int current_timestep);

// Free source system resources
void free_source_system(SourceSystem& src_sys);
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
   - `-s, --source`: Path to source config YAML file (can be specified multiple times)
   - `-S, --schedule`: Path to schedule .dat file (must match -s count, or use embedded schedule)
   - `--vis`: Enable visualization dumps (cells only)
   - `--vis-particle`: Also dump particles (requires --vis)
   - `--vis-start`: First timestep to dump (default: 0)
   - `--vis-max`: Maximum dumps (default: 100)
   - `--vis-skip`: Dump every N timesteps (default: 1)

2. **Configuration Loading**: Calls `load_config()` and `make_sim_params()`

3. **Source Loading**: Loads particle sources from schedule files (if any)

4. **System Setup**: Calls `allocate_system()` (with extra particles for sources), `load_geometry()`, `init_simulation()`, `init_source_particles_inactive()`

5. **Simulation Loop**:
   ```
   for each timestep:
       1. emit_particles (inject new particles from sources)
       2. Launch reset_sampling_kernel (clear accumulators)
       3. Launch solve_cell_kernel (physics + accumulate sampling)
       4. Launch finalize_sampling_kernel (compute density & temperature)
       5. Call sort_particles (reorder by cell)
       6. Swap double buffers
       7. Optionally dump cells/particles (based on --vis settings)
   ```

5. **Final Dump**: Always dumps `cell.dat` to output directory for evaluation

6. **Cleanup**: Calls `free_system()`

---

### `sim_config.cu`
**YAML configuration loading.** Parses config files using yaml-cpp library.

---

### `simulation.cu`
**System allocation and initialization.**

- `allocate_system(p_sys, c_sys, cfg, extra_particles)`: Allocates all GPU memory for particles and cells. The `extra_particles` parameter reserves space for source particles. Also initializes collision tracking arrays (`d_sigma_cr_max` with `INITIAL_SIGMA_CR_MAX`, `d_rng_collision` with curand states).
- `init_simulation(p_sys, c_sys, cfg, num_initial_particles)`: Initializes particle positions (avoiding solid objects) and velocities for the initial particles only.
- `init_source_particles_inactive(p_sys, init_particles, total_particles)`: Marks source particle slots as inactive (cell_id = -1).
- `init_collision_rng_kernel()`: CUDA kernel to initialize curand states for collision RNG.
- `free_system()`: Frees all GPU memory
- `is_inside_segment()`: Helper to check if a point is inside a solid object

---

### `kernels.cu`
**Physics kernel implementation.** Each CUDA block processes one cell independently.

**Execution Flow (per cell):**
1. **Load**: Copy particles from global → shared memory (coalesced)
2. **Sub-cell Indexing**: Assign particles to collision sub-cells (adaptive grid, ~4 particles per sub-cell)
3. **Collision**: NTC method with Hard Sphere model and fast LCG random number generation
4. **Sampling**: Accumulate velocity moments using shared memory reduction, write sums to global memory
5. **Movement**: Update positions using `p += v * dt`
6. **Segment Collision**: If cell has a segment, check ray-segment intersection and apply specular reflection
7. **Boundary**: Reflective walls (bounce particles back into domain)
8. **Re-locate**: Calculate new cell ID based on updated position
9. **Store**: Write back to global memory

**Collision Implementation (NTC Method):**
- **Sub-cell indexing**: Particles sorted into sub-cells for nearest-neighbor selection
- **Adaptive sub-cell grid**: `N_sc ≈ N_p/4` sub-cells (targets ~4 particles per sub-cell)
- **Expected collisions**: `M_coll = 0.5 × N × (N-1) × Fnum × (σ×c_r)_max × dt / V_subcell`
- **Accept/reject**: Probability `P = (σ×c_r) / (σ×c_r)_max`
- **Post-collision**: Isotropic scattering in center-of-mass frame
- **Fast LCG RNG**: Uses lightweight Linear Congruential Generator instead of curand for performance
- Each thread handles one sub-cell to avoid data races

**Sampling Implementation:**
- Uses shared memory reduction to minimize atomic operations
- Accumulates: `Σvx`, `Σvy`, `Σ(vx²+vy²)`
- `finalize_sampling_kernel` computes:
  - **Density**: `n = N_sim × Fnum / V_cell`
  - **Temperature**: `T = m⟨c²⟩/(2k_B)` where `⟨c²⟩ = ⟨v²⟩ - |⟨v⟩|²` (2D)

**Key Implementation Details:**
- Uses `__shared__` arrays for particle data (reduces global memory latency)
- Calculates new `cell_id` from position: `cell = cy * grid_nx + cx`
- Clamps positions and cell indices to valid domain
- Segment collision uses parametric ray-segment intersection with specular reflection: `v' = v - 2(v·n)n`
- Fast LCG RNG: `state = state * 1664525 + 1013904223` (Numerical Recipes parameters)

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

### `source.cu`
**Particle source/emitter implementation.** Allows injecting particles during simulation based on a schedule.

**Key Features:**
- **Schedule-based emission**: Particles are emitted at specific timesteps with specified counts
- **Maxwellian flux distribution**: Velocities sampled from proper flux distribution for inflow boundaries
- **Pre-allocation**: Total source particles are pre-allocated as inactive (cell_id = -1) at startup
- **Multiple sources**: Up to 16 sources can be configured simultaneously

**Velocity Sampling:**
- **Normal component**: Rayleigh distribution (flux-weighted) + bulk velocity
- **Tangential component**: Standard Maxwellian (normal distribution)
- **Z component**: Standard Maxwellian

**Source Config File Format (YAML):**
```yaml
# Total particles (must match schedule sum)
total_particles: 1000

# Line segment where particles spawn
geometry:
  start_x: 0.0
  start_y: 0.1
  end_x: 0.0
  end_y: 0.9

# Emission direction (normalized automatically)
direction:
  x: 1.0
  y: 0.0

# Velocity distribution parameters
velocity:
  temperature: 432.4    # Temperature in Kelvin for thermal velocity sampling
  stream_x: 500.0       # OR bulk_velocity: <m/s>
  stream_y: 0.0
```

**Schedule File Format (.dat):**
```
# Comments start with #
# timestep count
0 100
10 100
20 100
...
```

**Source Config Parameters:**
| Key | Description |
|-----|-------------|
| `total_particles` | Sum of all particles to be emitted |
| `geometry.start_x/y` | Segment start point |
| `geometry.end_x/y` | Segment end point |
| `direction.x/y` | Emission direction (normalized automatically) |
| `velocity.temperature` | Temperature in Kelvin for thermal velocity sampling |
| `velocity.stream_x/y` | Mean stream velocity components (2D) |
| `velocity.bulk_velocity` | Alternative to stream velocity projection |

**Inactive Particles:**
- Source particles are pre-allocated with `cell_id = INACTIVE_CELL_ID (-1)`
- When emitted, `cell_id` is set to the appropriate cell
- Inactive particles are sorted to the end of arrays and skipped in physics

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
**Debug dump implementation.** Copies GPU data to host and writes ASCII files.

**Internal Helpers:**
- `dump_cell_impl()`: Dumps cell data to file
- `dump_particle_impl()`: Dumps particle data to file

**Output Files:**
- `{timestep}-cell.dat`: Cell ID, particle count, offset, density, temperature
- `{timestep}-particle.dat`: Particle ID, position (x,y), velocity (x,y,z), species, cell ID (only if `--vis-particle`)
- `cell.dat`: Final cell state (mandatory output for evaluation)

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

### `draw_result.py`
Python script to draw cell statistics as heatmaps:

```bash
python scripts/draw_result.py -i outputs/test -c config.yaml [options]
```

**Options:**
| Flag | Description |
|------|-------------|
| `-i, --input` | Input directory with `cell.dat` or path to file directly |
| `-o, --output` | Output image filename (default: `<input>/heatmaps.png`) |
| `-c, --config` | Config YAML for grid dimensions |
| `-g, --geometry` | Geometry file to overlay solid objects |
| `--log-scale` | Use logarithmic scale for density |
| `--show` | Display plot interactively |
| `--dpi` | Image resolution (default: 150) |

**Output:**
- 3-panel figure with **Density**, **Temperature**, and **Particle Count** heatmaps
- Solid object boundaries drawn as white lines
- Statistics summary (min/max/mean) at bottom

---

### `vis_map.py`
Python script to create animated GIFs of density and temperature heatmaps (uses PyTorch for acceleration):

```bash
python scripts/vis_map.py -i outputs/test/visualization -c config.yaml [options]
```

**Options:**
| Flag | Description |
|------|-------------|
| `-i, --input` | Input directory with cell dump files (*-cell.dat) |
| `-o, --output` | Output directory for GIFs (default: same as input) |
| `-c, --config` | Config YAML for grid dimensions |
| `-g, --geometry` | Geometry file to overlay solid objects |
| `--downsample` | Downsample factor (e.g., 2 = halve resolution) |
| `--fps` | Frames per second (default: 10) |
| `--dpi` | Image resolution (default: 100) |
| `--log-scale` | Use logarithmic scale for density |
| `--device` | PyTorch device: `cuda` or `cpu` |

**Output:**
- `density.gif`: Animated density heatmap
- `temperature.gif`: Animated temperature heatmap

---

### `schedule.py`
Python script to generate particle emission schedules:

```bash
python scripts/schedule.py -c source_config.yaml -o schedule.dat --start 0 --end 5000 --profile uniform
```

**Options:**
| Flag | Description |
|------|-------------|
| `-c, --config` | Source config YAML (to read total_particles) |
| `-o, --output` | Output schedule .dat file |
| `--start` | First timestep for emission |
| `--end` | Last timestep (exclusive) |
| `--profile` | Emission profile: `uniform`, `ramp_up`, `ramp_down`, `gaussian`, `burst` |
| `--burst-timesteps` | For burst profile: comma-separated timesteps |

**Profiles:**
- `uniform`: Equal particles per timestep
- `ramp_up`: Linearly increasing emission
- `ramp_down`: Linearly decreasing emission
- `gaussian`: Bell curve centered at midpoint
- `burst`: All particles at specified timesteps

---

## Build System

**CMake** with CUDA language support:

```bash
# Configure
cmake -B build/Release -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build/Release

# Run (basic)
./build/Release/dsmc_solver -c assets/testcases/case-00/config.yaml -o outputs/test

# Run with visualization
./build/Release/dsmc_solver -c assets/testcases/case-00/config.yaml --vis -o outputs/test

# Run with geometry and particle sources
./build/Release/dsmc_solver -c assets/testcases/case-00/config.yaml \
    -g assets/testcases/case-00/geometry/circle.dat \
    -s assets/testcases/case-00/source/left.yaml \
    -S assets/testcases/case-00/source/schedule.dat \
    --vis --vis-particle -o outputs/test
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
