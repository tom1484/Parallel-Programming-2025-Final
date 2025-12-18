# Memory Usage Analysis

This document details the shared memory and global memory usage of the DSMC solver to help configure simulation parameters.

---

## 1. Shared Memory Usage (Per Block)

The `solve_cell_kernel` uses shared memory to hold particle data for one cell. Each CUDA block processes exactly one cell.

### 1.1 Shared Memory Breakdown

| Variable | Type | Size per Element | Count | Total Bytes |
|----------|------|------------------|-------|-------------|
| `s_pos[MAX_PARTICLES_PER_CELL]` | `double2` | 16 bytes | 128 | **2048 bytes** |
| `s_vel[MAX_PARTICLES_PER_CELL]` | `float2` | 8 bytes | 128 | **1024 bytes** |
| `s_species[MAX_PARTICLES_PER_CELL]` | `int` | 4 bytes | 128 | **512 bytes** |
| `s_subcell[MAX_PARTICLES_PER_CELL]` | `int` | 4 bytes | 128 | **512 bytes** |
| `s_segment` | `Segment` | 32 bytes | 1 | **32 bytes** |
| `s_num_particles` | `int` | 4 bytes | 1 | **4 bytes** |
| `s_vel_sum_x` | `float` | 4 bytes | 1 | **4 bytes** |
| `s_vel_sum_y` | `float` | 4 bytes | 1 | **4 bytes** |
| `s_vel_sq_sum` | `float` | 4 bytes | 1 | **4 bytes** |

### **Total Shared Memory per Block: 4,144 bytes (~4.0 KB)**

### 1.2 Hardware Limits

| GPU Architecture | Shared Memory per SM | Max Shared per Block | Blocks per SM (at 4KB) |
|------------------|---------------------|----------------------|------------------------|
| Volta (V100) | 96 KB | 48 KB | ~24 |
| Ampere (A100) | 164 KB | 48-164 KB | ~40 |
| Ada (RTX 4090) | 100 KB | 48-100 KB | ~25 |

**Current setting:** `SHARED_MEM_PER_BLOCK = 6144` in `config.h` (conservative estimate)

### 1.3 Scaling with MAX_PARTICLES_PER_CELL

If you change `MAX_PARTICLES_PER_CELL`, the shared memory scales as:

```
Shared Memory ≈ MAX_PARTICLES_PER_CELL × 32 bytes + 48 bytes
```

| MAX_PARTICLES_PER_CELL | Shared Memory | Notes |
|------------------------|---------------|-------|
| 64 | ~2.1 KB | Low density simulations |
| 128 (current) | ~4.1 KB | Default setting |
| 256 | ~8.2 KB | High density, still fits most GPUs |
| 512 | ~16.4 KB | Very high density |
| 1024 | ~32.8 KB | Near 48KB limit |

---

## 2. Global Memory Usage

### 2.1 Particle System (scales with total particles)

Let `N` = total particles (including source buffer)

| Array | Type | Size per Particle | Total Size |
|-------|------|-------------------|------------|
| `d_pos` | `double2` | 16 bytes | N × 16 |
| `d_vel` | `float2` | 8 bytes | N × 8 |
| `d_species` | `int` | 4 bytes | N × 4 |
| `d_cell_id` | `int` | 4 bytes | N × 4 |
| `d_sub_id` | `int` | 4 bytes | N × 4 |
| `d_pos_sorted` | `double2` | 16 bytes | N × 16 |
| `d_vel_sorted` | `float2` | 8 bytes | N × 8 |
| `d_species_sorted` | `int` | 4 bytes | N × 4 |

**Total per particle: 64 bytes** (with double buffering for sorting)

### 2.2 Cell System (scales with grid size)

Let `C` = total cells = `grid_nx × grid_ny`

| Array | Type | Size per Cell | Total Size |
|-------|------|---------------|------------|
| `d_density` | `float` | 4 bytes | C × 4 |
| `d_temperature` | `float` | 4 bytes | C × 4 |
| `d_vel_sum_x` | `float` | 4 bytes | C × 4 |
| `d_vel_sum_y` | `float` | 4 bytes | C × 4 |
| `d_vel_sq_sum` | `float` | 4 bytes | C × 4 |
| `d_cell_particle_count` | `int` | 4 bytes | C × 4 |
| `d_cell_offset` | `int` | 4 bytes | C × 4 |
| `d_write_offsets` | `int` | 4 bytes | C × 4 |
| `d_segments` | `Segment` | 32 bytes | C × 32 |

**Total per cell: 64 bytes**

Plus fixed allocations:
- `d_inactive_write_idx`: 4 bytes
- `d_temp_storage` (CUB): ~C bytes (for prefix sum)

### 2.3 Memory Formulas

```
Particle Memory (bytes) = N × 64
Cell Memory (bytes)     = C × 65  (approx, including CUB temp)
Total GPU Memory        = N × 64 + C × 65
```

---

## 3. Memory Usage Examples

### Example 1: Small Simulation
- Grid: 100 × 100 = 10,000 cells
- Particles: 1,000,000

```
Particle Memory = 1,000,000 × 64 = 64 MB
Cell Memory     = 10,000 × 65    = 0.65 MB
Total           ≈ 65 MB
```

### Example 2: Medium Simulation
- Grid: 500 × 500 = 250,000 cells
- Particles: 10,000,000

```
Particle Memory = 10,000,000 × 64 = 640 MB
Cell Memory     = 250,000 × 65    = 16 MB
Total           ≈ 656 MB
```

### Example 3: Large Simulation
- Grid: 1000 × 1000 = 1,000,000 cells
- Particles: 100,000,000

```
Particle Memory = 100,000,000 × 64 = 6.4 GB
Cell Memory     = 1,000,000 × 65   = 65 MB
Total           ≈ 6.5 GB
```

---

## 4. Constraints and Recommendations

### 4.1 Particles per Cell Constraint

The kernel uses `MAX_PARTICLES_PER_CELL = 128` for shared memory arrays. If a cell has more particles, they will overflow.

**Recommendation:**
- Average particles per cell: aim for 20-60 (allows for fluctuations)
- Peak particles per cell: should not exceed `MAX_PARTICLES_PER_CELL`

```
Safe particle count ≈ num_cells × 60
```

### 4.2 Grid Size Constraint (Occupancy)

Each cell requires one CUDA block. For good occupancy:
- Minimum cells: ~1000 (to fully utilize GPU)
- Maximum cells: No hard limit, but more cells = more kernel launches

### 4.3 Memory Bandwidth Considerations

The `solve_cell_kernel` loads/stores per cell:
- Load: `s_num_particles × 28 bytes` (pos + vel + species)
- Store: `s_num_particles × 28 bytes`

Total memory traffic per timestep:
```
Traffic ≈ 2 × N × 28 bytes = N × 56 bytes (for physics kernel)
        + N × 64 bytes (for sorting scatter - random writes)
        ≈ N × 120 bytes per timestep
```

---

## 5. Configuration Quick Reference

### Current Constants (`config.h`)

```cpp
#define THREADS_PER_BLOCK 64
#define SHARED_MEM_PER_BLOCK 6144    // Conservative estimate (actual: ~4144)
#define MAX_PARTICLES_PER_CELL 128
#define MAX_SUB_CELLS 36
```

### Type Sizes

```cpp
typedef double2 PositionType;  // 16 bytes
typedef float2 VelocityType;   // 8 bytes
struct Segment { ... };        // 32 bytes
```

### Tuning Guidelines

| Parameter | Increase if... | Decrease if... |
|-----------|---------------|----------------|
| `MAX_PARTICLES_PER_CELL` | Cells overflow (high density) | Shared memory limits |
| `grid_nx`, `grid_ny` | Need finer resolution | Memory constrained |
| `particle_weight` | Too many particles | Need more statistical accuracy |

---

## 6. How to Check Current Usage

Run with CUDA profiling to see actual memory usage:

```bash
# Memory allocation summary
nsys profile --stats=true ./dsmc_solver -c config.yaml -o outputs/test

# Detailed kernel memory analysis
ncu --set full -k solve_cell_kernel ./dsmc_solver -c config.yaml -o outputs/test
```

Or add to your code:
```cpp
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("GPU Memory: %.2f MB free / %.2f MB total\n", 
       free_mem / 1e6, total_mem / 1e6);
```
