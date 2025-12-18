# Adaptive Cell Division - Implementation Plan

This document outlines the design considerations and implementation plan for adding adaptive cell division to the DSMC solver, based on the algorithm described in [ALGORITHM.md](ALGORITHM.md).

---

## Overview

The adaptive mesh allows cells with too many particles to subdivide into 4 children, maintaining the shared memory constraint (~128 particles/cell). This implementation uses **Option A: Pre-allocated Maximum Capacity** to avoid runtime memory allocation.

### Key Design Decisions

- **Pre-allocated buffers**: All cell arrays sized to `max_cells` at startup
- **Flat array hierarchy**: All cells (parents + children) in single contiguous array
- **Host-side subdivision**: GPU detects overcrowded cells, CPU performs subdivision
- **No coarsening** (initially): Once subdivided, cells remain subdivided

---

## Phase 1: Data Structure Changes

### 1.1 Cell Hierarchy Representation

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Flat Array** ✓ | All cells (parents + children) in single array, indexed by ID | Simple indexing, cache-friendly | Fragmented after many subdivisions |
| B. Level-separated Arrays | Separate array per refinement level | Easy to iterate by level | More pointers, complex addressing |
| C. Quadtree Pointers | Each cell has explicit child pointers | Intuitive traversal | Pointer chasing, worse GPU performance |

**Selected**: **A (Flat Array)** - best for GPU memory access patterns

### 1.2 Cell Bounds Storage

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Explicit Bounds** ✓ | Store `x_min, x_max, y_min, y_max` per cell | Fast access, no computation | 16 bytes extra per cell |
| B. Computed from Level | Store only `level` + base cell ID, compute bounds | Memory efficient | Requires computation each access |
| C. Hybrid | Store bounds only for non-base-level cells | Balance of memory and speed | More complex logic |

**Selected**: **A (Explicit Bounds)** - memory is cheap, computation is expensive on GPU

### 1.3 Leaf Cell Tracking

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Separate Leaf List** ✓ | Maintain `d_leaf_cell_ids[]` array | O(1) kernel launch setup | Must rebuild on subdivision |
| B. Flag in Cell Struct | `is_leaf` flag, scan to find leaves | No separate array | O(n) scan each frame |
| C. Compacted List + Dirty Flag | Rebuild leaf list only when changed | Amortized efficiency | Complexity in tracking changes |

**Selected**: **A (Separate Leaf List)** - subdivision is rare, rebuild cost is acceptable

### Proposed Data Structures

```cpp
// New structure for adaptive cells
struct AdaptiveCell {
    int parent_id;           // -1 for root cells
    int children[4];         // -1 if leaf cell, otherwise child IDs
    int level;               // Refinement level (0 = base grid)
    int is_leaf;             // 1 if this cell processes particles
    
    // Geometry bounds (needed because children have different sizes)
    float x_min, x_max;
    float y_min, y_max;
};

struct AdaptiveCellSystem {
    AdaptiveCell* d_cells;   // Cell hierarchy
    int* d_leaf_cell_ids;    // List of active leaf cells (for kernel launch)
    int num_leaf_cells;      // Number of active leaf cells
    int num_cells;           // Total cells allocated (base + subdivided)
    int max_cells;           // Pre-allocated capacity
    int max_levels;          // Maximum refinement depth (e.g., 4)
};
```

---

## Phase 2: Buffer Allocation Strategy

### 2.1 Maximum Cell Capacity

| Sub-option | Description | Multiplier | Memory Impact |
|------------|-------------|------------|---------------|
| A. Conservative 2x | `max_cells = base_cells * 2` | 2x | Low overhead |
| B. Moderate 4x | `max_cells = base_cells * 4` | 4x | Medium overhead |
| C. Full Depth | `max_cells = base_cells * 4^max_levels` | Up to 256x | High overhead |
| **D. Configurable** ✓ | User specifies in config YAML | Variable | Flexible |

**Selected**: **D (Configurable)** with default of **4x** - allows tuning per simulation

### 2.2 Particle Array Sizing

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Unchanged** ✓ | Keep current particle allocation | No changes needed | May run out if particles concentrate |
| B. Add Safety Margin | Allocate 1.5x expected particles | Handles moderate concentration | Fixed overhead |
| C. Dynamic Growth | Reallocate if exceeded | Most flexible | Runtime allocation (violates Option A) |

**Selected**: **A (Unchanged)** - subdivision handles concentration, total particles unchanged

### 2.3 Per-Cell Data Arrays

| Sub-option | Description | Arrays Affected |
|------------|-------------|-----------------|
| **A. All Arrays Max Size** ✓ | Every per-cell array uses `max_cells` | `d_density`, `d_temperature`, `d_vel_sum_*`, `d_cell_particle_count`, `d_cell_offset`, `d_segments` |
| B. Selective Sizing | Only sorting arrays need max size, sampling arrays use leaf count | Reduces memory but adds complexity |

**Selected**: **A (All Arrays Max Size)** - simpler, consistent

### Proposed Allocation

```cpp
// Pre-allocate for worst-case subdivision
void allocate_adaptive_system(AdaptiveCellSystem& ac_sys, 
                               CellSystem& c_sys,
                               const SimConfig& cfg) {
    int base_cells = cfg.grid_nx * cfg.grid_ny;
    ac_sys.max_cells = base_cells * cfg.cell_multiplier;  // Default: 4x
    ac_sys.max_levels = cfg.max_refinement_levels;        // Default: 4
    
    cudaMalloc(&ac_sys.d_cells, ac_sys.max_cells * sizeof(AdaptiveCell));
    cudaMalloc(&ac_sys.d_leaf_cell_ids, ac_sys.max_cells * sizeof(int));
    
    // CellSystem arrays also use max_cells size
    cudaMalloc(&c_sys.d_density, ac_sys.max_cells * sizeof(float));
    cudaMalloc(&c_sys.d_temperature, ac_sys.max_cells * sizeof(float));
    // ... etc
}
```

---

## Phase 3: Cell ID Calculation Changes

### 3.1 Hierarchy Traversal Method

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Top-down Traversal** ✓ | Start at base cell, walk down to leaf | Always correct | O(depth) traversal |
| B. Lookup Table | Pre-compute fine grid → leaf cell mapping | O(1) lookup | Large table if deep subdivision |
| C. Morton Code | Use space-filling curve for locality | Cache efficient | Complex implementation |

**Selected**: **A (Top-down Traversal)** - depth is limited (max 4), simple to implement

### 3.2 Base Cell Identification

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Always Compute** ✓ | `base_id = floor(y/dy) * nx + floor(x/dx)` | No storage needed | Repeated computation |
| B. Store in AdaptiveCell | Each cell knows its base ancestor | Fast lookup | 4 bytes per cell |

**Selected**: **A (Always Compute)** - base cell calculation is trivial

### Proposed Implementation

```cpp
// Traverse hierarchy to find leaf cell
__device__ int get_leaf_cell_id(float x, float y, 
                                 const AdaptiveCell* cells,
                                 const SimParams& params) {
    // Start at base cell
    int cx = (int)(x / params.cell_dx);
    int cy = (int)(y / params.cell_dy);
    cx = max(0, min(cx, params.grid_nx - 1));
    cy = max(0, min(cy, params.grid_ny - 1));
    int cell_id = cy * params.grid_nx + cx;
    
    // Traverse down to leaf
    while (!cells[cell_id].is_leaf) {
        // Determine which quadrant
        float mid_x = (cells[cell_id].x_min + cells[cell_id].x_max) * 0.5f;
        float mid_y = (cells[cell_id].y_min + cells[cell_id].y_max) * 0.5f;
        int quad = (x >= mid_x) + 2 * (y >= mid_y);
        cell_id = cells[cell_id].children[quad];
    }
    return cell_id;
}
```

---

## Phase 4: Sorting Pipeline Changes

### 4.1 Histogram Array Sizing

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Use Max Cells** ✓ | `d_cell_particle_count[max_cells]` | Always fits | Wastes memory if few subdivisions |
| B. Use Current Leaf Count | Resize histogram to `num_leaf_cells` | Memory efficient | Need to track size, potential bugs |

**Selected**: **A (Use Max Cells)** - consistent with Phase 2 decisions

### 4.2 Prefix Sum Scope

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Full Array** ✓ | Prefix sum over `max_cells` | Simple, always correct | Wasted computation on empty cells |
| B. Leaf-only Compacted | Only sum over active leaf cells | Efficient | Need indirection layer |
| C. Segmented by Level | Separate prefix sum per level | Allows level-specific processing | Complex orchestration |

**Selected**: **A (Full Array)** - prefix sum is fast, simplicity wins

### 4.3 Cell ID in Particles

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Leaf Cell ID** ✓ | `d_cell_id` stores leaf cell index | Direct sorting | Must update on subdivision |
| B. Base + Local | Store base cell + sub-index | Stable across subdivision | Two-level indirection |

**Selected**: **A (Leaf Cell ID)** - simpler, matches current design

---

## Phase 5: Subdivision/Coarsening Logic

### 5.1 Subdivision Trigger

| Sub-option | Description | Threshold |
|------------|-------------|-----------|
| A. Hard Limit | Subdivide if `count > MAX_PARTICLES_PER_CELL` | 128 |
| **B. Soft Limit with Hysteresis** ✓ | Subdivide at 120, coarsen at 60 | Prevents oscillation |
| C. Adaptive Threshold | Based on local collision rate | Physics-aware but complex to tune |

**Selected**: **B (Soft Limit)** - prevents subdivision/coarsening thrashing

### 5.2 When to Check Subdivision

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| A. Every Timestep | Check after every sort | Always optimal | Overhead every frame |
| **B. Periodic** ✓ | Check every N timesteps | Reduced overhead | May exceed limit temporarily |
| C. Triggered | Check only if max count exceeded | Minimal overhead | Need tracking mechanism |

**Selected**: **B (Periodic)** with N=10-100

### 5.3 Subdivision Execution Location

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Host-side** ✓ | GPU detects, CPU performs subdivision | Simple control flow | Host-device sync |
| B. Device-side | Fully GPU-based subdivision | No sync needed | Complex atomic operations |
| C. Hybrid | GPU marks candidates, CPU allocates, GPU updates | Balance | Two-phase |

**Selected**: **A (Host-side)** - subdivision is rare, sync cost acceptable

### 5.4 Coarsening Policy

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. No Coarsening** ✓ | Once subdivided, stay subdivided | Simplest | Memory/cells grow monotonically |
| B. Immediate Coarsening | Coarsen when all 4 siblings below threshold | Optimal cell count | Thrashing risk |
| C. Delayed Coarsening | Coarsen after N timesteps below threshold | Stable | More state to track |
| D. Periodic Cleanup | Batch coarsening every M timesteps | Reduced overhead | Suboptimal between cleanups |

**Selected**: **A (No Coarsening)** initially - add **C** later if memory is an issue

### Proposed Implementation

```cpp
// Kernel to detect overcrowded cells
__global__ void check_subdivision_kernel(
    const int* d_cell_particle_count,
    const int* d_leaf_cell_ids,
    const AdaptiveCell* d_cells,
    int* d_needs_subdivision,  // Output: cells that need splitting
    int* d_subdivision_count,
    int num_leaf_cells,
    int max_particles_threshold,  // e.g., 120
    int max_level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaf_cells) return;
    
    int cell_id = d_leaf_cell_ids[idx];
    if (d_cell_particle_count[cell_id] > max_particles_threshold &&
        d_cells[cell_id].level < max_level) {
        int pos = atomicAdd(d_subdivision_count, 1);
        d_needs_subdivision[pos] = cell_id;
    }
}

// Host-side subdivision
void perform_subdivision(AdaptiveCellSystem& ac_sys, 
                         CellSystem& c_sys,
                         int* h_needs_subdivision, 
                         int count) {
    for (int i = 0; i < count; i++) {
        int parent_id = h_needs_subdivision[i];
        AdaptiveCell& parent = h_cells[parent_id];
        
        // Allocate 4 new children
        int first_child = ac_sys.num_cells;
        if (first_child + 4 > ac_sys.max_cells) {
            printf("Warning: max cells exceeded, skipping subdivision\n");
            continue;
        }
        ac_sys.num_cells += 4;
        
        float mid_x = (parent.x_min + parent.x_max) * 0.5f;
        float mid_y = (parent.y_min + parent.y_max) * 0.5f;
        
        // Initialize 4 children (quadrants: BL, BR, TL, TR)
        float x_bounds[4][2] = {
            {parent.x_min, mid_x}, {mid_x, parent.x_max},
            {parent.x_min, mid_x}, {mid_x, parent.x_max}
        };
        float y_bounds[4][2] = {
            {parent.y_min, mid_y}, {parent.y_min, mid_y},
            {mid_y, parent.y_max}, {mid_y, parent.y_max}
        };
        
        for (int q = 0; q < 4; q++) {
            int child_id = first_child + q;
            h_cells[child_id].parent_id = parent_id;
            h_cells[child_id].level = parent.level + 1;
            h_cells[child_id].is_leaf = 1;
            h_cells[child_id].x_min = x_bounds[q][0];
            h_cells[child_id].x_max = x_bounds[q][1];
            h_cells[child_id].y_min = y_bounds[q][0];
            h_cells[child_id].y_max = y_bounds[q][1];
            for (int c = 0; c < 4; c++) h_cells[child_id].children[c] = -1;
            
            // Inherit segment from parent
            h_segments[child_id] = h_segments[parent_id];
            
            parent.children[q] = child_id;
        }
        
        // Mark parent as non-leaf
        parent.is_leaf = 0;
    }
    
    // Rebuild leaf list and upload to GPU
    rebuild_leaf_list(ac_sys);
    upload_cells_to_gpu(ac_sys, c_sys);
}
```

---

## Phase 6: Kernel Launch Changes

### 6.1 Block-to-Cell Mapping

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| A. Direct Indexing | Block `i` → `d_leaf_cell_ids[i]` | Simple, clear | Indirection cost |
| **B. Compacted Launch** ✓ | Launch only `num_leaf_cells` blocks | No wasted blocks | Must update launch config |

**Selected**: **B (Compacted Launch)** - this is the whole point of adaptive mesh

### 6.2 Cell Bounds Access in Kernel

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Load from Global** ✓ | Each block loads its cell bounds from `d_cells` | Simple | One extra global read |
| B. Pass via Shared Memory | Load bounds into shared mem first | Faster subsequent access | Uses shared memory |
| C. Constant Memory | Store active cell bounds in constant memory | Very fast | Limited size (64KB) |

**Selected**: **A (Load from Global)** - one read per block is negligible

### 6.3 Particle Range Determination

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. From Prefix Sum** ✓ | Use `d_cell_offset[cell_id]` and count | Existing infrastructure | Works directly |
| B. Stored in Cell Struct | Each cell stores `particle_start`, `particle_count` | Locality | Redundant storage |

**Selected**: **A (From Prefix Sum)** - reuse existing sorting results

### Proposed Kernel Changes

```cpp
// Launch with dynamic grid size
solve_cell_kernel<<<num_leaf_cells, THREADS_PER_BLOCK>>>(
    p_sys, c_sys, ac_sys, params
);

// Modified kernel signature
__global__ void solve_cell_kernel(
    ParticleSystem p_sys,
    CellSystem c_sys,
    AdaptiveCellSystem ac_sys,
    SimParams params
) {
    int leaf_idx = blockIdx.x;
    int cell_id = ac_sys.d_leaf_cell_ids[leaf_idx];
    AdaptiveCell cell = ac_sys.d_cells[cell_id];
    
    // Cell dimensions from adaptive bounds
    float cell_dx = cell.x_max - cell.x_min;
    float cell_dy = cell.y_max - cell.y_min;
    
    // Get particle range from sorting results
    int start = c_sys.d_cell_offset[cell_id];
    int count = c_sys.d_cell_particle_count[cell_id];
    
    // Load particles into shared memory
    // ... (similar to current implementation)
    
    // Use cell.x_min, x_max, y_min, y_max for boundary checks
    // ... rest of kernel
}
```

---

## Phase 7: Geometry Integration

### 7.1 Segment Storage Location

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Per-Cell Segment** ✓ | Each leaf cell has its own segment copy | Simple lookup | Memory for deep subdivision |
| B. Segment Pool | Global segment array, cells reference by ID | Memory efficient | Indirection |
| C. Base Cell Only | Store with base cell, children inherit | Minimal storage | Must traverse to base |

**Selected**: **A (Per-Cell Segment)** - consistent with current design

### 7.2 Segment Inheritance on Subdivision

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Copy to All Children** ✓ | All 4 children get parent's segment | Simple | May have segment in cells it doesn't cross |
| B. Geometric Intersection | Check which children segment actually crosses | Accurate | Complex geometry math |
| C. Conservative Copy + Flag | Copy to all, mark `segment_relevant` flag | Balance | Extra flag per cell |

**Selected**: **A (Copy to All Children)** - segment collision check already handles miss case

### 7.3 Inside Flag Propagation

| Sub-option | Description | Pros | Cons |
|------------|-------------|------|------|
| **A. Inherit from Parent** ✓ | Children inherit `inside` flag | Simple, correct | None |
| B. Recompute | Check if child center is inside geometry | More accurate for edge cases | Expensive |

**Selected**: **A (Inherit from Parent)** - if parent is inside, all children are inside

---

## Summary of Selected Options

| Phase | Aspect | Selected Option |
|-------|--------|-----------------|
| 1.1 | Hierarchy Representation | Flat Array |
| 1.2 | Cell Bounds Storage | Explicit Bounds |
| 1.3 | Leaf Cell Tracking | Separate Leaf List |
| 2.1 | Maximum Cell Capacity | Configurable (default 4x) |
| 2.2 | Particle Array Sizing | Unchanged |
| 2.3 | Per-Cell Data Arrays | All Arrays Max Size |
| 3.1 | Hierarchy Traversal | Top-down Traversal |
| 3.2 | Base Cell Identification | Always Compute |
| 4.1 | Histogram Array Sizing | Use Max Cells |
| 4.2 | Prefix Sum Scope | Full Array |
| 4.3 | Cell ID in Particles | Leaf Cell ID |
| 5.1 | Subdivision Trigger | Soft Limit with Hysteresis |
| 5.2 | When to Check | Periodic (every N steps) |
| 5.3 | Subdivision Execution | Host-side |
| 5.4 | Coarsening Policy | No Coarsening (initially) |
| 6.1 | Block-to-Cell Mapping | Compacted Launch |
| 6.2 | Cell Bounds Access | Load from Global |
| 6.3 | Particle Range | From Prefix Sum |
| 7.1 | Segment Storage | Per-Cell Segment |
| 7.2 | Segment Inheritance | Copy to All Children |
| 7.3 | Inside Flag Propagation | Inherit from Parent |

---

## Implementation Order

| Step | Task | Complexity | Files Modified |
|------|------|------------|----------------|
| 1 | Add `AdaptiveCell` and `AdaptiveCellSystem` structures | Low | `data_types.h` |
| 2 | Add config options for `cell_multiplier`, `max_refinement_levels` | Low | `sim_config.h`, `sim_config.cu` |
| 3 | Pre-allocate adaptive buffers with max capacity | Medium | `simulation.cu` |
| 4 | Initialize base cells as leaf cells | Medium | `simulation.cu` |
| 5 | Implement `get_leaf_cell_id()` device function | Medium | `kernels.cu` |
| 6 | Modify sorting to use `max_cells` for all arrays | Medium | `sorting.cu` |
| 7 | Add `check_subdivision_kernel` | Medium | `kernels.cu` |
| 8 | Implement host-side `perform_subdivision()` | High | `simulation.cu` |
| 9 | Implement `rebuild_leaf_list()` | Medium | `simulation.cu` |
| 10 | Update `solve_cell_kernel` to use adaptive bounds | Medium | `kernels.cu` |
| 11 | Update kernel launches to use `num_leaf_cells` | Low | `main.cu` |
| 12 | Update geometry segment inheritance | Medium | `geometry.cu` |
| 13 | Update particle re-assignment after subdivision | Medium | `kernels.cu` |
| 14 | Add visualization support for adaptive cells | Low | `visualize.cu` |

---

## Configuration File Changes

```yaml
grid:
  nx: 10
  ny: 10
  lx: 0.3
  ly: 0.3

adaptive:
  enabled: true                    # Enable/disable adaptive mesh
  cell_multiplier: 4               # max_cells = base_cells * multiplier
  max_refinement_levels: 4         # Maximum subdivision depth
  subdivision_threshold: 120       # Subdivide if particles > threshold
  check_interval: 10               # Check for subdivision every N steps
```

---

## Future Enhancements

1. **Coarsening**: Add delayed coarsening when all 4 siblings are below threshold
2. **Load Balancing**: Track cell workload and rebalance if needed
3. **Geometry Splitting**: Properly clip segments to child cells
4. **Visualization**: Color cells by refinement level in debug output
5. **Statistics**: Report subdivision events and cell distribution
