# Memory Access Pattern Analysis and Proposed Improvements

This document analyzes the memory access patterns in the DSMC GPU solver and proposes optimizations to improve performance.

---

## 1. Current Memory Access Pattern Analysis

### 1.1 `solve_cell_kernel` - Main Physics Kernel

The current implementation follows the paper's cell-based strategy, where each CUDA block processes one cell independently. However, there are several memory access inefficiencies:

#### **Loading Phase (Global → Shared)**
```cpp
// Current implementation
for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
    int global_idx = cell_start_idx + i;
    s_pos[i] = p_sys.d_pos[global_idx];      // 16 bytes (double2)
    s_vel[i] = p_sys.d_vel[global_idx];      // 12 bytes (float3)
    s_species[i] = p_sys.d_species[global_idx]; // 4 bytes
}
```

**Issues:**
1. **Non-coalesced access for `double2`**: Since particles are sorted by cell, threads in a warp access consecutive global indices, which is good. However, `double2` (16 bytes) requires 128-bit aligned transactions. If `cell_start_idx` is not 128-bit aligned, memory transactions become misaligned.

2. **Separate loads for each array**: Position, velocity, and species are loaded in separate memory transactions. This results in 3 separate global memory accesses per particle, increasing memory bandwidth consumption.

3. **`float3` padding issue**: `float3` is 12 bytes but CUDA aligns it to 16 bytes in arrays. This wastes 25% of memory bandwidth when loading velocities. Additionally, this is a **2D simulation** where `vz` is never used, making `float3` even more wasteful.

#### **Storing Phase (Shared → Global)**
```cpp
// Current implementation
for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
    int global_idx = cell_start_idx + i;
    p_sys.d_pos[global_idx] = s_pos[i];
    p_sys.d_vel[global_idx] = s_vel[i];
    p_sys.d_species[global_idx] = s_species[i];
}
```

Same issues as loading: multiple transactions and potential alignment problems.

#### **Segment Access**
```cpp
Segment seg = c_sys.d_segments[cell_idx];
```

**Issue:** Every thread in the block reads the same `Segment` from global memory. While L1 cache may help, using `__shared__` memory for the segment would be more efficient.

---

### 1.2 `reorder_particles_kernel` - Sorting Scatter

```cpp
// Current implementation
sys.d_pos_sorted[write_idx] = sys.d_pos[idx];
sys.d_vel_sorted[write_idx] = sys.d_vel[idx];
sys.d_species_sorted[write_idx] = sys.d_species[idx];
```

**Issues:**
1. **Random write pattern**: `write_idx` is determined by `atomicAdd`, causing completely random write access patterns. This is inherent to scatter operations but could be mitigated.

2. **Multiple stores per particle**: Three separate store operations increase memory traffic.

---

### 1.3 `count_particles_kernel` - Histogram

```cpp
atomicAdd(&d_counts[cell], 1);
```

**Issue:** Atomic contention when many particles in the same warp belong to the same cell. This is common in DSMC since particles are spatially clustered.

---

## 2. Proposed Improvements

### 2.1 Reduce Velocity to `float2` (2D Simulation)

**Problem:** The current implementation uses `float3` for velocity, but this is a 2D simulation where `vz` is never used. Additionally, `float3` wastes bandwidth due to 16-byte alignment (12 bytes data + 4 bytes padding).

**Solution:** Use `float2` for velocity:

```cpp
// In config.h - Replace VelocityType
typedef float2 VelocityType;  // Only vx, vy needed for 2D

// Benefits:
// - float2 is naturally 8-byte aligned, no wasted padding
// - Reduces velocity memory footprint by 33% (12 bytes → 8 bytes)
// - Two float2 loads can be combined into one 128-bit transaction
```

**Changes Required:**
- Update `VelocityType` typedef in `config.h`
- Update all velocity initialization (remove `.z` component)
- Update `solve_cell_kernel` sampling (remove `vz` accumulation)
- Update temperature calculation (2D: `T = m⟨c²⟩/(2k_B)` instead of 3D formula)

**Expected Benefit:** 
- Reduces velocity array memory traffic by ~33%
- Better memory alignment (8 bytes vs 12 bytes with padding to 16)
- Simplified sampling code

---

### 2.2 Use `__ldg()` for Read-Only Cache

**Problem:** Standard loads may evict useful data from L1 cache.

**Solution:** Use `__ldg()` intrinsic for read-only data:

```cpp
// Loading particle data (read-only during this kernel's load phase)
s_pos[i] = __ldg(&p_sys.d_pos[global_idx]);
s_vel[i] = __ldg(&p_sys.d_vel[global_idx]);
s_species[i] = __ldg(&p_sys.d_species[global_idx]);

// Cell system data
int cell_offset = __ldg(&c_sys.d_cell_offset[cell_idx]);
int cell_count = __ldg(&c_sys.d_cell_particle_count[cell_idx]);
```

**Limitation:** `__ldg()` only works with built-in CUDA types (int, float, double, float2, double2, etc.). Custom structs like `Segment` cannot use `__ldg()`. For such cases, use shared memory instead (see 2.3).

**Expected Benefit:**
- Uses texture cache path (separate from L1)
- Better cache utilization for streaming access patterns
- Simple single-line changes with no structural modifications

---

### 2.3 Shared Memory for Segment Data

**Problem:** All 64 threads read the same `Segment` from global memory.

**Solution:** Load segment once into shared memory:

```cpp
__shared__ Segment s_segment;

if (tid == 0) {
    s_segment = c_sys.d_segments[cell_idx];
}
__syncthreads();

// Later, use s_segment instead of reading from global memory
if (s_segment.exists) {
    float t;
    if (segment_intersection(..., s_segment.start_x, s_segment.start_y, ...)) {
        reflect_particle(p, v, s_segment, t);
    }
}
```

**Expected Benefit:**
- Reduces global memory reads by 63 per block (from 64 to 1)
- Segment struct is 32 bytes, so this saves ~2KB of bandwidth per block

---

### 2.4 Warp-Level Reduction for Sampling

**Problem:** Current implementation uses `atomicAdd` on shared memory for velocity sums.

**Solution:** Use warp shuffle intrinsics for faster reduction:

```cpp
// Per-thread accumulators (2D: no vz needed)
float local_vx_sum = 0.0f, local_vy_sum = 0.0f, local_vsq_sum = 0.0f;

for (int i = tid; i < s_num_particles; i += THREADS_PER_BLOCK) {
    // ... accumulate ...
}

// Warp-level reduction using shuffle
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_vx_sum += __shfl_down_sync(0xFFFFFFFF, local_vx_sum, offset);
    local_vy_sum += __shfl_down_sync(0xFFFFFFFF, local_vy_sum, offset);
    local_vsq_sum += __shfl_down_sync(0xFFFFFFFF, local_vsq_sum, offset);
}

// Only lane 0 of each warp writes to shared memory
int warp_id = tid / warpSize;
int lane = tid % warpSize;
if (lane == 0) {
    atomicAdd(&s_vel_sum_x, local_vx_sum);
    atomicAdd(&s_vel_sum_y, local_vy_sum);
    atomicAdd(&s_vel_sq_sum, local_vsq_sum);
}
```

**Expected Benefit:**
- Reduces atomic operations from 64 to 2 per block (THREADS_PER_BLOCK=64, warpSize=32)
- Warp shuffles are much faster than shared memory atomics

---

### 2.5 Privatized Histogram for Counting

**Problem:** `count_particles_kernel` has high atomic contention when particles cluster in cells.

**Solution:** Use per-block privatization with shared memory:

```cpp
__global__ void count_particles_kernel_optimized(
    const int* __restrict__ d_cell_id,
    int* __restrict__ d_counts,
    int num_particles, int num_cells
) {
    // Shared memory histogram (works if num_cells <= shared mem limit)
    extern __shared__ int s_counts[];
    
    // Initialize shared counts
    for (int i = threadIdx.x; i < num_cells; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();
    
    // Count into shared memory (less contention)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        int cell = d_cell_id[idx];
        if (cell != INACTIVE_CELL_ID) {
            atomicAdd(&s_counts[cell], 1);
        }
    }
    __syncthreads();
    
    // Merge to global (one atomic per cell per block)
    for (int i = threadIdx.x; i < num_cells; i += blockDim.x) {
        if (s_counts[i] > 0) {
            atomicAdd(&d_counts[i], s_counts[i]);
        }
    }
}
```

**Note:** This is only beneficial when `num_cells` fits in shared memory (~16K cells for 48KB shared memory). For larger grids, consider block-level binning.

**Expected Benefit:**
- Significantly reduces global atomic contention
- Most atomic operations happen in fast shared memory

---

### 2.6 Aligned Memory Allocation

**Problem:** `cell_start_idx` may not be aligned to cache line / vector load boundaries.

**Solution:** Ensure cell offsets are aligned to 16 bytes (for 128-bit transactions):

```cpp
// In sorting phase, pad cell offsets to 16-byte alignment
// This wastes some memory but ensures coalesced access

// Alternative: Use __align__ specifier during allocation
cudaMalloc(&p_sys.d_pos, buffer_size * sizeof(PositionType));
// Ensure base pointer is 128-byte aligned (usually automatic with cudaMalloc)
```

For practical implementation, accept minor misalignment unless profiling shows significant stall cycles.

---

### 2.7 Prefetching with `__ldg()` and Read-Only Cache

**Problem:** Standard loads may evict useful data from L1 cache.

**Solution:** Use `__ldg()` intrinsic for read-only data:

```cpp
// Loading particle data (read-only during this kernel)
s_pos[i] = __ldg(&p_sys.d_pos[global_idx]);
s_vel[i] = __ldg(&p_sys.d_vel[global_idx]);
s_species[i] = __ldg(&p_sys.d_species[global_idx]);

// Also for segment access
if (tid == 0) {
    // Use const restrict pointers in function signature for compiler hints
    s_segment = __ldg(&c_sys.d_segments[cell_idx]);
}
```

**Expected Benefit:**
- Uses texture cache path (separate from L1)
- Better cache utilization for streaming access patterns

---

### 2.8 Asynchronous Memory Copy with CUDA Streams

**Problem:** `cudaMemcpy` for offset initialization blocks GPU execution.

```cpp
// Current blocking implementation in sorting.cu
CHECK_CUDA(cudaMemcpy(&last_offset, &c_sys.d_cell_offset[num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost));
CHECK_CUDA(cudaMemcpy(&last_count, &c_sys.d_cell_particle_count[num_cells - 1], sizeof(int), cudaMemcpyDeviceToHost));
```

**Solution:** Use pinned memory and async copies, or compute on device:

```cpp
// Device-side calculation kernel
__global__ void calc_total_active_kernel(
    const int* d_cell_offset,
    const int* d_cell_particle_count,
    int* d_inactive_write_idx,
    int num_cells
) {
    // Single thread kernel
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_inactive_write_idx = d_cell_offset[num_cells - 1] + d_cell_particle_count[num_cells - 1];
    }
}

// Replace the two cudaMemcpy calls with single kernel launch
calc_total_active_kernel<<<1, 1>>>(c_sys.d_cell_offset, c_sys.d_cell_particle_count, 
                                    c_sys.d_inactive_write_idx, num_cells);
```

**Expected Benefit:**
- Eliminates host-device synchronization in hot loop
- Removes PCIe transfer overhead

---

## 3. Summary of Proposed Changes

| Improvement | File(s) | Complexity | Expected Speedup | Status |
|-------------|---------|------------|------------------|--------|
| 2.1 `float2` velocity (2D) | `config.h`, `kernels.cu`, `simulation.cu`, `source.cu` | Medium | 15-20% memory bandwidth | ✅ Done |
| 2.2 `__ldg()` for read-only loads | `kernels.cu` | Low | 5-10% kernel time | ✅ Done* |
| 2.3 Shared memory for segment | `kernels.cu` | Low | 2-5% kernel time | ✅ Done |
| 2.4 Warp shuffle reduction | `kernels.cu` | Medium | 5-10% sampling time | Pending |
| 2.5 Privatized histogram | `sorting.cu` | Medium | 10-20% counting time | Pending |
| 2.6 Device-side offset calc | `sorting.cu` | Low | Eliminates sync stall | ✅ Done |

*Note: `__ldg()` is applied to primitive types only (cell_offset, cell_particle_count, pos, vel, species). Custom structs like `Segment` don't support `__ldg()` intrinsics.

---

## 4. Implementation Priority

### High Priority (Low effort, high impact) - COMPLETED
1. **`float2` velocity** - ✅ Significant memory savings, 2D simulation doesn't need vz
2. **Shared memory for segment** - ✅ Simple change, immediate benefit
3. **Device-side offset calculation** - ✅ Eliminates host sync
4. **`__ldg()` intrinsics** - ✅ Applied to primitive type reads

### Medium Priority (Moderate effort) - PENDING
5. **Warp shuffle reduction** - Requires careful implementation
6. **Privatized histogram** - Conditional on grid size

---

## 5. Profiling Recommendations

Before implementing these changes, profile with:

```bash
# Memory bandwidth utilization
nsys profile --stats=true ./dsmc_solver -c config.yaml -o outputs/profile

# Detailed kernel analysis
ncu --set full -o profile_report ./dsmc_solver -c config.yaml -o outputs/profile
```

Key metrics to check:
- **Global Load/Store Efficiency** - Target > 80%
- **Shared Memory Bank Conflicts** - Target: 0
- **Warp Execution Efficiency** - Target > 90%
- **Memory Throughput** - Compare to theoretical peak

---

## 6. References

- NVIDIA CUDA Best Practices Guide: Memory Coalescing
- NVIDIA CUDA C++ Programming Guide: Shared Memory
- Original Paper: "A GPU–CUDA based DSMC algorithm" by M.J. Goldsworthy
