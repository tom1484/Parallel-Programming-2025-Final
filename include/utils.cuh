#pragma once
#include <stdio.h>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

// Simple host-side prefix sum (exclusive scan)
// In production, use CUB or a parallel scan kernel for performance
inline void host_prefix_sum(int* input, int* output, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        output[i] = sum;
        sum += input[i];
    }
}