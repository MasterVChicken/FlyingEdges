#include "MCTables.h"
#include <stdio.h>

template <typename T>
__global__ void blockScan(T* input, T* output, T* blockSums, int n) {
    __shared__ T sharedMem[1024];  

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (gid < n) {
        sharedMem[tid] = input[gid];
    } else {
        sharedMem[tid] = 0;
    }
    __syncthreads();

    // Perform scan (prefix sum) in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        T temp = 0;
        if (tid >= stride) {
            temp = sharedMem[tid - stride];
        }
        __syncthreads();
        sharedMem[tid] += temp;
        __syncthreads();
    }

    // Write the result back to global memory
    if (gid < n) {
        output[gid] = sharedMem[tid];
    }

    // Write the block sum for the global scan
    if (blockSums != nullptr && tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = sharedMem[tid];
    }
}

template <typename T>
__global__ void addBlockSums(T* output, T* blockSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < n) {
        output[gid] += blockSums[blockIdx.x - 1];
    }
}

template <typename T>
void prefixSumManual(T* d_input, T* d_output, int n) {
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;

    T* d_blockSums;
    cudaMalloc(&d_blockSums, gridSize * sizeof(T));

    // Perform block-level scan
    blockScan<<<gridSize, blockSize>>>(d_input, d_output, d_blockSums, n);

    // Perform a scan on the block sums
    if (gridSize > 1) {
        T* d_blockPrefixSums;
        cudaMalloc(&d_blockPrefixSums, gridSize * sizeof(T));
        blockScan<<<1, gridSize>>>(d_blockSums, d_blockPrefixSums, nullptr, gridSize);

        // Add the block prefix sums to each block's output
        addBlockSums<<<gridSize, blockSize>>>(d_output, d_blockPrefixSums, n);

        cudaFree(d_blockPrefixSums);
    }

    cudaFree(d_blockSums);
}

template <typename T>
__global__ void assignIDs(const T* __restrict__ prefixSums, T* __restrict__ ids, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        ids[idx] = prefixSums[idx];
    }
}

template <typename T>
void processPass3(const T* d_yEdgeIntersections,
                        const T* d_zEdgeIntersections,
                        const T* d_triangleCounts,
                        T* d_yEdgePrefixSums,
                        T* d_zEdgePrefixSums,
                        T* d_trianglePrefixSums,
                        T* d_yEdgeIDs,
                        T* d_zEdgeIDs,
                        T* d_triangleIDs,
                        int numRows) {
    
    // Get prefix sum for y axis intersections, z axis intersections and triangles
    prefixSumManual(d_yEdgeIntersections, d_yEdgePrefixSums, numRows);
    prefixSumManual(d_zEdgeIntersections, d_zEdgePrefixSums, numRows);
    prefixSumManual(d_triangleCounts, d_trianglePrefixSums, numRows);

    int blockSize = 256;
    int gridSize = (numRows + blockSize - 1) / blockSize;

    assignIDs<<<gridSize, blockSize>>>(d_yEdgePrefixSums, d_yEdgeIDs, numRows);
    assignIDs<<<gridSize, blockSize>>>(d_zEdgePrefixSums, d_zEdgeIDs, numRows);
    assignIDs<<<gridSize, blockSize>>>(d_trianglePrefixSums, d_triangleIDs, numRows);

    cudaDeviceSynchronize();
}