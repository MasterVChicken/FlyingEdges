#include "MCTables.h"

// caseNumber 4, 5, 6, 7 indicate the case has intersection on Y axis
__device__ bool hasYEdgeIntersection(int caseNumber) {
    int edgeIndices[] = {4, 5, 6, 7}; 
    for (int i = 0; i < 4; i++) {
        if (caseNumber == edgeIndices[i]) {
            return true;
        }
    }
    return false;
}

// caseNumber 8, 9, 10, 11 indicate the case has intersection on Z axis
__device__ bool hasZEdgeIntersection(int caseNumber) {
    int edgeIndices[] = {8, 9, 10, 11}; 
    for (int i = 0; i < 4; i++) {
        if (caseNumber == edgeIndices[i]) {
            return true;
        }
    }
    return false;
}


__device__ int getTriangleCount(int caseNumber) {
    return d_edgeCases[caseNumber][0]; 
}

// gridSize here refers to size of edges, not scalars
template <typename T>
__global__ void processPass2(const int* __restrict__ edgeCases,
                                const int* __restrict__ trimPositionsLeft,
                                const int* __restrict__ trimPositionsRight,
                                int* __restrict__ mcCases,
                                int* __restrict__ yEdgeIntersections,
                                int* __restrict__ zEdgeIntersections,
                                int* __restrict__ triangleCounts,
                                const T isoValue,
                                const int3 gridSize) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= gridSize.y - 1 || k >= gridSize.z - 1) return;

    // iterate each cell
    for (int i = trimPositionsLeft[j * gridSize.z + k]; i < trimPositionsRight[j * gridSize.z + k]; ++i) {
        int cellIndex = i + j * gridSize.x + k * gridSize.x * gridSize.y;

        int caseNumber = edgeCases[cellIndex];  
        mcCases[cellIndex] = caseNumber;

        int yIntersection = hasYEdgeIntersection(caseNumber) ? 1 : 0;
        int zIntersection = hasZEdgeIntersection(caseNumber) ? 1 : 0;

        yEdgeIntersections[cellIndex] += yIntersection;
        zEdgeIntersections[cellIndex] += zIntersection;

        int triangleCount = getTriangleCount(caseNumber);
        triangleCounts[cellIndex] = triangleCount;
    }
}

