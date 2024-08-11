#include "MCTables.h"
#include <stdio.h>

// Linear interpolation
__device__ float interpolate(float isoValue, float s1, float s2, float p1, float p2) {
    if (fabs(isoValue - s1) < 1e-5) return p1;
    if (fabs(isoValue - s2) < 1e-5) return p2;
    if (fabs(s1 - s2) < 1e-5) return p1;
    return p1 + (isoValue - s1) * (p2 - p1) / (s2 - s1);
}

// Generate intersection
__device__ void computeEdgeIntersection(float* vertices, float* scalars, float isoValue, 
                                        int edgeIndex, float3* edgePoints) {
    int v1 = d_edgeMap[edgeIndex][0];
    int v2 = d_edgeMap[edgeIndex][1];
    
    edgePoints[edgeIndex].x = interpolate(isoValue, scalars[v1], scalars[v2], vertices[v1*3], vertices[v2*3]);
    edgePoints[edgeIndex].y = interpolate(isoValue, scalars[v1], scalars[v2], vertices[v1*3+1], vertices[v2*3+1]);
    edgePoints[edgeIndex].z = interpolate(isoValue, scalars[v1], scalars[v2], vertices[v1*3+2], vertices[v2*3+2]);
}

// Generate iso-surface output
template <typename T>
__global__ void generateOutput(const T* __restrict__ scalarField, 
                               const T* __restrict__ vertices,
                               const int* __restrict__ edgeCases,
                               const int* __restrict__ mcCases,
                               const int* __restrict__ yEdgePrefixSums,
                               const int* __restrict__ zEdgePrefixSums,
                               const int* __restrict__ trianglePrefixSums,
                               float3* __restrict__ outputVertices,
                               int3* __restrict__ outputTriangles,
                               const T isoValue,
                               const int3 gridSize) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= gridSize.y - 1 || k >= gridSize.z - 1) return;

    for (int i = yEdgePrefixSums[j * gridSize.z + k]; i < zEdgePrefixSums[j * gridSize.z + k]; ++i) {
        int cellIndex = i + j * gridSize.x + k * gridSize.x * gridSize.y;

        int caseNumber = mcCases[cellIndex];
        if (caseNumber == 0 || caseNumber == 255) continue;

        float3 edgePoints[12];
        computeEdgeIntersection(vertices + cellIndex * 3 * 8, scalarField + cellIndex * 8, isoValue, caseNumber, edgePoints);

        int triangleCount = d_edgeCases[caseNumber][0];
        int outputTriBase = trianglePrefixSums[cellIndex];

        for (int t = 0; t < triangleCount; ++t) {
            int v0 = d_edgeCases[caseNumber][1 + 3 * t];
            int v1 = d_edgeCases[caseNumber][2 + 3 * t];
            int v2 = d_edgeCases[caseNumber][3 + 3 * t];

            outputTriangles[outputTriBase + t] = make_int3(outputTriBase + v0, outputTriBase + v1, outputTriBase + v2);
            outputVertices[outputTriBase + v0] = edgePoints[v0];
            outputVertices[outputTriBase + v1] = edgePoints[v1];
            outputVertices[outputTriBase + v2] = edgePoints[v2];
        }
    }
}

template <typename T>
void processPass4(const T* d_scalarField,
                  const T* d_vertices,
                  const int* d_edgeCases,
                  const int* d_mcCases,
                  const int* d_yEdgePrefixSums,
                  const int* d_zEdgePrefixSums,
                  const int* d_trianglePrefixSums,
                  float3* d_outputVertices,
                  int3* d_outputTriangles,
                  T isoValue,
                  int3 gridSize) {
    dim3 blockSize(16, 16);
    dim3 gridSize = { (gridSize.y + blockSize.x - 1) / blockSize.x,
                      (gridSize.z + blockSize.y - 1) / blockSize.y };

    generateOutput<<<gridSize, blockSize>>>(d_scalarField, d_vertices, d_edgeCases, d_mcCases,
                                            d_yEdgePrefixSums, d_zEdgePrefixSums, d_trianglePrefixSums,
                                            d_outputVertices, d_outputTriangles, isoValue, gridSize);

    cudaDeviceSynchronize();
}
