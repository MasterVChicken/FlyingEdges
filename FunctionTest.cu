#include <chrono>

#include "Pass1.cuh"
#include "Pass2.cuh"
#include "Pass3.cuh"
#include "Pass4.cuh"


const int3 gridSize = {512, 512, 512};

void checkCudaError(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    
    float* d_scalarField;
    float* d_vertices;
    int* d_edgeCases;
    int* d_mcCases;
    int* d_yEdgePrefixSums;
    int* d_zEdgePrefixSums;
    int* d_trianglePrefixSums;

    float3* d_outputVertices;
    int3* d_outputTriangles;

    // Allocate memory
    checkCudaError(cudaMalloc(&d_scalarField, gridSize.x * gridSize.y * gridSize.z * sizeof(float)), "Allocating d_scalarField");
    checkCudaError(cudaMalloc(&d_vertices, gridSize.x * gridSize.y * gridSize.z * 8 * sizeof(float)), "Allocating d_vertices");
    checkCudaError(cudaMalloc(&d_edgeCases, gridSize.x * gridSize.y * gridSize.z * sizeof(int)), "Allocating d_edgeCases");
    checkCudaError(cudaMalloc(&d_mcCases, gridSize.x * gridSize.y * gridSize.z * sizeof(int)), "Allocating d_mcCases");
    checkCudaError(cudaMalloc(&d_yEdgePrefixSums, gridSize.y * gridSize.z * sizeof(int)), "Allocating d_yEdgePrefixSums");
    checkCudaError(cudaMalloc(&d_zEdgePrefixSums, gridSize.y * gridSize.z * sizeof(int)), "Allocating d_zEdgePrefixSums");
    checkCudaError(cudaMalloc(&d_trianglePrefixSums, gridSize.y * gridSize.z * sizeof(int)), "Allocating d_trianglePrefixSums");

    checkCudaError(cudaMalloc(&d_outputVertices, gridSize.x * gridSize.y * gridSize.z * 12 * sizeof(float3)), "Allocating d_outputVertices");
    checkCudaError(cudaMalloc(&d_outputTriangles, gridSize.x * gridSize.y * gridSize.z * 5 * sizeof(int3)), "Allocating d_outputTriangles");

    float isoValue = 0.5f;

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Creating start event");
    checkCudaError(cudaEventCreate(&stop), "Creating stop event");
    float milliseconds = 0;

    checkCudaError(cudaEventRecord(start), "Recording start for Pass 1");
    processPass1(d_scalarField, d_edgeCases, gridSize);
    checkCudaError(cudaEventRecord(stop), "Recording stop for Pass 1");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop for Pass 1");
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time for Pass 1");
    std::cout << "Pass 1 Time: " << milliseconds << " ms" << std::endl;

    checkCudaError(cudaEventRecord(start), "Recording start for Pass 2");
    processPass2(d_scalarField, d_edgeCases, d_mcCases, d_yEdgePrefixSums, d_zEdgePrefixSums, gridSize);
    checkCudaError(cudaEventRecord(stop), "Recording stop for Pass 2");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop for Pass 2");
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time for Pass 2");
    std::cout << "Pass 2 Time: " << milliseconds << " ms" << std::endl;

    checkCudaError(cudaEventRecord(start), "Recording start for Pass 3");
    processPass3(d_scalarField, d_edgeCases, d_mcCases, d_yEdgePrefixSums, d_zEdgePrefixSums, d_trianglePrefixSums, gridSize);
    checkCudaError(cudaEventRecord(stop), "Recording stop for Pass 3");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop for Pass 3");
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time for Pass 3");
    std::cout << "Pass 3 Time: " << milliseconds << " ms" << std::endl;

    checkCudaError(cudaEventRecord(start), "Recording start for Pass 4");
    processPass4(d_scalarField, d_vertices, d_edgeCases, d_mcCases, d_yEdgePrefixSums, d_zEdgePrefixSums, d_trianglePrefixSums, d_outputVertices, d_outputTriangles, isoValue, gridSize);
    checkCudaError(cudaEventRecord(stop), "Recording stop for Pass 4");
    checkCudaError(cudaEventSynchronize(stop), "Synchronizing stop for Pass 4");
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time for Pass 4");
    std::cout << "Pass 4 Time: " << milliseconds << " ms" << std::endl;

    cudaFree(d_scalarField);
    cudaFree(d_vertices);
    cudaFree(d_edgeCases);
    cudaFree(d_mcCases);
    cudaFree(d_yEdgePrefixSums);
    cudaFree(d_zEdgePrefixSums);
    cudaFree(d_trianglePrefixSums);
    cudaFree(d_outputVertices);
    cudaFree(d_outputTriangles);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// To do:
// 1.
// To improve performance, we might consider using different stream to run all the pass functions at the same time
// eg: We can divide the grid into several subgrids and execute 4 pass function on subgrids individually
// 2. Minimize the initialization time compared to vtk-m

