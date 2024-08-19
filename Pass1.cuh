#include "MCTables.h"
#define BLOCK_SIZE 256;

template <typename T>
__device__ int computeEdgeCaseNumber(T left, T right, T isoValue) {
    int caseNumber = 0;

    // has intersection
    if (left < isoValue && right >= isoValue) {
        caseNumber = 1;  
    }
    
    // has intersection
    else if (left >= isoValue && right < isoValue) {
        caseNumber = 2;  
    }
   
    // no intersection
    else {
        caseNumber = 0;  
    }

    return caseNumber;
}


// parameters:
// scalarField: Input scalar field data for iso-surface extraction
// edgeCases: Output array storing the edge case number for each edge
// trimPositionsLeft, trimPositionsRight: Output array for trim positions recording
// isoValue: iso-value for iso-surface extraction
// gridSize: kernel function launch parameter(gridSize here refers to grid size of scalars, so in total we can have (gridSize.x - 1) edges)

// Using __restrict__ here to tell compiler to optimize aggressively
template <typename T>
__global__ void processPass1(const T *__restrict__ scalarField,
                             int *__restrict__ edgeCases,
                             int *__restrict__ trimPositionsLeft,
                             int *__restrict__ trimPositionsRight,
                             const T isoValue, const int3 gridSize) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= gridSize.y - 1 || k >= gridSize.z - 1) return;

    int xL = -1;  // left initialized as not found
    int xR = -1;  // right initialized as not found

    for (int i = 0; i < gridSize.x - 1; ++i) {

        int edgeIndex = i + j * gridSize.x + k * gridSize.x * gridSize.y;

        T val1 = scalarField[edgeIndex];
        T val2 = scalarField[edgeIndex + 1];

        if ((val1 < isoValue && val2 >= isoValue) ||
            (val1 >= isoValue && val2 < isoValue)) {

            // get first left as leftest
            if (xL == -1) {
                xL = i;
            }

            // get last right as rightest
            xR = i + 1;

            edgeCases[edgeIndex] = computeEdgeCaseNumber(val1, val2, isoValue);
        } else {
            edgeCases[edgeIndex] = 0;
        }
    }

    trimPositionsLeft[j + k * gridSize.y] = (xL == -1) ? 0 : xL;
    trimPositionsRight[j + k * gridSize.y] = (xR == -1) ? 0 : xR;
}


