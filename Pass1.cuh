#include "MCTables.h"

template <typename T>
__device__ int computeEdgeCaseNumber(T left, T right, T isoValue) {
    int caseNumber = 0;

    if (left < isoValue && right >= isoValue) {
        caseNumber = 1;  
    }
    
    else if (left >= isoValue && right < isoValue) {
        caseNumber = 2;  
    }
   
    else {
        caseNumber = 0;  
    }

    return caseNumber;
}

template <typename T>
__global__ void processXEdges(const T *__restrict__ scalarField,
                              int *__restrict__ edgeCases,
                              T *__restrict__ trimPositionsLeft,
                              T *__restrict__ trimPositionsRight,
                              const T isoValue, const int3 gridSize) {

  // get y and z posistions
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j >= gridSize.y - 1 || k >= gridSize.z - 1)
    return;

  // minimize global memory access
  __shared__ T sharedScalarField[BLOCK_SIZE];

  for (int i = 0; i < gridSize.x - 1; ++i) {

    int edgeIndex = i + j * gridSize.x + k * gridSize.x * gridSize.y;
    sharedScalarField[threadIdx.x] = scalarField[edgeIndex];
    __syncthreads();

    T val1 = sharedScalarField[threadIdx.x];
    T val2 = (i < gridSize.x - 2) ? scalarField[edgeIndex + 1]
                                  : scalarField[edgeIndex];

    T xL = static_cast<T>(i);
    T xR = static_cast<T>(i + 1);
    int intersectionCount = 0;

    if ((val1 < isoValue && val2 >= isoValue) ||
        (val1 >= isoValue && val2 < isoValue)) {
      edgeCases[edgeIndex] = computeEdgeCaseNumber(val1, val2, isoValue);
      intersectionCount++;

      if (val1 < isoValue) {
        xL = static_cast<T>(i) + (isoValue - val1) / (val2 - val1);
      }
      if (val2 < isoValue) {
        xR = static_cast<T>(i + 1) - (isoValue - val2) / (val1 - val2);
      }

      if (i == gridSize.x - 2 ||
          (scalarField[edgeIndex + 1] >= isoValue && val2 < isoValue)) {
        break;
      }
    } else {
      edgeCases[edgeIndex] = 0;
    }

    trimPositionsLeft[edgeIndex] = xL;
    trimPositionsRight[edgeIndex] = xR;
    __syncthreads();
  }
}
