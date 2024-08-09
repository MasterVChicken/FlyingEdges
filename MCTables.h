#pragma once
#include "cuda_runtime.h"


// Represent every single status for all 8 nodes
const int caseNum 256;
const int verticeNum 8;
const int edgeNum 12;

__constant__ int d_trim_table[trimSize];
__constant__ int d_edgeUses[caseNum/2][edgeNum];
__constant__ int d_edgeCases[caseNum][16];
__constant__ int d_edgeMap[edgeNum][2];
__constant__ int d_verticeMap[verticeNum][3];

// The number of trims corresponding to edge cases
int numTrims[caseNum] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 2, 4, 3, 3, 4, 4, 5, 4, 3, 5, 2,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 3, 5, 2, 4, 5, 5, 4, 5, 4, 2, 1,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 2, 3, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 5, 4, 4, 5, 3, 2, 5, 2, 4, 1,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 3, 5, 4, 4, 5, 5, 2, 3, 2, 4, 1,
    3, 4, 4, 5, 4, 5, 5, 2, 4, 5, 3, 4, 3, 4, 2, 1, 2, 3, 3, 2, 3, 2, 4, 1, 3, 4, 2, 1, 2, 1, 1, 0
};


// Each edge position(12 in total)
// Table size reduce to half due to symmetry
// We are supposed to access each case via: edgecase < 128 ? edgeUses[edgecase] : edgeUses[127 - (edgecase - 128)]
int edgeUses[caseNum/2][edgeNum] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
    { 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 },
    { 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0 }, { 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 },
    { 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0 }, { 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0 },
    { 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 }, { 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1 },
    { 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 }, { 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1 },
    { 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1 }, { 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1 },
    { 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
    { 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0 }, { 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
    { 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0 }, { 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0 },
    { 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }, { 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
    { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0 }, { 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0 },
    { 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1 }, { 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1 },
    { 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1 }, { 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1 },
    { 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1 }, { 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1 },
    { 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1 }, { 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1 },
    { 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0 }, { 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0 },
    { 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0 }, { 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0 },
    { 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0 }, { 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 },
    { 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0 }, { 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0 },
    { 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 }, { 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 },
    { 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1 }, { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1 },
    { 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1 }, { 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1 },
    { 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1 }, { 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1 },
    { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0 }, { 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0 },
    { 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0 }, { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
    { 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0 }, { 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0 },
    { 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0 }, { 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0 },
    { 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1 }, { 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1 },
    { 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1 }, { 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1 },
    { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 }, { 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1 },
    { 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1 }, { 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1 },
    { 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0 }, { 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0 },
    { 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0 }, { 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0 },
    { 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0 }, { 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0 },
    { 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0 }, { 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0 },
    { 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1 }, { 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1 },
    { 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1 }, { 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 },
    { 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1 }, { 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1 },
    { 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1 }, { 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1 },
    { 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0 }, { 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0 },
    { 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0 }, { 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0 },
    { 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0 }, { 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0 },
    { 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1 }, { 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 },
    { 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1 },
    { 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1 }, { 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1 },
    { 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1 }, { 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1 },
    { 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0 }, { 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0 },
    { 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0 }, { 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0 },
    { 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0 }, { 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0 },
    { 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1 }, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
    { 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1 }, { 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1 },
    { 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1 }, { 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 },
    { 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1 }, { 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 },
    { 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0 }, { 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 },
    { 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0 }, { 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0 },
    { 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0 }, { 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0 },
    { 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0 }, { 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 },
    { 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1 }, { 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1 },
    { 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1 }, { 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1 },
    { 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1 }, { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 },
    { 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 }, { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
};

// The first number indicates how many triangles does the case have
// The following every 3 consecutive numbers represent the trim edge corresponding to triangles
// At most, there will be 5 triangles in a cube
int edgeCases[caseNum][16] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 0, 9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 5, 4, 8, 9, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 4, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 1, 10, 8, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 5, 0, 9, 1, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 5, 1, 10, 5, 10, 9, 9, 10, 8, 0, 0, 0, 0, 0, 0 },
    { 1, 5, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 4, 8, 5, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 11, 1, 0, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 1, 4, 8, 1, 8, 11, 11, 8, 9, 0, 0, 0, 0, 0, 0 },
    { 2, 4, 5, 11, 10, 4, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 5, 11, 0, 11, 8, 8, 11, 10, 0, 0, 0, 0, 0, 0 },
    { 3, 4, 0, 9, 4, 9, 10, 10, 9, 11, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 11, 8, 11, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 2, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 2, 0, 4, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 9, 5, 8, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 2, 9, 5, 2, 5, 6, 6, 5, 4, 0, 0, 0, 0, 0, 0 },
    { 2, 8, 6, 2, 4, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 10, 6, 2, 10, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 9, 5, 0, 8, 6, 2, 1, 10, 4, 0, 0, 0, 0, 0, 0 },
    { 4, 2, 10, 6, 9, 10, 2, 9, 1, 10, 9, 5, 1, 0, 0, 0 },
    { 2, 5, 11, 1, 8, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 4, 6, 2, 4, 2, 0, 5, 11, 1, 0, 0, 0, 0, 0, 0 },
    { 3, 9, 11, 1, 9, 1, 0, 8, 6, 2, 0, 0, 0, 0, 0, 0 },
    { 4, 1, 9, 11, 1, 6, 9, 1, 4, 6, 6, 2, 9, 0, 0, 0 },
    { 3, 4, 5, 11, 4, 11, 10, 6, 2, 8, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 11, 10, 5, 10, 2, 5, 2, 0, 6, 2, 10, 0, 0, 0 },
    { 4, 2, 8, 6, 9, 10, 0, 9, 11, 10, 10, 4, 0, 0, 0, 0 },
    { 3, 2, 10, 6, 2, 9, 10, 9, 11, 10, 0, 0, 0, 0, 0, 0 },
    { 1, 9, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 2, 7, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 2, 7, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 8, 2, 7, 8, 7, 4, 4, 7, 5, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 2, 7, 1, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 1, 10, 0, 10, 8, 2, 7, 9, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 2, 7, 0, 7, 5, 1, 10, 4, 0, 0, 0, 0, 0, 0 },
    { 4, 1, 7, 5, 1, 8, 7, 1, 10, 8, 2, 7, 8, 0, 0, 0 },
    { 2, 5, 11, 1, 9, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 4, 8, 0, 5, 11, 1, 2, 7, 9, 0, 0, 0, 0, 0, 0 },
    { 3, 7, 11, 1, 7, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 1, 7, 11, 4, 7, 1, 4, 2, 7, 4, 8, 2, 0, 0, 0 },
    { 3, 11, 10, 4, 11, 4, 5, 9, 2, 7, 0, 0, 0, 0, 0, 0 },
    { 4, 2, 7, 9, 0, 5, 8, 8, 5, 11, 8, 11, 10, 0, 0, 0 },
    { 4, 7, 0, 2, 7, 10, 0, 7, 11, 10, 10, 4, 0, 0, 0, 0 },
    { 3, 7, 8, 2, 7, 11, 8, 11, 10, 8, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 8, 6, 7, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 9, 0, 4, 9, 4, 7, 7, 4, 6, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 8, 6, 0, 6, 5, 5, 6, 7, 0, 0, 0, 0, 0, 0 },
    { 2, 5, 4, 7, 4, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 6, 7, 9, 6, 9, 8, 4, 1, 10, 0, 0, 0, 0, 0, 0 },
    { 4, 9, 6, 7, 9, 1, 6, 9, 0, 1, 1, 10, 6, 0, 0, 0 },
    { 4, 1, 10, 4, 0, 8, 5, 5, 8, 6, 5, 6, 7, 0, 0, 0 },
    { 3, 10, 5, 1, 10, 6, 5, 6, 7, 5, 0, 0, 0, 0, 0, 0 },
    { 3, 9, 8, 6, 9, 6, 7, 11, 1, 5, 0, 0, 0, 0, 0, 0 },
    { 4, 11, 1, 5, 9, 0, 7, 7, 0, 4, 7, 4, 6, 0, 0, 0 },
    { 4, 8, 1, 0, 8, 7, 1, 8, 6, 7, 11, 1, 7, 0, 0, 0 },
    { 3, 1, 7, 11, 1, 4, 7, 4, 6, 7, 0, 0, 0, 0, 0, 0 },
    { 4, 9, 8, 7, 8, 6, 7, 11, 4, 5, 11, 10, 4, 0, 0, 0 },
    { 5, 7, 0, 6, 7, 9, 0, 6, 0, 10, 5, 11, 0, 10, 0, 11 },
    { 5, 10, 0, 11, 10, 4, 0, 11, 0, 7, 8, 6, 0, 7, 0, 6 },
    { 2, 10, 7, 11, 6, 7, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 6, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 4, 8, 0, 10, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 9, 5, 10, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 8, 9, 5, 8, 5, 4, 10, 3, 6, 0, 0, 0, 0, 0, 0 },
    { 2, 6, 4, 1, 3, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 6, 8, 0, 6, 0, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0 },
    { 3, 1, 3, 6, 1, 6, 4, 0, 9, 5, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 1, 3, 5, 3, 8, 5, 8, 9, 8, 3, 6, 0, 0, 0 },
    { 2, 11, 1, 5, 3, 6, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 5, 11, 1, 4, 8, 0, 3, 6, 10, 0, 0, 0, 0, 0, 0 },
    { 3, 1, 0, 9, 1, 9, 11, 3, 6, 10, 0, 0, 0, 0, 0, 0 },
    { 4, 3, 6, 10, 1, 4, 11, 11, 4, 8, 11, 8, 9, 0, 0, 0 },
    { 3, 11, 3, 6, 11, 6, 5, 5, 6, 4, 0, 0, 0, 0, 0, 0 },
    { 4, 11, 3, 6, 5, 11, 6, 5, 6, 8, 5, 8, 0, 0, 0, 0 },
    { 4, 0, 6, 4, 0, 11, 6, 0, 9, 11, 3, 6, 11, 0, 0, 0 },
    { 3, 6, 11, 3, 6, 8, 11, 8, 9, 11, 0, 0, 0, 0, 0, 0 },
    { 2, 3, 2, 8, 10, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 4, 10, 3, 4, 3, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0 },
    { 3, 8, 10, 3, 8, 3, 2, 9, 5, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 9, 3, 2, 9, 4, 3, 9, 5, 4, 10, 3, 4, 0, 0, 0 },
    { 3, 8, 4, 1, 8, 1, 2, 2, 1, 3, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 1, 2, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 0, 9, 1, 2, 4, 1, 3, 2, 2, 8, 4, 0, 0, 0 },
    { 3, 5, 2, 9, 5, 1, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0 },
    { 3, 3, 2, 8, 3, 8, 10, 1, 5, 11, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 11, 1, 4, 10, 0, 0, 10, 3, 0, 3, 2, 0, 0, 0 },
    { 4, 2, 8, 10, 2, 10, 3, 0, 9, 1, 1, 9, 11, 0, 0, 0 },
    { 5, 11, 4, 9, 11, 1, 4, 9, 4, 2, 10, 3, 4, 2, 4, 3 },
    { 4, 8, 4, 5, 8, 5, 3, 8, 3, 2, 3, 5, 11, 0, 0, 0 },
    { 3, 11, 0, 5, 11, 3, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0 },
    { 5, 2, 4, 3, 2, 8, 4, 3, 4, 11, 0, 9, 4, 11, 4, 9 },
    { 2, 11, 2, 9, 3, 2, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 2, 7, 9, 6, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 4, 8, 2, 7, 9, 10, 3, 6, 0, 0, 0, 0, 0, 0 },
    { 3, 7, 5, 0, 7, 0, 2, 6, 10, 3, 0, 0, 0, 0, 0, 0 },
    { 4, 10, 3, 6, 8, 2, 4, 4, 2, 7, 4, 7, 5, 0, 0, 0 },
    { 3, 6, 4, 1, 6, 1, 3, 7, 9, 2, 0, 0, 0, 0, 0, 0 },
    { 4, 9, 2, 7, 0, 3, 8, 0, 1, 3, 3, 6, 8, 0, 0, 0 },
    { 4, 4, 1, 3, 4, 3, 6, 5, 0, 7, 7, 0, 2, 0, 0, 0 },
    { 5, 3, 8, 1, 3, 6, 8, 1, 8, 5, 2, 7, 8, 5, 8, 7 },
    { 3, 9, 2, 7, 11, 1, 5, 6, 10, 3, 0, 0, 0, 0, 0, 0 },
    { 4, 3, 6, 10, 5, 11, 1, 0, 4, 8, 2, 7, 9, 0, 0, 0 },
    { 4, 6, 10, 3, 7, 11, 2, 2, 11, 1, 2, 1, 0, 0, 0, 0 },
    { 5, 4, 8, 2, 4, 2, 7, 4, 7, 1, 11, 1, 7, 10, 3, 6 },
    { 4, 9, 2, 7, 11, 3, 5, 5, 3, 6, 5, 6, 4, 0, 0, 0 },
    { 5, 5, 11, 3, 5, 3, 6, 5, 6, 0, 8, 0, 6, 9, 2, 7 },
    { 5, 2, 11, 0, 2, 7, 11, 0, 11, 4, 3, 6, 11, 4, 11, 6 },
    { 4, 6, 11, 3, 6, 8, 11, 7, 11, 2, 2, 11, 8, 0, 0, 0 },
    { 3, 3, 7, 9, 3, 9, 10, 10, 9, 8, 0, 0, 0, 0, 0, 0 },
    { 4, 4, 10, 3, 0, 4, 3, 0, 3, 7, 0, 7, 9, 0, 0, 0 },
    { 4, 0, 8, 10, 0, 10, 7, 0, 7, 5, 7, 10, 3, 0, 0, 0 },
    { 3, 3, 4, 10, 3, 7, 4, 7, 5, 4, 0, 0, 0, 0, 0, 0 },
    { 4, 7, 9, 8, 7, 8, 1, 7, 1, 3, 4, 1, 8, 0, 0, 0 },
    { 3, 9, 3, 7, 9, 0, 3, 0, 1, 3, 0, 0, 0, 0, 0, 0 },
    { 5, 5, 8, 7, 5, 0, 8, 7, 8, 3, 4, 1, 8, 3, 8, 1 },
    { 2, 5, 3, 7, 1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 11, 1, 9, 10, 7, 9, 8, 10, 10, 3, 7, 0, 0, 0 },
    { 5, 0, 4, 10, 0, 10, 3, 0, 3, 9, 7, 9, 3, 5, 11, 1 },
    { 5, 10, 7, 8, 10, 3, 7, 8, 7, 0, 11, 1, 7, 0, 7, 1 },
    { 4, 3, 4, 10, 3, 7, 4, 1, 4, 11, 11, 4, 7, 0, 0, 0 },
    { 5, 5, 3, 4, 5, 11, 3, 4, 3, 8, 7, 9, 3, 8, 3, 9 },
    { 4, 11, 0, 5, 11, 3, 0, 9, 0, 7, 7, 0, 3, 0, 0, 0 },
    { 2, 0, 8, 4, 7, 11, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 11, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 11, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 4, 8, 7, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 5, 0, 7, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 5, 4, 8, 5, 8, 9, 7, 3, 11, 0, 0, 0, 0, 0, 0 },
    { 2, 1, 10, 4, 11, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 10, 8, 0, 10, 0, 1, 11, 7, 3, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 9, 5, 1, 10, 4, 7, 3, 11, 0, 0, 0, 0, 0, 0 },
    { 4, 7, 3, 11, 5, 1, 9, 9, 1, 10, 9, 10, 8, 0, 0, 0 },
    { 2, 5, 7, 3, 1, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 5, 7, 3, 5, 3, 1, 4, 8, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 9, 7, 3, 9, 3, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0 },
    { 4, 7, 8, 9, 7, 1, 8, 7, 3, 1, 4, 8, 1, 0, 0, 0 },
    { 3, 3, 10, 4, 3, 4, 7, 7, 4, 5, 0, 0, 0, 0, 0, 0 },
    { 4, 0, 10, 8, 0, 7, 10, 0, 5, 7, 7, 3, 10, 0, 0, 0 },
    { 4, 4, 3, 10, 0, 3, 4, 0, 7, 3, 0, 9, 7, 0, 0, 0 },
    { 3, 3, 9, 7, 3, 10, 9, 10, 8, 9, 0, 0, 0, 0, 0, 0 },
    { 2, 7, 3, 11, 2, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 2, 0, 4, 2, 4, 6, 3, 11, 7, 0, 0, 0, 0, 0, 0 },
    { 3, 5, 0, 9, 7, 3, 11, 8, 6, 2, 0, 0, 0, 0, 0, 0 },
    { 4, 11, 7, 3, 5, 6, 9, 5, 4, 6, 6, 2, 9, 0, 0, 0 },
    { 3, 4, 1, 10, 6, 2, 8, 11, 7, 3, 0, 0, 0, 0, 0, 0 },
    { 4, 7, 3, 11, 2, 1, 6, 2, 0, 1, 1, 10, 6, 0, 0, 0 },
    { 4, 0, 9, 5, 2, 8, 6, 1, 10, 4, 7, 3, 11, 0, 0, 0 },
    { 5, 9, 5, 1, 9, 1, 10, 9, 10, 2, 6, 2, 10, 7, 3, 11 },
    { 3, 3, 1, 5, 3, 5, 7, 2, 8, 6, 0, 0, 0, 0, 0, 0 },
    { 4, 5, 7, 1, 7, 3, 1, 4, 2, 0, 4, 6, 2, 0, 0, 0 },
    { 4, 8, 6, 2, 9, 7, 0, 0, 7, 3, 0, 3, 1, 0, 0, 0 },
    { 5, 6, 9, 4, 6, 2, 9, 4, 9, 1, 7, 3, 9, 1, 9, 3 },
    { 4, 8, 6, 2, 4, 7, 10, 4, 5, 7, 7, 3, 10, 0, 0, 0 },
    { 5, 7, 10, 5, 7, 3, 10, 5, 10, 0, 6, 2, 10, 0, 10, 2 },
    { 5, 0, 9, 7, 0, 7, 3, 0, 3, 4, 10, 4, 3, 8, 6, 2 },
    { 4, 3, 9, 7, 3, 10, 9, 2, 9, 6, 6, 9, 10, 0, 0, 0 },
    { 2, 11, 9, 2, 3, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 2, 3, 11, 2, 11, 9, 0, 4, 8, 0, 0, 0, 0, 0, 0 },
    { 3, 11, 5, 0, 11, 0, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0 },
    { 4, 8, 5, 4, 8, 3, 5, 8, 2, 3, 3, 11, 5, 0, 0, 0 },
    { 3, 11, 9, 2, 11, 2, 3, 10, 4, 1, 0, 0, 0, 0, 0, 0 },
    { 4, 0, 1, 8, 1, 10, 8, 2, 11, 9, 2, 3, 11, 0, 0, 0 },
    { 4, 4, 1, 10, 0, 3, 5, 0, 2, 3, 3, 11, 5, 0, 0, 0 },
    { 5, 3, 5, 2, 3, 11, 5, 2, 5, 8, 1, 10, 5, 8, 5, 10 },
    { 3, 5, 9, 2, 5, 2, 1, 1, 2, 3, 0, 0, 0, 0, 0, 0 },
    { 4, 4, 8, 0, 5, 9, 1, 1, 9, 2, 1, 2, 3, 0, 0, 0 },
    { 2, 0, 2, 1, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 8, 1, 4, 8, 2, 1, 2, 3, 1, 0, 0, 0, 0, 0, 0 },
    { 4, 9, 2, 3, 9, 3, 4, 9, 4, 5, 10, 4, 3, 0, 0, 0 },
    { 5, 8, 5, 10, 8, 0, 5, 10, 5, 3, 9, 2, 5, 3, 5, 2 },
    { 3, 4, 3, 10, 4, 0, 3, 0, 2, 3, 0, 0, 0, 0, 0, 0 },
    { 2, 3, 8, 2, 10, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 6, 3, 11, 6, 11, 8, 8, 11, 9, 0, 0, 0, 0, 0, 0 },
    { 4, 0, 4, 6, 0, 6, 11, 0, 11, 9, 3, 11, 6, 0, 0, 0 },
    { 4, 11, 6, 3, 5, 6, 11, 5, 8, 6, 5, 0, 8, 0, 0, 0 },
    { 3, 11, 6, 3, 11, 5, 6, 5, 4, 6, 0, 0, 0, 0, 0, 0 },
    { 4, 1, 10, 4, 11, 8, 3, 11, 9, 8, 8, 6, 3, 0, 0, 0 },
    { 5, 1, 6, 0, 1, 10, 6, 0, 6, 9, 3, 11, 6, 9, 6, 11 },
    { 5, 5, 0, 8, 5, 8, 6, 5, 6, 11, 3, 11, 6, 1, 10, 4 },
    { 4, 10, 5, 1, 10, 6, 5, 11, 5, 3, 3, 5, 6, 0, 0, 0 },
    { 4, 5, 3, 1, 5, 8, 3, 5, 9, 8, 8, 6, 3, 0, 0, 0 },
    { 5, 1, 9, 3, 1, 5, 9, 3, 9, 6, 0, 4, 9, 6, 9, 4 },
    { 3, 6, 0, 8, 6, 3, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 6, 1, 4, 3, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 5, 8, 3, 9, 8, 6, 3, 9, 3, 5, 10, 4, 3, 5, 3, 4 },
    { 2, 0, 5, 9, 10, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 6, 0, 8, 6, 3, 0, 4, 0, 10, 10, 0, 3, 0, 0, 0 },
    { 1, 6, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 10, 11, 7, 6, 10, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 10, 11, 7, 10, 7, 6, 8, 0, 4, 0, 0, 0, 0, 0, 0 },
    { 3, 7, 6, 10, 7, 10, 11, 5, 0, 9, 0, 0, 0, 0, 0, 0 },
    { 4, 11, 7, 6, 11, 6, 10, 9, 5, 8, 8, 5, 4, 0, 0, 0 },
    { 3, 1, 11, 7, 1, 7, 4, 4, 7, 6, 0, 0, 0, 0, 0, 0 },
    { 4, 8, 0, 1, 8, 1, 7, 8, 7, 6, 11, 7, 1, 0, 0, 0 },
    { 4, 9, 5, 0, 7, 4, 11, 7, 6, 4, 4, 1, 11, 0, 0, 0 },
    { 5, 9, 1, 8, 9, 5, 1, 8, 1, 6, 11, 7, 1, 6, 1, 7 },
    { 3, 10, 1, 5, 10, 5, 6, 6, 5, 7, 0, 0, 0, 0, 0, 0 },
    { 4, 0, 4, 8, 5, 6, 1, 5, 7, 6, 6, 10, 1, 0, 0, 0 },
    { 4, 9, 7, 6, 9, 6, 1, 9, 1, 0, 1, 6, 10, 0, 0, 0 },
    { 5, 6, 1, 7, 6, 10, 1, 7, 1, 9, 4, 8, 1, 9, 1, 8 },
    { 2, 5, 7, 4, 4, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 6, 8, 0, 5, 6, 5, 7, 6, 0, 0, 0, 0, 0, 0 },
    { 3, 9, 4, 0, 9, 7, 4, 7, 6, 4, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 6, 8, 7, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 7, 2, 8, 7, 8, 11, 11, 8, 10, 0, 0, 0, 0, 0, 0 },
    { 4, 7, 2, 0, 7, 0, 10, 7, 10, 11, 10, 0, 4, 0, 0, 0 },
    { 4, 0, 9, 5, 8, 11, 2, 8, 10, 11, 11, 7, 2, 0, 0, 0 },
    { 5, 11, 2, 10, 11, 7, 2, 10, 2, 4, 9, 5, 2, 4, 2, 5 },
    { 4, 1, 11, 7, 4, 1, 7, 4, 7, 2, 4, 2, 8, 0, 0, 0 },
    { 3, 7, 1, 11, 7, 2, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0 },
    { 5, 4, 1, 11, 4, 11, 7, 4, 7, 8, 2, 8, 7, 0, 9, 5 },
    { 4, 7, 1, 11, 7, 2, 1, 5, 1, 9, 9, 1, 2, 0, 0, 0 },
    { 4, 1, 5, 7, 1, 7, 8, 1, 8, 10, 2, 8, 7, 0, 0, 0 },
    { 5, 0, 10, 2, 0, 4, 10, 2, 10, 7, 1, 5, 10, 7, 10, 5 },
    { 5, 0, 7, 1, 0, 9, 7, 1, 7, 10, 2, 8, 7, 10, 7, 8 },
    { 2, 9, 7, 2, 1, 4, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 8, 7, 2, 8, 4, 7, 4, 5, 7, 0, 0, 0, 0, 0, 0 },
    { 2, 0, 7, 2, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 8, 7, 2, 8, 4, 7, 9, 7, 0, 0, 7, 4, 0, 0, 0 },
    { 1, 9, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 2, 6, 10, 2, 10, 9, 9, 10, 11, 0, 0, 0, 0, 0, 0 },
    { 4, 0, 4, 8, 2, 6, 9, 9, 6, 10, 9, 10, 11, 0, 0, 0 },
    { 4, 5, 10, 11, 5, 2, 10, 5, 0, 2, 6, 10, 2, 0, 0, 0 },
    { 5, 4, 2, 5, 4, 8, 2, 5, 2, 11, 6, 10, 2, 11, 2, 10 },
    { 4, 1, 11, 9, 1, 9, 6, 1, 6, 4, 6, 9, 2, 0, 0, 0 },
    { 5, 9, 6, 11, 9, 2, 6, 11, 6, 1, 8, 0, 6, 1, 6, 0 },
    { 5, 4, 11, 6, 4, 1, 11, 6, 11, 2, 5, 0, 11, 2, 11, 0 },
    { 2, 5, 1, 11, 8, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 2, 6, 10, 9, 2, 10, 9, 10, 1, 9, 1, 5, 0, 0, 0 },
    { 5, 9, 2, 6, 9, 6, 10, 9, 10, 5, 1, 5, 10, 0, 4, 8 },
    { 3, 10, 2, 6, 10, 1, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0 },
    { 4, 10, 2, 6, 10, 1, 2, 8, 2, 4, 4, 2, 1, 0, 0, 0 },
    { 3, 2, 5, 9, 2, 6, 5, 6, 4, 5, 0, 0, 0, 0, 0, 0 },
    { 4, 2, 5, 9, 2, 6, 5, 0, 5, 8, 8, 5, 6, 0, 0, 0 },
    { 2, 2, 4, 0, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 2, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 8, 11, 11, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 4, 9, 0, 4, 10, 9, 10, 11, 9, 0, 0, 0, 0, 0, 0 },
    { 3, 0, 11, 5, 0, 8, 11, 8, 10, 11, 0, 0, 0, 0, 0, 0 },
    { 2, 4, 11, 5, 10, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 1, 8, 4, 1, 11, 8, 11, 9, 8, 0, 0, 0, 0, 0, 0 },
    { 2, 9, 1, 11, 0, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 4, 1, 8, 4, 1, 11, 8, 0, 8, 5, 5, 8, 11, 0, 0, 0 },
    { 1, 5, 1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 3, 5, 10, 1, 5, 9, 10, 9, 8, 10, 0, 0, 0, 0, 0, 0 },
    { 4, 4, 9, 0, 4, 10, 9, 5, 9, 1, 1, 9, 10, 0, 0, 0 },
    { 2, 0, 10, 1, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 4, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 2, 5, 8, 4, 9, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 0, 5, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

// start, end
int edgeMap[edgeNum][2] = {
    { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 0, 2 }, { 1, 3 },
    { 4, 6 }, { 5, 7 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
  };

// x, y, z of vertices
int verticeMap[verticeNum][3] = {
    { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 },
    { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 1 },
  };

void initializeTable(const int* h_numTrims, const int* h_edgeUses, const int* h_edgeCases, const int* h_edgeMap, const int* h_verticeMap) {
    cudaMemcpyToSymbol(d_numTrims, h_numTrims, caseNum * sizeof(int));
    cudaMemcpyToSymbol(d_edgeUses, h_edgeUses, (caseNum/2) * edgeNum * sizeof(int));
    cudaMemcpyToSymbol(d_edgeCases, h_edgeCases, 16* caseNum * sizeof(int));
    cudaMemcpyToSymbol(d_edgeMap, h_edgeMap, edgeNum * 2 * sizeof(int));
    cudaMemcpyToSymbol(d_verticeMap, h_verticeMap, verticeNum * 2 * sizeof(int));
}

__device__ int getValueFromEdgeUses(int index) {
    return index < 128 ? d_edgeUses[index] : d_edgeUses[127 - (index - 128)];
}