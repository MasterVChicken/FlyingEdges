//
// Created by Yanliang Li on 6/1/24.
//

#ifndef FLYINGEDGES_CONFIG_H
#define FLYINGEDGES_CONFIG_H

#pragma once

#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>

using std::size_t;

using scalar_t = float;

using uchar = unsigned char;

using ushort = unsigned short;

using cube_t = std::array<std::array<scalar_t, 3>, 8>;
using scalarCube_t = std::array<scalar_t, 8>;

#endif //FLYINGEDGES_CONFIG_H
