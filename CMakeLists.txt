cmake_minimum_required(VERSION 3.27)
project(FlyingEdges C)

set(CMAKE_C_STANDARD 11)

include_directories(.)
include_directories(config)
include_directories(utils)

add_executable(FlyingEdges
        config/config.h
)
