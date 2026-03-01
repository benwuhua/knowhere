# cmake/libs/libpipnn_diskann.cmake
# PiPNN-DiskANN: requires DiskANN to be already included
# Eigen is header-only, no link needed — just include dirs

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

message(STATUS "PiPNN-DiskANN: Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
