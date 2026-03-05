# cmake/libs/libpipnn_diskann.cmake
# PiPNN-DiskANN: requires DiskANN to be already included
# Prefer imported target so include paths propagate reliably.

find_package(Eigen3 REQUIRED CONFIG)

if(TARGET Eigen3::Eigen)
  message(STATUS "PiPNN-DiskANN: using Eigen target Eigen3::Eigen")
else()
  # Fallback for non-standard Eigen package exports.
  include_directories(${EIGEN3_INCLUDE_DIRS})
  message(STATUS "PiPNN-DiskANN: Eigen3::Eigen target missing, fallback include dirs: ${EIGEN3_INCLUDE_DIRS}")
endif()
