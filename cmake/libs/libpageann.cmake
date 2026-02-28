# PageANN: Page-based ANN index (from paper "Scalable Disk-Based ANN Search with Page-Aligned Graph")
# This builds the Paper PageANN library as a static library for use in Knowhere.
# PageANN has its own PQFlashIndex with page-level search, separate from DiskANN's node-level search.

add_definitions(-DKNOWHERE_WITH_PAGEANN)

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIR})
find_package(aio REQUIRED)
include_directories(${AIO_INCLUDE})

# PageANN uses its own namespace (diskann_pageann) to avoid symbol conflicts with DiskANN
include_directories(thirdparty/PageANN/include)

find_package(double-conversion REQUIRED)
include_directories(${double-conversion_INCLUDE_DIRS})

# OpenBLAS and LAPACK for matrix operations (replaces MKL)
# Use standard CMake finders which find system OpenBLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# PageANN source files - the Paper's PQFlashIndex with page_search
set(PAGEANN_SOURCES
    thirdparty/PageANN/src/ann_exception.cpp
    thirdparty/PageANN/src/disk_utils.cpp
    thirdparty/PageANN/src/distance.cpp
    thirdparty/PageANN/src/index.cpp
    thirdparty/PageANN/src/in_mem_data_store.cpp
    thirdparty/PageANN/src/in_mem_graph_store.cpp
    thirdparty/PageANN/src/ooc_in_mem_data_store.cpp
    thirdparty/PageANN/src/ooc_in_mem_graph_store.cpp
    thirdparty/PageANN/src/linux_aligned_file_reader.cpp
    thirdparty/PageANN/src/math_utils.cpp
    thirdparty/PageANN/src/memory_mapper.cpp
    thirdparty/PageANN/src/partition.cpp
    thirdparty/PageANN/src/pq.cpp
    thirdparty/PageANN/src/pq_data_store.cpp
    thirdparty/PageANN/src/pq_flash_index.cpp
    thirdparty/PageANN/src/pq_l2_distance.cpp
    thirdparty/PageANN/src/scratch.cpp
    thirdparty/PageANN/src/logger.cpp
    thirdparty/PageANN/src/utils.cpp
    thirdparty/PageANN/src/filter_utils.cpp
    thirdparty/PageANN/src/index_factory.cpp
    thirdparty/PageANN/src/abstract_index.cpp
    thirdparty/PageANN/src/abstract_data_store.cpp
    thirdparty/PageANN/src/natural_number_map.cpp
    thirdparty/PageANN/src/natural_number_set.cpp)

find_package(folly REQUIRED)

add_library(pageann STATIC ${PAGEANN_SOURCES})

# PageANN uses namespace 'pageann' (renamed from diskann) to avoid symbol collisions
target_compile_definitions(pageann PRIVATE PAGEANN_LIBRARY)

# Suppress warnings-as-errors for thirdparty code and hide internal symbols
target_compile_options(pageann PRIVATE -Wno-error -fvisibility=hidden)

target_link_libraries(
  pageann
  PUBLIC ${AIO_LIBRARIES}
         ${DISKANN_BOOST_PROGRAM_OPTIONS_LIB}
         ${BLAS_LIBRARIES}
         ${LAPACK_LIBRARIES}
         nlohmann_json::nlohmann_json
         Folly::folly
         fmt::fmt-header-only
         prometheus-cpp::core
         prometheus-cpp::push
         glog::glog)
if(__X86_64)
  target_compile_options(
    pageann PRIVATE -fno-builtin-malloc -fno-builtin-calloc
                    -fno-builtin-realloc -fno-builtin-free)
endif()
list(APPEND KNOWHERE_LINKER_LIBS pageann)
