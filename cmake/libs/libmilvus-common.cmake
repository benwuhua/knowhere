
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
set( MILVUS-COMMON-VERSION b6629f7 )
set( GIT_REPOSITORY  "https://github.com/zilliztech/milvus-common.git" )

message(STATUS "milvus-common repo: ${GIT_REPOSITORY}")
message(STATUS "milvus-common version: ${MILVUS-COMMON-VERSION}")
message(STATUS "Building milvus-common-${MILVUS-COMMON-VERSION} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )

include( FetchContent )

find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(folly REQUIRED)
find_package(prometheus-cpp REQUIRED)

# Add alias for fmt if header-only (fmt::fmt-header-only -> fmt::fmt)
if(TARGET fmt::fmt-header-only AND NOT TARGET fmt::fmt)
    add_library(fmt::fmt ALIAS fmt::fmt-header-only)
endif()

# Use local milvus-common if available
if(EXISTS ${CMAKE_SOURCE_DIR}/milvus-common/CMakeLists.txt)
    message(STATUS "Using local milvus-common")
    set(milvus-common_SOURCE_DIR ${CMAKE_SOURCE_DIR}/milvus-common)
    set(milvus-common_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-build)
else()
FetchContent_Declare(
        milvus-common
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${MILVUS-COMMON-VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/milvus-common-build
        SOURCE_SUBDIR   cpp
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_GetProperties( milvus-common )
if ( NOT milvus-common_POPULATED )
    FetchContent_Populate( milvus-common )
endif()
endif()

# Adding the following target:
# milvus-common
add_subdirectory( ${milvus-common_SOURCE_DIR}
                  ${milvus-common_BINARY_DIR} )

# Link atomic library to milvus-common to fix atomic operations
target_link_libraries(milvus-common PUBLIC atomic)

set( MILVUS_COMMON_INCLUDE_DIR ${milvus-common_SOURCE_DIR}/include CACHE INTERNAL "Path to milvus-common include directory" )

include_directories(${MILVUS_COMMON_INCLUDE_DIR})
