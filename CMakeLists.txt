cmake_minimum_required (VERSION 3.18)

project(tritontflitebackend LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TARGET_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm MALI GPU support in backend utilities" OFF)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

set(TFLITE_TAG "v2.4.1" CACHE STRING "Version of TFLite to build")

# ArmNN Options
set(ARMNN_TAG "v21.05" CACHE STRING "Tag for ArmNN repo")
set(ARMNN_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Set ArmNN build type")
option(ARMNN_DELEGATE_ENABLE "Build ArmNN Delegate" ON)

# ACL Options
set(ACL_TAG ${ARMNN_TAG} CACHE STRING "Tag for ACL repo")
set(ACL_DEBUG "0" CACHE STRING "Whether to compile ACL in debug mode")
set(ACL_CC_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "CC compiler to use for scons acl build")
set(ACL_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "CXX compiler to use for scons acl build")
set(ACL_ENABLE_NEON "1" CACHE STRING "Enable Neon Backend in build")
set(ACL_ENABLE_CL "0" CACHE STRING "Enable OpenCL Backend in build")

set(FLATBUFFERS_VERSION "1.12.0" CACHE STRING "Version of Flatbuffers to use")
set(JOBS "8" CACHE STRING "Number of parallel jobs for builds")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ACL_DEBUG "1")
endif()

if (TRITON_ENABLE_MALI_GPU)
  set(ACL_ENABLE_CL "1")
endif()

if (NOT DEFINED ACL_ARCH)
  if (${TARGET_ARCH} MATCHES "armv7")
    set(ACL_ARCH "armv7a")
  elseif(${TARGET_ARCH} MATCHES "armhf")
    set(ACL_ARCH "armv7a-hf")
  elseif(${TARGET_ARCH} MATCHES "aarch64")
    set(ACL_ARCH "arm64-v8a")
  elseif(${TARGET_ARCH} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(ARMNN_DELEGATE_ENABLE OFF)
  else()
    message(FATAL_ERROR "Unsupported target architecture: ${TARGET_ARCH}")
  endif()
endif()

# ACL before 21.05 compiled with only gcc7
if (NOT(ACL_TAG STREQUAL "v21.05"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
    set(ACL_CC_COMPILER "gcc-7")
    set(ACL_CXX_COMPILER "g++-7")
  endif()
endif()

set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_C_FLAGS "-fPIC")

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/jishminor/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG ${TFLITE_TAG}
  GIT_SHALLOW ON
  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/xnnpack_commit.patch || true # patch updates the commit hash for xnnpack for tensorflow v2.4.1
  SOURCE_SUBDIR tensorflow/lite
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend tensorflow-lite)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_tflite.ldscript libtriton_tflite.ldscript COPYONLY)

include(ExternalProject)

if (ARMNN_DELEGATE_ENABLE)
  #
  # Build ArmNN dependencies using custom command
  #
  set(FLATBUFFERS_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/flatbuffers)
  set(ACL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/acl)
  set(ARMNN_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external/armnn)
  file(MAKE_DIRECTORY ${ARMNN_LOCATION})

  # Flatbuffers (note that tflite brings in it's own flatbuffers, so we rename it here)
  ExternalProject_Add(
    flatbuffers-armnn
    URL https://github.com/google/flatbuffers/archive/v${FLATBUFFERS_VERSION}.zip
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
  )

  # ACL (note requires GCC 7 if using v21.02)
  ExternalProject_Add(
    acl
    PREFIX ${ACL_LOCATION}
    GIT_REPOSITORY https://review.mlplatform.org/ml/ComputeLibrary
    GIT_TAG ${ACL_TAG}
    GIT_SHALLOW ON
    CONFIGURE_COMMAND ""
    BUILD_COMMAND CC=${ACL_CC_COMPILER} CXX=${ACL_CXX_COMPILER} scons -j${JOBS} arch=${ACL_ARCH} debug=${ACL_DEBUG} neon=${ACL_ENABLE_NEON} opencl=${ACL_ENABLE_CL} embed_kernels=1 extra_cxx_flags=-fPIC benchmark_tests=0 build=native validation_tests=0 internal_only=0 examples=0
    BUILD_IN_SOURCE ON
    INSTALL_COMMAND ""
  )

  # ArmNN and its tflite delegate
  ExternalProject_Add(
    armnn
    PREFIX ${ARMNN_LOCATION}
    GIT_REPOSITORY https://review.mlplatform.org/ml/armnn
    GIT_TAG ${ARMNN_TAG}
    GIT_SHALLOW ON
    PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/tflite_armnn_delegate.patch || true # Never fail as patch is always applied after repo clone
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ARMNN_LOCATION} -DBUILD_ARMNN_TFLITE_DELEGATE=1 -DTENSORFLOW_ROOT=${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-src -DTFLITE_LIB_ROOT=${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-build -DARMCOMPUTE_ROOT=${ACL_LOCATION}/src/acl -DARMCOMPUTENEON=${ACL_ENABLE_NEON} -DARMCOMPUTECL=${ACL_ENABLE_CL} -DBUILD_UNIT_TESTS=0 -DCMAKE_BUILD_TYPE=${ARMNN_BUILD_TYPE} -DBUILD_ARMNN_SERIALIZER=0 -DARMNNREF=1 -DFLATBUFFERS_ROOT=${FLATBUFFERS_LOCATION}/flatbuffers-${FLATBUFFERS_VERSION}/install
    DEPENDS flatbuffers-armnn acl tensorflow-lite
  )
endif()

#
# Handle libs for TFLite Backend
#

add_library(
  triton-tflite-backend SHARED
  src/tflite.cc
  src/tflite_utils.cc
  src/tflite_utils.h
)

if (ARMNN_DELEGATE_ENABLE)
  add_dependencies(triton-tflite-backend armnn)
endif()

add_library(
  TritonTFLiteBackend::triton-tflite-backend ALIAS triton-tflite-backend
)

target_include_directories(
  triton-tflite-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-src
    ${ARMNN_LOCATION}/include # for armnn headers
    ${ARMNN_LOCATION}/src/armnn/delegate/include # for delegate headers
)

target_compile_features(triton-tflite-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-tflite-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

# ARMNN_DELEGATE_ENABLE exposed in header so set PUBLIC
if(${ARMNN_DELEGATE_ENABLE})
  target_compile_definitions(
    triton-backend-utils
    PUBLIC ARMNN_DELEGATE_ENABLE=1
  )
endif() # ARMNN_DELEGATE_ENABLE

set_target_properties(
  triton-tflite-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_tflite
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tflite.ldscript
    LINK_FLAGS "-Wl,--no-as-needed,--version-script libtriton_tflite.ldscript"
)

target_link_libraries(
  triton-tflite-backend
  PRIVATE
    triton-core-serverapi  # from repo-core
    triton-core-backendapi # from repo-core
    triton-core-serverstub # from repo-core
    triton-backend-utils   # from repo-backend
    tensorflow-lite
    ${CMAKE_DL_LIBS}
)

if (ARMNN_DELEGATE_ENABLE)
  target_link_libraries(
    triton-tflite-backend
    PRIVATE
      "-L${ARMNN_LOCATION}/lib" # from armnn
      -larmnn
      -larmnnDelegate
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonTFLiteBackend)

install(
  TARGETS
    triton-tflite-backend
  EXPORT
    triton-tflite-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
)

if (ARMNN_DELEGATE_ENABLE)
  # Install ArmNN libraries and license
  install(
    DIRECTORY
      ${ARMNN_LOCATION}/lib/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
    FILES_MATCHING PATTERN "*.so*"
  )

  install(
    FILES
      ${ARMNN_LOCATION}/src/armnn/LICENSE
    RENAME armnn.LICENSE
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
  )
endif()

# Install Tensorflow license
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/tensorflow-lite-src/LICENSE
  RENAME tensorflow.LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/tflite
)

install(
  EXPORT
    triton-tflite-backend-targets
  FILE
    TritonTFLiteBackendTargets.cmake
  NAMESPACE
    TritonTFLiteBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonTFLiteBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTFLiteBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTFLiteBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-tflite-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonTFLiteBackendTargets.cmake
  NAMESPACE TritonTFLiteBackend::
)

export(PACKAGE TritonTFLiteBackend)
