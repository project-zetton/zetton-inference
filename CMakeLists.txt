cmake_minimum_required(VERSION 3.5)

# ##############################################################################
# CMake Project ##
# ##############################################################################

# include useful macros
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake)

# read metadata from package.xml and setup project
zetton_cc_metadata()

# set project name and version
project(
  ${extracted_name}
  VERSION ${extracted_version}
  LANGUAGES C CXX)

# =========#
# Settings #
# =========#

# introduce default building settings
zetton_cc_settings()

# =============#
# Dependencies #
# =============#

# ----------------------#
# Third-party libraries #
# ----------------------#

find_package(Threads)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

# ---------------#
# CUDA libraries #
# ---------------#

set(USE_GPU
    ON
    CACHE BOOL "Use GPU")
if(USE_GPU)
  add_definitions(-DUSE_GPU=1)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
else()
  add_definitions(-DUSE_GPU=0)
endif()

# -----------------#
# Zetton libraries #
# -----------------#

find_package(zetton_common REQUIRED)

# ##############################################################################
# Build #
# ##############################################################################

# ==============#
# Build targets #
# ==============#

include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# ---------------#
# Library target #
# ---------------#

# find all source files
file(GLOB_RECURSE src_files ${PROJECT_SOURCE_DIR}/src/*.cc
     ${PROJECT_SOURCE_DIR}/src/*/*.cc)

if(USE_GPU)
  if(NOT CUDA_VERSION VERSION_LESS 11.0)
    # CUDA 11.0 removes nppicom
    zetton_list_filterout(CUDA_nppi_LIBRARY "nppicom")
    zetton_list_filterout(CUDA_npp_LIBRARY "nppicom")
  endif()
  list(APPEND EXTRA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
endif()

# common library
zetton_cc_library(
  NAME
  ${PROJECT_NAME}
  SRCS
  ${src_files}
  INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  DEPS
  ${EXTRA_LIBRARIES}
  ${OpenCV_LIBS}
  Threads::Threads
  Eigen3::Eigen
  zetton_common::zetton_common)

# ------------------#
# Executable targets #
# ------------------#

zetton_cc_apps()
zetton_cc_examples()

# -------------#
# Test targets #
# -------------#

zetton_cc_tests("inference")
