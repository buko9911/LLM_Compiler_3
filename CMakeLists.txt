cmake_minimum_required(VERSION 3.24)
project(PlenaCompiler VERSION 0.1.0 LANGUAGES C CXX)

# Large static graphs spend substantial time in compiler-side analysis.
# Single-configuration builds should not silently ship an unoptimized compiler.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(PLENA_TARGET_ONLY "Build target/analysis libraries, assembler and unit tests without MLIR" OFF)
if(PLENA_TARGET_ONLY)
  add_subdirectory(lib/Target)
  add_subdirectory(lib/Analysis)
  add_subdirectory(tools/plena-asm)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(unittests)
  endif()
  return()
endif()

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/plena)
add_subdirectory(lib)
add_subdirectory(tools)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()
