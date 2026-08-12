cmake_minimum_required(VERSION 3.22)

project(operatorlib VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(operatorlib STATIC
  src/reduce.cu
  src/softmax.cu
  src/transpose.cu
  src/rmsnorm.cu
  src/gemm.cu
  src/attention.cu
)

target_include_directories(operatorlib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(operatorlib PUBLIC CUDA::cudart)
target_compile_options(operatorlib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
set_target_properties(operatorlib PROPERTIES
  CUDA_ARCHITECTURES 87
  POSITION_INDEPENDENT_CODE ON
)

option(OPERATORLIB_BUILD_TESTS "Build standalone launcher smoke tests" ON)
if(OPERATORLIB_BUILD_TESTS)
  enable_testing()

  add_executable(operatorlib_core_smoke tests/core_ops_smoke.cu)
  target_link_libraries(operatorlib_core_smoke PRIVATE operatorlib)
  set_target_properties(operatorlib_core_smoke PROPERTIES CUDA_ARCHITECTURES 87)
  add_test(NAME operatorlib_core_smoke COMMAND operatorlib_core_smoke)

  add_executable(operatorlib_gemm_smoke tests/gemm_smoke.cu)
  target_link_libraries(operatorlib_gemm_smoke PRIVATE operatorlib CUDA::cublas)
  set_target_properties(operatorlib_gemm_smoke PROPERTIES CUDA_ARCHITECTURES 87)
  add_test(NAME operatorlib_gemm_smoke COMMAND operatorlib_gemm_smoke)

  add_executable(operatorlib_attention_smoke tests/attention_smoke.cu)
  target_link_libraries(operatorlib_attention_smoke PRIVATE operatorlib)
  set_target_properties(operatorlib_attention_smoke PROPERTIES CUDA_ARCHITECTURES 87)
  add_test(NAME operatorlib_attention_smoke COMMAND operatorlib_attention_smoke)
endif()
