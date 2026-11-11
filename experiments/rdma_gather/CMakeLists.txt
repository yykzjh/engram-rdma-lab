cmake_minimum_required(VERSION 3.18)
project(engram_rdma_bench LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
add_executable(engram_rdma_bench benchmark.cpp)
target_compile_features(engram_rdma_bench PRIVATE cxx_std_17)
target_compile_options(engram_rdma_bench PRIVATE -O3 -Wall -Wextra)
target_link_libraries(engram_rdma_bench PRIVATE ibverbs mlx5 CUDA::cuda_driver CUDA::nvrtc Threads::Threads)

option(ENGRAM_BUILD_GPU_REQUESTER "Build the experimental NCCL GIN GPU requester" OFF)
set(ENGRAM_GPU_ARCH compute_100 CACHE STRING "PTX target architecture")
if(ENGRAM_BUILD_GPU_REQUESTER)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_path(NCCL_INCLUDE_DIR nccl_device.h HINTS ${NCCL_ROOT}/include REQUIRED)
  find_library(NCCL_LIBRARY nccl HINTS ${NCCL_ROOT}/lib ${NCCL_ROOT}/lib64 REQUIRED)
  file(GLOB_RECURSE NCCL_DEVICE_HEADERS ${NCCL_INCLUDE_DIR}/nccl_device/*.h ${NCCL_INCLUDE_DIR}/nccl_device/*.cuh)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_request.ptx
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -std=c++17 -O3 -lineinfo
      -arch=${ENGRAM_GPU_ARCH} -ptx -I${NCCL_INCLUDE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/gpu_request.cu -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_request.ptx
    DEPENDS gpu_request.cu ${NCCL_DEVICE_HEADERS} VERBATIM)
  add_custom_target(engram_gpu_ptx DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gpu_request.ptx)
  add_executable(engram_gpu_request gpu_request.cpp)
  add_dependencies(engram_gpu_request engram_gpu_ptx)
  target_compile_features(engram_gpu_request PRIVATE cxx_std_17)
  target_compile_options(engram_gpu_request PRIVATE -O3 -Wall -Wextra -Wno-missing-field-initializers)
  target_include_directories(engram_gpu_request PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}/cccl)
  target_link_libraries(engram_gpu_request PRIVATE MPI::MPI_CXX ${NCCL_LIBRARY} ibverbs mlx5 CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
endif()
