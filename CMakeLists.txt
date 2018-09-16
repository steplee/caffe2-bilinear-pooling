cmake_minimum_required(VERSION 3.4)

project(BilinearOp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

#find_package(OpenCV 3.3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

list(APPEND Eigen_INCLUDE_DIRS /usr/include/eigen3)

#list(APPEND OpenCV_LIB opencv_xfeatures2d)

add_definitions(-lpthread -g)

# Caffe2 stuff
find_library(CAFFE2_LIB caffe2 REQUIRED)
find_library(CAFFE2_GPU_LIB caffe2_gpu REQUIRED)
#find_library(GLOG_LIB NAMES glog HINTS /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(NCCL_LIB nccl)
find_library(GFLAGS_LIB gflags)
#find_package(CUDA REQUIRED)
#find_package(Boost REQUIRED serialization)
find_package(Catch2 REQUIRED)

message(STATUS "Caffe2 lib: ${CAFFE2_LIB}")
message(STATUS "Caffe2 gpu lib: ${CAFFE2_GPU_LIB}")
message(STATUS "Caffe2 incl: ${CAFFE2_INCLUDE_DIRS}")
#message(STATUS "GLOG lib: ${GLOG_LIB} glog::glog")
message(STATUS "OpenCV lib: ${OpenCV_LIBRARIES}")
message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
message(STATUS "CUDA incl: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuRAND lib: ${CUDA_curand_LIBRARY}")
message(STATUS "Boost incl: ${Boost_INCLUDE_DIR}")
message(STATUS "GFlags: ${GFLAGS_LIB}")
message(STATUS "Proto: ${PROTOBUF_LIBRARY}")

add_library(bilinear_op SHARED
  src/bilinear_op_gpu.cu
  src/bilinear_op.cc
)



foreach (lib bilinear_op)
    target_include_directories(${lib}
      PUBLIC
      /usr/local/cuda/include
      ${MY_CAFFE2_INCLUDE_DIRS}
      ${Boost_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIRS}
      ${Eigen_INCLUDE_DIRS}
      ${CAFFE2_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src/
      ${OpenCV_INCLUDE_DIRS})

  target_link_libraries(${lib}
    PUBLIC
      ${CAFFE2_LIB}
      ${CAFFE2_GPU_LIB}
      ${Boost_LIBRARIES}
      ${NCCL_LIB}
      ${CUDA_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      ${CMAKE_THREAD_LIBS_INIT}
      ${GFLAGS_LIB}
      #${GLOG_LIB}
      ${PROTOBUF_LIBRARY}
      ${OpenCV_LIBRARIES})
  set_property(TARGET ${lib} PROPERTY CUDA_STANDARD 11)

endforeach(lib)


# A little test app
add_executable(tests
    src/test/test_main.cc
    src/test/test_op.cc
)

add_executable(storage_order
  src/test/find_storage_order.cc
)
target_link_libraries(storage_order
      ${CAFFE2_LIB}
      ${CAFFE2_GPU_LIB}
)
target_include_directories( storage_order PUBLIC
  /usr/include/eigen3
)

message(STATUS ${CUDA_LIBRARIES} CUDAAA)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build)
target_link_libraries(tests
  PUBLIC
  ${CUDA_LIBRARIES}
  ${CUDA_LIB}
  ${CAFFE2_LIB}
  ${CAFFE2_GPU_LIB}
  cudart
  -Wl,--whole-archive,--no-as-needed bilinear_op -Wl,--no-whole-archive
  Catch2::Catch2
  )

target_include_directories(tests
    PUBLIC
    ${MY_CAFFE2_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${CAFFE2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${OpenCV_INCLUDE_DIRS})
