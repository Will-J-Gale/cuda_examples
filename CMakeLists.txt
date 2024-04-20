cmake_minimum_required(VERSION 3.16)
project(cuda_test)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC 
    ${CUDA_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    include
)

cuda_add_library(
    ${PROJECT_NAME}_kernels 
    src/vector_add.cu
    src/conv2d.cu
)


target_link_libraries(
    ${PROJECT_NAME}_kernels
    ${OpenCV_LIBRARIES}
)

target_link_libraries(
    ${PROJECT_NAME} 
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBRARIES}
    ${PROJECT_NAME}_kernels)