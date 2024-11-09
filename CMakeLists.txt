cmake_minimum_required(VERSION 3.29)
project(Programming_Parallel_Computers_GPU)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(Programming_Parallel_Computers_GPU main.cpp)
target_link_libraries(Programming_Parallel_Computers_GPU OpenCL::OpenCL)
target_link_libraries(Programming_Parallel_Computers_GPU OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
