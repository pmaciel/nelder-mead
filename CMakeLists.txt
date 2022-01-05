cmake_minimum_required(VERSION 3.10)

project(nm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(nm Nelder-Mead.cc)
target_link_libraries(nm Eigen3::Eigen)

