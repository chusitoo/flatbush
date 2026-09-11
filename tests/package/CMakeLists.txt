cmake_minimum_required(VERSION 3.12)

project(flatbush_package_test LANGUAGES CXX)

find_package(flatbush CONFIG REQUIRED)

add_executable(package_test main.cpp)
target_link_libraries(package_test PRIVATE flatbush::flatbush)

enable_testing()
add_test(NAME package_test COMMAND package_test)