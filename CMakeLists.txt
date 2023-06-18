cmake_minimum_required(VERSION 3.12)

project(Flatbush VERSION 1.0.0 LANGUAGES CXX)

set(this_target flatbush)

option(WITH_SPAN "Use custom span if not available in the C++ standard" ON)
option(WITH_TESTS "Build unit tests" ON)
option(WITH_BENCHMARKS "Build benchmarks" ON)

include(GNUInstallDirs)
add_library(${this_target} INTERFACE)

target_include_directories(${this_target} INTERFACE 
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  "$<INSTALL_INTERFACE:include>"
)

if(WITH_SPAN)
  add_compile_options(-DFLATBUSH_SPAN)
endif()

if(WITH_BENCHMARKS)
  set(BENCHMARKS bench)
  add_executable(${BENCHMARKS} "bench.cpp")
endif()

if(WITH_TESTS)
  set(UNIT_TESTS unit_test)
  add_executable(${UNIT_TESTS} "test.cpp")

  enable_testing()
  add_test(NAME ${UNIT_TESTS} COMMAND ${UNIT_TESTS})
endif()

install(FILES flatbush.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
