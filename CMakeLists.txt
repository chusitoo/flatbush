cmake_minimum_required(VERSION 3.12)

project(flatbush VERSION 1.0.0 LANGUAGES CXX)

set(FLATBUSH_TARGET ${PROJECT_NAME})

option(WITH_SPAN "Use custom span if not available in the C++ standard" ON)
option(WITH_TESTS "Build unit tests" OFF)
option(WITH_BENCHMARKS "Build benchmarks" OFF)

include(GNUInstallDirs)

add_library(${FLATBUSH_TARGET} INTERFACE)

target_include_directories(${FLATBUSH_TARGET} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  "$<INSTALL_INTERFACE:include>"
)

if(WITH_SPAN)
  add_compile_options(-DFLATBUSH_SPAN)
endif()

if(WITH_BENCHMARKS)
  set(CMAKE_BUILD_TYPE Release)

  include(FetchContent)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  set(bench_target bench)
  add_executable(${bench_target} "bench.cpp")
  target_link_libraries(${bench_target} PRIVATE benchmark::benchmark)
endif()

if(WITH_TESTS)
  set(CMAKE_BUILD_TYPE Debug)

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  include(GoogleTest)
  enable_testing()

  set(test_target unit_test)
  add_executable(${test_target} "test.cpp")
  target_link_libraries(${test_target} PRIVATE gtest_main)
  gtest_discover_tests(${test_target})
endif()

if(WITH_FUZZING)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)

  include(FetchContent)
  FetchContent_Declare(
    fuzztest
    GIT_REPOSITORY https://github.com/google/fuzztest.git
    GIT_TAG 2025-08-05
  )
  FetchContent_MakeAvailable(fuzztest)

  include(GoogleTest)
  enable_testing()

  fuzztest_setup_fuzzing_flags()
  set(fuzz_target fuzz_test)
  add_executable(${fuzz_target} "fuzz.cpp")
  link_fuzztest(${fuzz_target})
  gtest_discover_tests(${fuzz_target})
endif()

configure_file(
    "cmake/configVersion.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    @ONLY
)

configure_file(
    "cmake/config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    @ONLY
)

install(FILES flatbush.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${PROJECT_NAME}
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}"
        EXPORT "${FLATBUSH_TARGET}Targets"
)

install(EXPORT "${FLATBUSH_TARGET}Targets"
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}"
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}"
)
