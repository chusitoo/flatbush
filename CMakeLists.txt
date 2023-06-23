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
  set(bench_target bench)
  add_executable(${bench_target} "bench.cpp")
  set_target_properties(${test_target} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
  )
endif()

if(WITH_TESTS)
  set(test_target unit_test)
  add_executable(${test_target} "test.cpp")
  set_target_properties(${test_target} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
  )

  enable_testing()
  add_test(NAME ${test_target} COMMAND ${test_target})
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
