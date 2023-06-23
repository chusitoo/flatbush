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
  set(BENCHMARKS bench)
  add_executable(${BENCHMARKS} "bench.cpp")
  target_compile_features(${BENCHMARKS} INTERFACE cxx_std_11)
endif()

if(WITH_TESTS)
  set(compiler_supports_cpp_11 TRUE)

  foreach(feature ${CMAKE_CXX_COMPILE_FEATURES})
      if (${feature} STREQUAL cxx_std_14)
          set(compiler_supports_cpp_14 TRUE)
      elseif (${feature} STREQUAL cxx_std_17)
          set(compiler_supports_cpp_17 TRUE)
      elseif (${feature} STREQUAL cxx_std_20)
          set(compiler_supports_cpp_20 TRUE)
      elseif (${feature} STREQUAL cxx_std_23)
          set(compiler_supports_cpp_23 TRUE)
      endif()
  endforeach()

  set(UNIT_TESTS unit_test)
  add_executable(${UNIT_TESTS} "test.cpp")
  target_compile_features(${UNIT_TESTS} INTERFACE cxx_std_11)

  enable_testing()
  add_test(NAME ${UNIT_TESTS} COMMAND ${UNIT_TESTS})
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
