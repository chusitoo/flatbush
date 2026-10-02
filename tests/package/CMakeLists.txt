cmake_minimum_required(VERSION 3.12)

project(flatbush_package_test LANGUAGES CXX)

find_package(flatbush CONFIG REQUIRED)

if(DEFINED FLATBUSH_EXPECTED_VERSION)
	if(NOT "${flatbush_VERSION}" STREQUAL "${FLATBUSH_EXPECTED_VERSION}")
		message(FATAL_ERROR
			"Installed flatbush version ${flatbush_VERSION} does not match release "
			"${FLATBUSH_EXPECTED_VERSION}. Update project(VERSION ...) in CMakeLists.txt "
			"and commit the change before tagging the release."
		)
	endif()
	find_package(flatbush "${FLATBUSH_EXPECTED_VERSION}" EXACT CONFIG REQUIRED)
endif()

add_executable(package_test main.cpp)
target_link_libraries(package_test PRIVATE flatbush::flatbush)

enable_testing()
add_test(NAME package_test COMMAND package_test)