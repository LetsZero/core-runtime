cmake_minimum_required(VERSION 3.16)
project(zero-core-runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Options
option(ZERO_BUILD_TESTS "Build unit tests" ON)
option(ZERO_ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ZERO_ENABLE_ASAN AND NOT MSVC)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# Header-only library
add_library(zero-core INTERFACE)
target_include_directories(zero-core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
if(ZERO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(DIRECTORY include/zero DESTINATION include)
install(TARGETS zero-core EXPORT zero-core-targets)
