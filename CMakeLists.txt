cmake_minimum_required(VERSION 3.16)
project(zero-core-runtime VERSION 1.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
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
option(ZERO_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ZERO_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if(ZERO_ENABLE_ASAN AND NOT MSVC)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(ZERO_ENABLE_UBSAN AND NOT MSVC)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

if(ZERO_ENABLE_TSAN AND NOT MSVC)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

# Build metadata
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ZERO_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT ZERO_GIT_COMMIT)
    set(ZERO_GIT_COMMIT "unknown")
endif()
string(TIMESTAMP ZERO_BUILD_DATE "%Y-%m-%d")

# Header-only library
add_library(zero-core INTERFACE)
target_include_directories(zero-core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(zero-core INTERFACE
    ZERO_VERSION="${PROJECT_VERSION}"
    ZERO_GIT_COMMIT="${ZERO_GIT_COMMIT}"
    ZERO_BUILD_DATE="${ZERO_BUILD_DATE}"
)

# Tests
if(ZERO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(DIRECTORY include/zero DESTINATION include)
install(TARGETS zero-core EXPORT zero-core-targets)
