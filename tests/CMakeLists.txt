if(NOT ZERO_BUILD_TESTS)
    return()
endif()

# Basic test
add_executable(zero_basic_test basic_test.cpp)
target_link_libraries(zero_basic_test PRIVATE zero-core)
add_test(NAME ZeroBasicTest COMMAND zero_basic_test)

# Benchmark test
add_executable(zero_benchmark benchmark_test.cpp)
target_link_libraries(zero_benchmark PRIVATE zero-core)
add_test(NAME ZeroBenchmark COMMAND zero_benchmark)
