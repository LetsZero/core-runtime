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

# Activation tests (v1.1)
add_executable(zero_activation_test test_activations.cpp)
target_link_libraries(zero_activation_test PRIVATE zero-core)
add_test(NAME ZeroActivationTest COMMAND zero_activation_test)

# FP8 dtype tests (spec 001)
add_executable(zero_dtype_fp8_test test_dtype_fp8.cpp)
target_link_libraries(zero_dtype_fp8_test PRIVATE zero-core)
add_test(NAME ZeroDTypeFP8Test COMMAND zero_dtype_fp8_test)
