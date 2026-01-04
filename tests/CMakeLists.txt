if(NOT ZERO_BUILD_TESTS)
    return()
endif()

# Simple test without external dependencies
add_executable(zero_basic_test
    basic_test.cpp
)

target_link_libraries(zero_basic_test PRIVATE zero-core)

add_test(NAME ZeroBasicTest COMMAND zero_basic_test)
