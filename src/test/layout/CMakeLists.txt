if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    alliance_cpp_test(test_layout_export "linux;regression;layout")
    alliance_cpp_test(test_shuffle_contract "linux;regression;layout")
endif()
