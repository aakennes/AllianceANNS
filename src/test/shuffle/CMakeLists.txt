if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    alliance_cpp_test(test_shuffle_memory "linux;layout;regression;memory")
    alliance_cpp_test(test_shuffle_artifacts "linux;layout;storage;regression")
    alliance_python_test(test_shuffle_cli ${CMAKE_CURRENT_SOURCE_DIR}/test_shuffle_cli.py
        "linux;layout;cli" $<TARGET_FILE:alliance_shuffle_disk>)
endif()
