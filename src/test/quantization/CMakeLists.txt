alliance_cpp_test(test_packed_rabitq_storage "unit;quantization")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    alliance_cpp_test(test_disk_build_quantization "linux;regression;build;quantization")
endif()
