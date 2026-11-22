if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
endif()
foreach(target test_paged_file test_id_only_edges test_variable_pages test_disk_metadata_reader)
    alliance_cpp_test(${target} "linux;regression;storage")
endforeach()
alliance_cpp_test(test_disk_index_lifetime "unit;linux;build;storage;lifetime")
alliance_cpp_test(test_io_handler_failure "unit;linux;io;storage;search" DEFINITIONS ALLIANCE_PIPELINE_PROFILE)
alliance_cpp_test(test_io_operation_lifetime "unit;linux;io;storage;build;search;lifetime"
    WRAP io_submit io_getevents io_destroy posix_memalign aligned_alloc)
alliance_cpp_test(test_checkpoint_store "unit;linux;build;storage;checkpoint"
    WRAP write fsync renameat close)
