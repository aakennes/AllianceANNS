alliance_cpp_test(test_entry_sampling "unit;search")
alliance_cpp_test(test_memory_navigation "unit;search;compatibility")
alliance_cpp_test(test_rerank_contract "unit;search")
alliance_cpp_test(test_search_observability "unit;search" DEFINITIONS ALLIANCE_PIPELINE_PROFILE)
alliance_cpp_test(test_search_options "unit;search;cli"
    SOURCES test_search_options.cpp
        ${PROJECT_SOURCE_DIR}/src/tools/search/search_disk_options.cpp
        ${PROJECT_SOURCE_DIR}/src/tools/search/search_preparation.cpp)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(target test_multilevel_search test_prefilter test_rerank_estimator test_search_worker test_cache_contract)
        alliance_cpp_test(${target} "linux;regression;search")
    endforeach()
    alliance_cpp_test(test_disk_search_workspace "unit;linux;search;lifetime"
        SOURCES test_disk_search_workspace.cpp ../support/test_disk_search_workspace_peer.cpp
        WRAP posix_memalign free)
endif()
