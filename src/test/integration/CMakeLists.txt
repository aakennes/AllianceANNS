if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
endif()
alliance_cpp_test(test_mixed_pages "integration;storage;layout;search;linux;odirect"
    SOURCES test_mixed_pages.cpp ../support/io_observer.cpp)
alliance_cpp_test(test_ml_only "integration;storage;search;linux;odirect"
    SOURCES test_ml_only.cpp ${PROJECT_SOURCE_DIR}/src/tools/benchmark/io_observer.cpp
        ${PROJECT_SOURCE_DIR}/src/tools/search/search_disk_options.cpp)
alliance_python_test(test_build_completion ${CMAKE_CURRENT_SOURCE_DIR}/test_build_completion.py
    "integration;build;storage;linux" --build $<TARGET_FILE:alliance_build_disk>
    --verifier $<TARGET_FILE:alliance_verify_layout>)
alliance_python_test(test_build_artifacts ${CMAKE_CURRENT_SOURCE_DIR}/test_build_artifacts.py
    "integration;build;storage;linux" --build $<TARGET_FILE:alliance_build_disk>
    --shuffle $<TARGET_FILE:alliance_shuffle_disk> --search $<TARGET_FILE:alliance_search_disk>
    --verifier $<TARGET_FILE:alliance_verify_layout> --finalizer ${PROJECT_SOURCE_DIR}/scripts/finalize_build.py)
alliance_python_test(test_disk_scripts ${PROJECT_SOURCE_DIR}/scripts/tests/test_disk_scripts.py "cli;scripts;linux")
alliance_python_test(test_selectivity_static_cache ${PROJECT_SOURCE_DIR}/scripts/tests/test_selectivity_static_cache.py
    "cli;scripts;cache;linux")
alliance_python_test(test_build_workflow ${PROJECT_SOURCE_DIR}/scripts/tests/test_build_workflow.py
    "cli;scripts;build;linux")
alliance_python_test(test_navigation_workflow ${PROJECT_SOURCE_DIR}/scripts/tests/test_navigation_workflow.py
    "cli;scripts;search;linux")
alliance_python_test(test_cluster_scratch_cleanup ${PROJECT_SOURCE_DIR}/scripts/tests/test_cluster_scratch_cleanup.py
    "cli;scripts;build;storage;linux")
