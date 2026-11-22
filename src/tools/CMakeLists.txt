# Production entry points and index validation/preparation tools.
add_executable(alliance_gengt dataset/gen_ground_truth.cpp)
add_executable(alliance_build_disk
    build/build_disk.cpp
    build/build_disk_options.cpp
    build/build_disk_runner.cpp
    build/build_disk_plan.cpp
    build/disk/build_inputs.cpp
    build/disk/prepared_index.cpp
)
set(ALLIANCE_SEARCH_PREPARATION_SOURCES
    search/search_disk_options.cpp search/search_preparation.cpp)
add_executable(alliance_search_disk search/search_disk.cpp search/search_disk_runner.cpp
    search/search_disk_report.cpp ${ALLIANCE_SEARCH_PREPARATION_SOURCES})
add_executable(alliance_shuffle_disk layout/shuffle_disk.cpp layout/shuffle/options.cpp layout/shuffle/runner.cpp)
add_executable(alliance_prepare_navigation layout/prepare_navigation.cpp)
add_executable(alliance_prepare_static_cache layout/prepare_static_cache.cpp)
add_executable(alliance_collect_landing layout/collect_landing.cpp)
add_executable(verify_index validation/verify_index.cpp)
add_executable(alliance_verify_layout validation/verify_layout.cpp)
add_executable(alliance_benchmark benchmark/benchmark.cpp benchmark/io_observer.cpp
    benchmark/session.cpp benchmark/batch.cpp ${ALLIANCE_SEARCH_PREPARATION_SOURCES})
target_link_libraries(alliance_benchmark PRIVATE ${CMAKE_DL_LIBS})
foreach(target alliance_gengt alliance_build_disk alliance_search_disk alliance_shuffle_disk
        alliance_prepare_navigation alliance_prepare_static_cache alliance_collect_landing verify_index alliance_verify_layout alliance_benchmark)
    target_link_libraries(${target} PRIVATE alliance_core)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/alliance)
endforeach()

option(ALLIANCE_BUILD_DIAGNOSTICS "Build standalone storage performance diagnostics" OFF)
if(ALLIANCE_BUILD_DIAGNOSTICS)
    add_executable(bench_iops diagnostics/bench_iops.cpp)
    target_link_libraries(bench_iops PRIVATE aio pthread)
endif()
