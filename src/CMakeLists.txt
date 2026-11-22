# Components own compilation boundaries; alliance_core remains the compatible link target.
find_package(BLAS REQUIRED)
add_library(alliance_component_options INTERFACE)
target_link_libraries(alliance_component_options INTERFACE alliance_rabitq_external ${BLAS_LIBRARIES})
if(NOT APPLE)
    target_link_libraries(alliance_component_options INTERFACE aio)
endif()

option(ALLIANCE_VALIDATE_CANDIDATES "Check live multi-level candidate ID uniqueness" OFF)
option(ALLIANCE_PIPELINE_DIAGNOSTICS "Enable expensive multi-level profiling and co-block scans" OFF)
option(ALLIANCE_PIPELINE_PROFILE "Enable cheap per-phase timers (no co-block scans)" OFF)
if(ALLIANCE_VALIDATE_CANDIDATES)
    target_compile_definitions(alliance_component_options INTERFACE ALLIANCE_VALIDATE_CANDIDATES)
endif()
if(ALLIANCE_PIPELINE_DIAGNOSTICS)
    target_compile_definitions(alliance_component_options INTERFACE ALLIANCE_PIPELINE_DIAGNOSTICS)
endif()
if(ALLIANCE_PIPELINE_PROFILE OR ALLIANCE_PIPELINE_DIAGNOSTICS)
    target_compile_definitions(alliance_component_options INTERFACE ALLIANCE_PIPELINE_PROFILE)
endif()

add_library(alliance_storage_objects OBJECT
    alliance/index/index_disk.cpp
    alliance/storage/index_disk_metadata.cpp
    alliance/storage/atomic_file.cpp
    alliance/storage/index_disk_storage.cpp
    alliance/quantization/index_disk_rabitq.cpp
)
target_link_libraries(alliance_storage_objects PUBLIC alliance_component_options)
target_include_directories(alliance_storage_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/alliance)

add_library(alliance_build_objects OBJECT
    alliance/disk_build/left_stream/executor.cpp
    alliance/disk_build/stages/api.cpp
    alliance/disk_build/stages/execute.cpp
    alliance/disk_build/stages/resident.cpp
    alliance/disk_build/stages/upper.cpp
    alliance/disk_build/input/populate.cpp
    alliance/disk_build/input/vectors.cpp
    alliance/disk_build/memory/budget.cpp
    alliance/disk_build/memory/process.cpp
    alliance/disk_build/memory/report.cpp
    alliance/disk_build/partition/vamana.cpp
    alliance/disk_build/partition/kmeans.cpp
    alliance/disk_build/partition/assignment.cpp
    alliance/disk_build/partition/membership.cpp
    alliance/disk_build/stages/workspace.cpp
    alliance/disk_build/stages/cluster_prepare.cpp
    alliance/disk_build/stages/cluster_graph.cpp
    alliance/disk_build/cluster_wow/build.cpp
    alliance/disk_build/cluster_wow/commit.cpp
)
target_link_libraries(alliance_build_objects PUBLIC alliance_component_options)
target_include_directories(alliance_build_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/alliance)

add_library(alliance_search_objects OBJECT
    alliance/search/entry_routing.cpp
    alliance/search/search_sync.cpp
    alliance/search/raw_pipeline.cpp
    alliance/search/multilevel_traversal.cpp
    alliance/search/multilevel_edge_access.cpp
    alliance/search/multilevel_query.cpp
    alliance/search/multilevel_rerank.cpp
    alliance/search/disk_search_workspace.cpp
)
target_link_libraries(alliance_search_objects PUBLIC alliance_component_options)
target_include_directories(alliance_search_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/alliance)

add_library(alliance_layout_objects OBJECT
    alliance/layout/index_disk_layout.cpp
    alliance/layout/index_disk_static_cache.cpp
    alliance/shuffle/stages/level_plan.cpp
    alliance/shuffle/stages/orders.cpp
    alliance/shuffle/adjacency/weighted_graph.cpp
    alliance/shuffle/adjacency/scratch_io.cpp
    alliance/shuffle/adjacency/scratch_aio.cpp
    alliance/shuffle/adjacency/external_sort.cpp
    alliance/shuffle/adjacency/external_graph.cpp
    alliance/shuffle/order/block_bnf.cpp
    alliance/shuffle/diagnostics/stats.cpp
    alliance/shuffle/format/write.cpp
    alliance/shuffle/format/plan.cpp
    alliance/shuffle/level_geometry.cpp
    alliance/shuffle/stages/export.cpp
    alliance/shuffle/stages/artifacts.cpp
    alliance/shuffle/stages/verify.cpp
)
target_link_libraries(alliance_layout_objects PUBLIC alliance_component_options)
target_include_directories(alliance_layout_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/alliance)

add_library(alliance_core STATIC
    $<TARGET_OBJECTS:alliance_storage_objects>
    $<TARGET_OBJECTS:alliance_build_objects>
    $<TARGET_OBJECTS:alliance_search_objects>
    $<TARGET_OBJECTS:alliance_layout_objects>
)
target_link_libraries(alliance_core PUBLIC alliance_component_options)

add_subdirectory(tools)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
