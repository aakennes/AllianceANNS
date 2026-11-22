# Independent upstream dependency; never resolve kernels from the WoW snapshot.
include(FetchContent)
set(ALLIANCE_RABITQ_SOURCE_DIR "" CACHE PATH "Existing official RaBitQ checkout at the pinned commit")
set(ALLIANCE_RABITQ_COMMIT e8fdff963509d527cbe870bb683776b6c03601fd)
function(alliance_add_external_rabitq)
  # Keep legacy global ISA/include flags out of the external library.
  set(CMAKE_CXX_FLAGS "")
  foreach(component FASTSCAN INDEX SAMPLES TESTS BENCHMARKS PYTHON_BINDINGS)
    set(RABITQ_BUILD_${component} OFF)
  endforeach()
  set(RABITQ_USE_OPENMP OFF)
  set(RABITQ_ENABLE_NATIVE_OPTIMIZATION OFF)
  if(ALLIANCE_RABITQ_SOURCE_DIR)
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${ALLIANCE_RABITQ_SOURCE_DIR} rev-parse HEAD
      OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(NOT revision STREQUAL ALLIANCE_RABITQ_COMMIT)
      message(FATAL_ERROR "External RaBitQ must be v0.7.1 commit ${ALLIANCE_RABITQ_COMMIT}")
    endif()
    add_subdirectory(${ALLIANCE_RABITQ_SOURCE_DIR} ${CMAKE_BINARY_DIR}/_deps/rabitq-build EXCLUDE_FROM_ALL)
  else()
    FetchContent_Declare(alliance_rabitq
      GIT_REPOSITORY https://github.com/VectorDB-NTU/RaBitQ-Library.git
      GIT_TAG ${ALLIANCE_RABITQ_COMMIT})
    FetchContent_MakeAvailable(alliance_rabitq)
  endif()
  # Old quantization/checkpoint code still uses the embedded namespace. Hide all
  # upstream symbols inside this DSO to prevent cross-version symbol interposition.
  add_library(alliance_rabitq_external SHARED ${CMAKE_SOURCE_DIR}/src/alliance/quantization/external_rabitq.cpp)
  target_include_directories(alliance_rabitq_external PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(alliance_rabitq_external PRIVATE rabitqlib::core)
  set_target_properties(alliance_rabitq_external PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  if(UNIX AND NOT APPLE)
    target_link_options(alliance_rabitq_external PRIVATE "LINKER:--exclude-libs,ALL")
  elseif(APPLE)
    target_link_options(alliance_rabitq_external PRIVATE
      "LINKER:-exported_symbol,_alliance_rabitq_dot"
      "LINKER:-exported_symbol,_alliance_rabitq_pack"
      "LINKER:-exported_symbol,_alliance_rabitq_unpack"
      "LINKER:-exported_symbol,_alliance_rabitq_version")
  endif()
endfunction()
alliance_add_external_rabitq()
