# Entered only with BUILD_TESTING=ON; production targets never depend on tests.
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(support/TestTargets.cmake)

alliance_python_test(test_repository_structure
    ${PROJECT_SOURCE_DIR}/scripts/tests/test_repository_structure.py "unit;structure;scripts")

add_subdirectory(build)
add_subdirectory(storage)
add_subdirectory(layout)
add_subdirectory(shuffle)
add_subdirectory(quantization)
add_subdirectory(search)
add_subdirectory(benchmark)
add_subdirectory(integration)
