alliance_cpp_test(test_benchmark_input "unit;benchmark;protocol")
# Python tests remain with their owning script package; registration belongs here.
foreach(name test_legacy_log test_benchmark_metrics test_caselaw_profiles test_caselaw_adapter
        test_benchmark_spec_export test_benchmark_runner test_benchmark_validation
        test_benchmark_integration test_benchmark_cli test_benchmark_datasets)
    alliance_python_test(${name} ${PROJECT_SOURCE_DIR}/scripts/tests/${name}.py "unit;benchmark;protocol")
endforeach()
