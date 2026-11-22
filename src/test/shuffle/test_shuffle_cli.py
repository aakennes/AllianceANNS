"""CLI contract checks: invalid invocations must fail before any file is opened."""
import subprocess
import sys


def run(binary, args, expected=None):
    result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=15)
    if expected is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and expected in result.stderr, (args, result.returncode, result.stderr)
    return result


def main(binary):
    help_text = run(binary, ["--help"]).stdout
    assert "max_iterations 32" in help_text and "level_k 2" in help_text
    required = ["--graph_file", "/absent/input", "--meta_file", "/absent/meta",
                "--out_graph_file", "/absent/output", "--out_meta_file", "/absent/output.meta"]
    for key in ("--graph_file", "--block_size", "--method", "--move_margin"):
        run(binary, [key], "missing value")
    for args, message in [
        (["--method", "affinity_bnf"], "retired or unknown"),
        (["--method", "weight_sorted"], "retired or unknown"),
        (["--move_margin", "1"], "retired option"),
        (["--min_colocatable_for_bnf", "0"], "retired option"),
        (["--static_cache_budget", "1"], "retired option"),
        (["--static_landing_file", "x"], "retired option"),
        (["--block_size", "12288"], "power"),
        (["--vector_block_size", "256"], "power"),
        (["--max_iterations", "0"], "iterations"),
        (["--level_k", "0"], "level_k"),
        (["--shuffle_o", "3"], "shuffle_o"),
        (["--shuffle_threads", "257"], "shuffle_threads"),
        (["--affinity_batch_size", "0"], "affinity_batch_size"),
        (["--shuffle_mem_gb", "nan"], "finite"),
        (["--shuffle_storage", "mmap"], "shuffle_storage"),
        (["--shuffle_storage", "external"], "requires"),
        (["--shuffle_storage", "auto"], "requires"),
        (["--shuffle_storage", "external", "--shuffle_memory_mib", "0"], "positive"),
        (["--shuffle_storage", "external", "--shuffle_memory_mib", "600", "--method", "identity"], "requires block_bnf"),
        (["--block_size", "4096junk"], "integer"),
        (["--block_size", "-1"], "integer"),
        (["--block_size", "4096", "--block_size", "4096"], "duplicate"),
    ]:
        run(binary, required + args, message)
    for flags in (["--shuffle_storage", "memory"],
                  ["--shuffle_storage", "memory", "--shuffle_memory_mib", "600"],
                  ["--shuffle_storage", "auto", "--shuffle_memory_mib", "600"]):
        result = subprocess.run([binary, *required, *flags], capture_output=True, text=True)
        assert result.returncode != 0 and "requires" not in result.stderr, result.stderr
    print("shuffle CLI contract passed")


if __name__ == "__main__":
    main(sys.argv[1])
