"""Synthetic archive replay tests; never open production inputs or run ANN."""

import csv
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib.validation import ValidationError, preflight_curve, verify_curve


class Fixture:
    def __init__(self, root, algorithm="allianceann", short=False):
        self.root = Path(root)
        self.directory = self.root / "raw" / "r1" / "ef16"
        self.directory.mkdir(parents=True)
        self.spec = {"memory": {"loading_limit_bytes": 9000, "search_rss_limit_bytes": 3000},
                     "aggregation": {"policy": "mean", "repeats": 1}}
        self.curve = {"id": algorithm + "-origin", "algorithm": algorithm, "workload": "origin",
                      "cache_policy": "native", "efs": [16], "repeats": 1, "queries": 2,
                      "k": 2, "bits": 1, "inputs": {}, "expected": {"measured": {}}}
        for key, name in (("base", "base.bin"), ("query", "query.fvecs"), ("ranges", "ranges.bin"),
                          ("gt", "gt.ivecs"), ("attributes", "attributes.bin")):
            self.curve["inputs"][key] = str(self.root / name)
        (self.root / "base.bin").write_bytes(struct.pack("<II8f", 4, 2, 0, 0, 1, 0, 2, 0, 3, 0))
        (self.root / "query.fvecs").write_bytes(struct.pack("<IffIff", 2, 0, 0, 2, 2, 0))
        (self.root / "ranges.bin").write_bytes(struct.pack("<4i", 0, 3, 0, 3))
        (self.root / "gt.ivecs").write_bytes(struct.pack("<6I", 2, 0, 1, 2, 2, 1))
        (self.root / "attributes.bin").write_bytes(struct.pack("<4i", 0, 1, 2, 3))
        self.results = [[(0, 0.0), (1, 1.0)], [(2, 0.0), (1, 1.0)]]
        if short:
            self.results[1] = self.results[1][:1]
        matches = [len(row) for row in self.results]
        self.measured = {"ef": 16, "repeat": 1, "queries": 2, "matches": sum(matches), "recall": sum(matches)/4,
                         "batch_wall_seconds": 1.0, "qps": 2.0, "p95_ms": 200.0,
                         "loading_peak_rss_bytes": 4000, "search_rss_limit_bytes": 3000,
                         "rss_budget_passed": True}
        if algorithm == "allianceann":
            self.observations = [dict(query_id=i, latency_seconds=0.1*(i+1), requests=2, completions=2,
                                      bytes=12288, graph_io=1, rerank_io=1, hops=1, matches=matches[i]) for i in range(2)]
            self.measured.update(bits=1, k=2, io_requests=4, io_completions=4, io_bytes=24576,
                                 edge_block_bytes=4096, vector_block_bytes=8192, graph_io=2, rerank_io=2,
                                 observer_graph_requests=2, observer_vector_requests=2,
                                 prefilter_queries=0, graph_queries=2, dynamic_cache_clear_per_query=False,
                                 dynamic_cache_clear_per_ef=True, dynamic_cache_auto_size=False)
        elif algorithm == "pipewow":
            self.observations = [dict(query_id=i, latency_seconds=0.1*(i+1), requests=1, completions=1,
                                      bytes=8192, cache_hits=2, cache_lookups=3, matches=matches[i]) for i in range(2)]
            self.measured.update(bits=1, k=2, io_requests=2, io_completions=2, io_bytes=16384, native_io=2,
                                 page_bytes=8192, dynamic_hits=4, dynamic_lookups=6, cache_clear_per_query=False,
                                 cache_clear_per_ef=True, static_cache_bytes=0)
        else:
            self.observations = [dict(query_id=i, returned=len(self.results[i]), latency_ms=100.0*(i+1),
                                      requests=1, bytes=8192, cache_hits=1, pre=0, matches=matches[i],
                                      **{"in": 1, "post": 0}) for i in range(2)]
            self.measured.update(io_requests=2, io_bytes=16384, completed_bytes=16384, graph_reads=2,
                                 graph_bytes=16384, attribute_reads=0, attribute_bytes=0, attribute_eof_reads=0,
                                 observer_errors=0, observer_verified=True, result_errors=0,
                                 short_queries=int(short), cache_hits=2, pre_filter=0, in_filter=2, post_filter=0,
                                 search_peak_rss_bytes=2000, search_cgroup_peak_bytes=2100,
                                 static_cache_bytes=8192, io_requests_per_query=1.0,
                                 io_4k_equivalent_per_query=2.0)
        if algorithm != "pipeannfilter":
            self.measured["requested_repeats"] = 1
            self.curve["quantization_fingerprint"] = "12345"
            (self.root / "stdout.log").write_text("SEARCH_VERIFIED fingerprint=12345\n", encoding="utf-8")
            self.measured.update({key: 0 for key in ("invalid", "duplicates", "predicate", "short",
                                                    "observer_errors", "unknown_fd", "bad_layout", "pending",
                                                    "cache_entries_before_timing")})
            self.measured.update(search_phase_peak_rss_bytes=2000, loading_cgroup_peak_bytes=4100,
                                 search_phase_cgroup_peak_bytes=2100)
        self.save()

    def save(self):
        (self.directory / "measured.json").write_text(json.dumps(self.measured), encoding="utf-8")
        with (self.directory / "queries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.observations[0]))
            writer.writeheader()
            writer.writerows(self.observations)
        raw = bytearray()
        if self.curve["algorithm"] == "pipeannfilter":
            for row in self.results:
                raw += struct.pack("<I", len(row))
                for node, distance in row:
                    raw += struct.pack("<If", node, distance)
        else:
            raw += struct.pack("<4Q", 0x434153454c415731, 2, 2, 16)
            for row in self.results:
                for node, distance in row:
                    raw += struct.pack("<Qf", node, distance)
        (self.directory / "results.bin").write_bytes(raw)

    def verify(self):
        return verify_curve(self.spec, self.curve, self.root)


class ValidationTests(unittest.TestCase):
    def make_fixture(self, algorithm="allianceann", short=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Fixture(temporary.name, algorithm, short)

    def test_all_native_formats_recompute_metrics(self):
        for algorithm in ("allianceann", "pipewow", "pipeannfilter"):
            with self.subTest(algorithm=algorithm):
                fixture = self.make_fixture(algorithm)
                proof = fixture.verify()
                self.assertEqual(proof["status"], "verified")
                row = proof["rows"][0]
                self.assertIs(row["validated"], True)
                self.assertEqual((row["recall"], row["qps"], row["p95_ms"]), (1.0, 2.0, 200.0))
                self.assertEqual(row["loading_peak_rss_bytes"], 4000)
                self.assertEqual(row["search_peak_rss_bytes"], 2000)
                self.assertEqual(row["io_4k_equivalent_per_query"], row["io_bytes_per_query"] / 4096)
                self.assertEqual(row["io_8k_equivalent_per_query"], row["io_bytes_per_query"] / 8192)
                self.assertEqual(len(proof["proofs"][0]["archive_sha256"]), 3)
                if algorithm == "pipeannfilter":
                    self.assertNotIn("loading_cgroup_peak_bytes", row)
                    self.assertNotIn("bits", fixture.measured)

    def test_preflight_checks_dependencies_and_closes_inputs(self):
        fixture = self.make_fixture()
        ready = preflight_curve(fixture.spec, fixture.curve)
        self.assertEqual((ready["status"], ready["dataset_vectors"], ready["dimension"]), ("ready", 4, 2))
        # A closed mapping permits immediate replacement on Windows as well.
        gt = fixture.root / "gt.ivecs"
        gt.write_bytes(gt.read_bytes()[:-1])
        with self.assertRaisesRegex(ValidationError, "GT length"):
            preflight_curve(fixture.spec, fixture.curve)
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaises(ImportError):
                preflight_curve(fixture.spec, fixture.curve)

    def test_paf_short_is_counted_not_silently_zero(self):
        fixture = self.make_fixture("pipeannfilter", short=True)
        row = fixture.verify()["rows"][0]
        self.assertEqual((row["short_queries"], row["returned_results_replayed"], row["recall"]), (1, 3, .75))
        fixture.measured["short_queries"] = 0
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "short query"):
            fixture.verify()

    def test_id_duplicate_predicate_and_exact_distance(self):
        cases = [("duplicate", [(0, 0.0), (0, 0.0)]),
                 ("out of range", [(0, 0.0), (9, 1.0)]),
                 ("exact L2", [(0, 1.0), (1, 1.0)])]
        for message, values in cases:
            with self.subTest(message=message):
                fixture = self.make_fixture()
                fixture.results[0] = values
                fixture.save()
                with self.assertRaisesRegex(ValidationError, message):
                    fixture.verify()
        fixture = self.make_fixture()
        # GT remains in range while returned node 3 violates the narrower range.
        (fixture.root / "ranges.bin").write_bytes(struct.pack("<4i", 0, 1, 0, 3))
        fixture.results[0] = [(0, 0.0), (3, 9.0)]
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "result violates predicate"):
            fixture.verify()

    def test_nonfinite_distance_and_gt_integrity(self):
        fixture = self.make_fixture()
        fixture.results[0][0] = (0, float("nan"))
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "nonfinite"):
            fixture.verify()
        fixture = self.make_fixture()
        (fixture.root / "gt.ivecs").write_bytes(struct.pack("<6I", 2, 0, 0, 2, 2, 1))
        with self.assertRaisesRegex(ValidationError, "duplicate GT"):
            fixture.verify()
        fixture = self.make_fixture()
        (fixture.root / "gt.ivecs").write_bytes(struct.pack("<6I", 3, 0, 1, 2, 2, 1))
        with self.assertRaisesRegex(ValidationError, "GT row header"):
            fixture.verify()

    def test_exact_l2_keeps_original_one_e_minus_four_tolerance(self):
        fixture = self.make_fixture()
        fixture.results[0][0] = (0, 1e-4)
        fixture.save()
        self.assertEqual(fixture.verify()["status"], "verified")
        fixture.results[0][0] = (0, 1.0001e-4)
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "exact L2 mismatch"):
            fixture.verify()

    def test_metrics_are_independently_recomputed(self):
        for key in ("recall", "qps", "p95_ms", "io_requests", "io_bytes", "matches"):
            fixture = self.make_fixture()
            fixture.measured[key] += 1
            fixture.save()
            with self.subTest(key=key), self.assertRaises(ValidationError):
                fixture.verify()
        fixture = self.make_fixture()
        fixture.observations[1]["query_id"] = 0
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "nonsequential"):
            fixture.verify()

    def test_native_io_cache_and_observer_contracts(self):
        cases = [("allianceann", "observer_graph_requests", 0),
                 ("allianceann", "dynamic_cache_clear_per_query", True),
                 ("allianceann", "pending", 1),
                 ("pipewow", "native_io", 0),
                 ("pipewow", "dynamic_hits", 0),
                 ("pipewow", "static_cache_bytes", 1),
                 ("pipeannfilter", "observer_verified", False),
                 ("pipeannfilter", "completed_bytes", 1),
                 ("pipeannfilter", "graph_bytes", 1)]
        for algorithm, key, value in cases:
            fixture = self.make_fixture(algorithm)
            fixture.measured[key] = value
            fixture.save()
            with self.subTest(algorithm=algorithm, key=key), self.assertRaises(ValidationError):
                fixture.verify()

    def test_memory_uses_phase_peaks_and_rejects_missing_evidence(self):
        for key, value in (("loading_peak_rss_bytes", 9001), ("search_phase_peak_rss_bytes", 3001),
                           ("search_rss_limit_bytes", 0), ("rss_budget_passed", False)):
            fixture = self.make_fixture()
            fixture.measured[key] = value
            fixture.save()
            with self.subTest(key=key), self.assertRaises(ValidationError):
                fixture.verify()
        fixture = self.make_fixture()
        del fixture.measured["search_phase_peak_rss_bytes"]
        fixture.measured["peak_rss_bytes"] = 2000
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "missing required field"):
            fixture.verify()

    def test_expected_fields_and_zero_prefilter_are_enforced(self):
        fixture = self.make_fixture()
        fixture.curve["expected"] = {"measured": {"edge_block_bytes": 8192}}
        with self.assertRaisesRegex(ValidationError, "expected.measured"):
            fixture.verify()
        fixture = self.make_fixture()
        fixture.curve["expected"]["require_zero_prefilter_queries"] = True
        fixture.measured.update(prefilter_queries=1, graph_queries=1)
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "unexpected prefilter"):
            fixture.verify()
        fixture = self.make_fixture()
        fixture.curve["expected"]["unknown"] = 0
        with self.assertRaisesRegex(ValidationError, "unsupported validation expectation"):
            fixture.verify()
        fixture = self.make_fixture()
        fixture.curve["expected"]["maximum_measured"] = {"graph_io": 1}
        with self.assertRaisesRegex(ValidationError, "maximum_measured exceeded"):
            fixture.verify()

    def test_fingerprint_is_bound_to_native_success_marker(self):
        fixture = self.make_fixture()
        fixture.curve["quantization_fingerprint"] = "12345"
        (fixture.root / "stdout.log").write_text("SEARCH_VERIFIED fingerprint=12345\n", encoding="utf-8")
        self.assertEqual(fixture.verify()["verified_quantization_fingerprint"], "12345")
        for text in ("", "SEARCH_VERIFIED fingerprint=12346\n", "SEARCH_VERIFIED fingerprint=12345\nfailed\n"):
            (fixture.root / "stdout.log").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "fingerprint"):
                fixture.verify()
        fixture = self.make_fixture()
        del fixture.curve["quantization_fingerprint"]
        with self.assertRaisesRegex(ValidationError, "missing expected quantization fingerprint"):
            fixture.verify()
        fixture = self.make_fixture()
        fixture.measured["quantization_fingerprint"] = "999"
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "measured quantization fingerprint"):
            fixture.verify()

    def test_paf_attribute_eof_preserves_requested_byte_accounting(self):
        fixture = self.make_fixture("pipeannfilter")
        fixture.observations[1]["bytes"] = 4096
        fixture.measured.update(io_bytes=12288, completed_bytes=8192+24, graph_reads=1, graph_bytes=8192,
                                attribute_reads=1, attribute_bytes=4096, attribute_eof_reads=1,
                                io_4k_equivalent_per_query=1.5)
        fixture.save()
        self.assertEqual(fixture.verify()["rows"][0]["io_bytes_per_query"], 6144)
        fixture.measured["completed_bytes"] += 1
        fixture.save()
        with self.assertRaisesRegex(ValidationError, "EOF byte deficit"):
            fixture.verify()

    def test_truncation_trailing_payload_and_missing_repeat_fail_closed(self):
        for algorithm in ("allianceann", "pipeannfilter"):
            fixture = self.make_fixture(algorithm)
            path = fixture.directory / "results.bin"
            good = path.read_bytes()
            for bad in (good[:-1], good + b"x"):
                path.write_bytes(bad)
                with self.subTest(algorithm=algorithm, size=len(bad)), self.assertRaises(ValidationError):
                    fixture.verify()
        fixture = self.make_fixture()
        fixture.curve["repeats"] = fixture.spec["aggregation"]["repeats"] = 2
        with self.assertRaisesRegex(ValidationError, "batch coverage"):
            fixture.verify()

    def test_native_repeat_identity_cannot_be_inferred_from_directory(self):
        for algorithm in ("allianceann", "pipewow", "pipeannfilter"):
            fixture = self.make_fixture(algorithm)
            fixture.measured["repeat"] = 2
            fixture.save()
            with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValidationError, "mismatch: repeat"):
                fixture.verify()
        for algorithm in ("allianceann", "pipewow"):
            fixture = self.make_fixture(algorithm)
            fixture.measured["requested_repeats"] = 2
            fixture.save()
            with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValidationError, "requested_repeats"):
                fixture.verify()
            del fixture.measured["requested_repeats"]
            fixture.save()
            with self.assertRaisesRegex(ValidationError, "missing required field: requested_repeats"):
                fixture.verify()
        fixture = self.make_fixture("pipeannfilter")
        del fixture.measured["repeat"]
        fixture.save()
        self.assertEqual(fixture.verify()["status"], "verified")  # Historical one-run PAF format.
        fixture.curve["repeats"] = fixture.spec["aggregation"]["repeats"] = 2
        shutil.copytree(str(fixture.root / "raw" / "r1"), str(fixture.root / "raw" / "r2"))
        with self.assertRaisesRegex(ValidationError, "missing required field: repeat"):
            fixture.verify()

    def test_extra_batches_and_copied_repeats_are_rejected(self):
        fixture = self.make_fixture()
        extra = fixture.root / "raw" / "r1" / "ef32"
        extra.mkdir()
        (extra / "measured.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "batch coverage"):
            fixture.verify()
        fixture = self.make_fixture()
        fixture.curve["repeats"] = fixture.spec["aggregation"]["repeats"] = 2
        fixture.measured["requested_repeats"] = 2
        fixture.save()
        shutil.copytree(str(fixture.root / "raw" / "r1"), str(fixture.root / "raw" / "r2"))
        with self.assertRaisesRegex(ValidationError, "mismatch: repeat"):
            fixture.verify()

    def test_shape_validation_precedes_results(self):
        fixture = self.make_fixture()
        (fixture.root / "base.bin").write_bytes(struct.pack("<II", 4, 3))
        with self.assertRaisesRegex(ValidationError, "base matrix shape"):
            fixture.verify()
        fixture = self.make_fixture()
        (fixture.root / "query.fvecs").write_bytes(struct.pack("<IffIff", 3, 0, 0, 2, 2, 0))
        with self.assertRaisesRegex(ValidationError, "fvecs row header"):
            fixture.verify()


if __name__ == "__main__":
    unittest.main()
