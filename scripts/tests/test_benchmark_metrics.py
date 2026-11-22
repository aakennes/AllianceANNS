"""Synthetic protocol tests; no native process or dataset is used."""

import copy
from pathlib import Path
import statistics
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from benchlib.metrics import aggregate_metric, aggregate_repeats, METRIC_WISE_TRIM, QPS_PAIRED_TRIM


def rows():
    return [dict(repeat=i, validated=True, qps=float(i), io=value, recall=.9)
            for i, value in enumerate([30, 1000, 20, 10, 1], 1)]


class AggregationProtocolTests(unittest.TestCase):
    def test_paired_trim_retains_same_rows_when_io_and_qps_disagree(self):
        original = rows()
        before = copy.deepcopy(original)
        paired = aggregate_repeats(original, ["qps", "io", "recall"], QPS_PAIRED_TRIM)
        independent = aggregate_repeats(original, ["qps", "io", "recall"], METRIC_WISE_TRIM)
        self.assertEqual(paired["kept_repeat_ids"], [2, 3, 4])
        self.assertEqual(paired["dropped_repeat_ids"], [1, 5])
        self.assertEqual(paired["metrics"]["io"], statistics.mean([1000, 20, 10]))
        self.assertEqual(independent["metrics"]["io"], 20)
        self.assertEqual(paired["metrics"]["qps"], independent["metrics"]["qps"])
        self.assertIsNone(independent["kept_repeat_ids"])
        self.assertEqual(original, before)

    def test_reverse_order_and_qps_ties_are_deterministic(self):
        data = rows()
        for row in data:
            row["io"] = 6 - row["repeat"]
        self.assertEqual(aggregate_repeats(data[::-1], ["qps", "io"], QPS_PAIRED_TRIM)["metrics"], {"qps": 3, "io": 3})
        for row in data:
            row["qps"] = 7
        result = aggregate_repeats(data[::-1], ["io"], QPS_PAIRED_TRIM)
        self.assertEqual(result["kept_repeat_ids"], [2, 3, 4])

    def test_invalid_repeat_cannot_be_hidden_by_trimming(self):
        for mutate in (lambda r: r[0].update(validated=False), lambda r: r[0].update(io=float("nan")),
                       lambda r: r[0].update(repeat=2), lambda r: r[0].update(qps=0)):
            data = rows()
            mutate(data)
            with self.assertRaises(ValueError):
                aggregate_repeats(data, ["qps", "io"], QPS_PAIRED_TRIM)
        with self.assertRaises(ValueError):
            aggregate_repeats(rows()[:3], ["io"], QPS_PAIRED_TRIM, expected_repeats=3)

    def test_legacy_metric_arithmetic_is_unchanged(self):
        values = [1, 2, 4, 9, 1000]
        self.assertEqual(aggregate_metric(values, "trimmed_mean"), statistics.mean(sorted(values)[1:-1]))
        self.assertEqual(aggregate_metric(values, "median"), statistics.median(values))
        self.assertEqual(aggregate_metric(values, "mean"), statistics.mean(values))
        from benchlib import legacy_log as legacy
        data = [dict(workload="p1", threads=1, repeat=row["repeat"], efs=10, qps=row["qps"],
                     recall=.9, graph_io=row["io"], rerank_io=1, total_io=row["io"]+1) for row in rows()]
        for policy in ("median", "mean", "trimmed_mean"):
            cfg = dict(aggregate=policy, workloads={"p1": {}}, threads=[1], efs=[10], repeats=5,
                       parameters={"ml_meta_file": "unused"})
            result = legacy.summarize(data, cfg)[0]
            for metric in ("qps", "recall", "graph_io", "rerank_io", "total_io"):
                self.assertEqual(result[metric], aggregate_metric([row[metric] for row in data], policy))


if __name__ == "__main__":
    unittest.main()
