"""Named aggregation policies for already validated benchmark repetitions."""

import math
import statistics


METRIC_WISE_TRIM = "metric-wise-trim"
QPS_PAIRED_TRIM = "qps-paired-trim"


def aggregate_metric(values, policy):
    """Preserve the legacy per-metric arithmetic, including its trimmed_mean alias."""
    values = list(values)
    if not values:
        raise ValueError("empty metric group")
    if policy in (METRIC_WISE_TRIM, "trimmed_mean"):
        if len(values) < 3:
            raise ValueError("metric-wise trim requires at least three repeats")
        return statistics.mean(sorted(values)[1:-1])
    if policy == "median":
        return statistics.median(values)
    if policy == "mean":
        return statistics.mean(values)
    raise ValueError("unknown metric policy: " + str(policy))


def aggregate_repeats(rows, metrics, policy, expected_repeats=5):
    """Aggregate one EF/algorithm/cache group; all repeats must pass before trimming.

    QPS ties are ordered by repeat ID. Paired trim uses the same retained rows
    for every metric, including IO and recall. Input rows are never mutated.
    P95 is a mean of batch P95 values, not a pooled-query percentile.
    """
    rows, metrics = list(rows), tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be nonempty and unique")
    if type(expected_repeats) is not int or expected_repeats < 1:
        raise ValueError("expected repeat count must be a positive integer")
    if (len(rows) != expected_repeats or any(type(r.get("repeat")) is not int for r in rows)
            or {r.get("repeat") for r in rows} != set(range(1, expected_repeats + 1))):
        raise ValueError("incomplete or duplicate repeat group")
    if any(r.get("validated") is not True for r in rows):
        raise ValueError("every repeat must pass validation before aggregation")
    for row in rows:
        for metric in set(metrics) | ({"qps"} if policy == QPS_PAIRED_TRIM else set()):
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid metric: " + metric)
        if "qps" in row and row["qps"] <= 0:
            raise ValueError("qps must be positive")
    kept_ids = dropped_ids = None
    if policy == QPS_PAIRED_TRIM:
        if expected_repeats != 5:
            raise ValueError("qps-paired-trim requires exactly five valid repeats")
        ranked = sorted(rows, key=lambda row: (row["qps"], row["repeat"]))
        kept = ranked[1:-1]
        values = {m: statistics.mean(row[m] for row in kept) for m in metrics}
        kept_ids = [row["repeat"] for row in kept]
        dropped_ids = [ranked[0]["repeat"], ranked[-1]["repeat"]]
        retained = 3
    else:
        values = {m: aggregate_metric((row[m] for row in rows), policy) for m in metrics}
        retained = len(rows) - 2 if policy in (METRIC_WISE_TRIM, "trimmed_mean") else len(rows)
    return {"policy": policy, "measured_repeats": len(rows), "retained_repeats": retained,
            "kept_repeat_ids": kept_ids, "dropped_repeat_ids": dropped_ids, "metrics": values}
