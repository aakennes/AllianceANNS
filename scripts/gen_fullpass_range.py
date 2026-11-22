#!/usr/bin/env python3
"""Generate a full-pass (no-filter) range file for cache hit rate control experiments.

Usage:
  python3 scripts/gen_fullpass_range.py \
      --src /path/to/test.ranges.ibin \
      --out /path/to/test.ranges.fullpass.ibin \
      --att_file /path/to/base.attribute.ibin

Reads the source range file to get nq, reads the attribute file to get
[min_att, max_att], writes nq pairs of (min_att, max_att).
"""
import argparse
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source range file (to get nq)")
    parser.add_argument("--out", required=True, help="Output full-pass range file")
    parser.add_argument("--att_file", required=True,
                        help="Base attribute file (.ibin, int32 per node) to derive [min, max]")
    args = parser.parse_args()

    with open(args.src, "rb") as f:
        raw = f.read()
    nq = len(raw) // 8
    pairs = struct.unpack(f"<{2 * nq}i", raw)

    with open(args.att_file, "rb") as f:
        att_raw = f.read()
    n_att = len(att_raw) // 4
    atts = struct.unpack(f"<{n_att}i", att_raw)
    lo, hi = min(atts), max(atts)

    with open(args.out, "wb") as f:
        for _ in range(nq):
            f.write(struct.pack("ii", lo, hi))

    print(f"Written {nq} full-pass ranges [{lo}, {hi}] -> {args.out}")
    print(f"Attribute stats: n={n_att} min={lo} max={hi}")
    widths = [pairs[2*i+1] - pairs[2*i] for i in range(nq)]
    print(f"Original range stats: min_width={min(widths)} max_width={max(widths)} "
          f"mean_width={sum(widths)/len(widths):.0f}")


if __name__ == "__main__":
    main()
