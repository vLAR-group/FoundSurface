#!/usr/bin/env python
"""Aggregate per-scene `*_results_net.json` files written by the test script.

Example
-------
    python tools/summarize_results.py \
        outputs/scannetpp/distance/Scannetpp_Raylet_fit_top1_view50/test --split test

Each scene directory contains `<split>_results_net.json`, mapping a method name
to its metrics. Values are averaged over scenes (one value per scene), which is
what the paper tables report. Normal metrics are included automatically when
present in the JSON.
"""
import argparse
import glob
import json
import os
import sys

DISTANCE_METRICS = ["ADE", "RMSE", "REL_ADE", "SQ_REL_ADE", "A1", "A2", "A3"]
NORMAL_METRICS = ["NORMAL_MAE", "NORMAL_MED_ERR", "NORMAL_PTC5", "NORMAL_PTC10", "NORMAL_PTC30"]


def load_scene(path, method):
    with open(path) as f:
        data = json.load(f)
    if method is not None:
        if method not in data:
            raise KeyError(f"{path}: no method '{method}' (have {list(data)})")
        return data[method]
    for value in data.values():
        if isinstance(value, dict) and ("ADE" in value or "NORMAL_MAE" in value):
            return value
    raise KeyError(f"{path}: no metrics found")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory containing one sub-directory per scene")
    ap.add_argument("--split", default="test", help="results filename prefix (default: test)")
    ap.add_argument("--method", default=None, help="method key inside the JSON (default: first)")
    ap.add_argument("--csv", default=None, help="optional path for a CSV dump")
    args = ap.parse_args()

    pattern = os.path.join(args.root, "*", f"{args.split}_results_net.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files matched {pattern}")

    rows = [(os.path.basename(os.path.dirname(p)), load_scene(p, args.method)) for p in files]
    metrics = [m for m in DISTANCE_METRICS if any(m in v for _, v in rows)]
    metrics += [m for m in NORMAL_METRICS if any(m in v for _, v in rows)]

    print(f"{len(rows)} scenes under {args.root}")
    print(f"{'scene':<28}" + "".join(f"{m:>16}" for m in metrics))

    sums = {m: 0.0 for m in metrics}
    present = {m: 0 for m in metrics}
    for scene, values in rows:
        line = f"{scene:<28}"
        for m in metrics:
            if m in values:
                sums[m] += float(values[m])
                present[m] += 1
                line += f"{float(values[m]):>16.5f}"
            else:
                line += f"{'-':>14}"
        print(line)

    print("-" * (28 + 16 * len(metrics)))
    mean_line = f"{'mean':<28}"
    for m in metrics:
        mean_line += f"{sums[m] / present[m]:>16.5f}" if present[m] else f"{'-':>14}"
    print(mean_line)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("scene," + ",".join(metrics) + "\n")
            for scene, values in rows:
                f.write(scene + "," + ",".join(str(values.get(m, "")) for m in metrics) + "\n")
            f.write("mean," + ",".join(
                f"{sums[m] / present[m]:.8f}" if present[m] else "" for m in metrics) + "\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
