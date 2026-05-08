# This file takes as input the unmatched underspecified queries (so we have first done the full matching and then kept the matching information but decoupled into individual queries that still share pointers that indicate which of them are crossed matched).
# In this script we are trying to generate merged instances that accumulate all the relevant windows of a given underspecified query that fall within the same window.
# We do this for evaluation purposes since we might want to evaluate together windows to be able to control the MATCHES of different GT, even if we evaluate them individually.

import json
import argparse
from collections import defaultdict
import os

def get_time_bin(time, window_size, fps):
    """Return the time bin index for a given timestamp."""
    seconds_per_bin = window_size / fps
    return int(time // seconds_per_bin)

def merge_jsonl(input_file, window_size=500, fps=3):
    merged_entries = defaultdict(lambda: defaultdict(list))

    # Read input JSONL
    with open(input_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            uid = entry["unique_narration_id"]

            # Use only the start time to decide the bin
            start_time = entry["relevant_windows"][0]
            bin_id = get_time_bin(start_time, window_size, fps)
            merged_entries[(uid, bin_id)]["entries"].append(entry)

    
    final_entries = []
    for (uid, b), data in merged_entries.items():
        entries = data["entries"]

        # Start from the first entry as reference
        base = dict(entries[0])
        # Merge all relevant windows
        all_windows = []
        for e in entries:
            all_windows.append(e["relevant_windows"])
        base["relevant_windows"] = sorted(all_windows)

        # Pick one original_narration_id (first one)
        base["original_narration_id"] = entries[0]["original_narration_id"]

        final_entries.append(base)

    # Save output
    output_file = os.path.splitext(input_file)[0] + f"_merged_window_{window_size}.jsonl"
    with open(output_file, "w") as f:
        for entry in final_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Merged file saved to {output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("--window_size", type=int, default=500, help="Window size in frames (default: 500)")
    parser.add_argument("--fps", type=int, default=3, help="Frames per second (default: 3)")
    args = parser.parse_args()

    merge_jsonl(args.input_file, args.window_size, args.fps)

# Usage:
# python matching_window_based.py ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm_val.jsonl --window_size 500
# python matching_window_based.py ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm_val.jsonl --window_size 500
# python matching_window_based.py ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3/HD_EPIC_Narrations_underspecified_full_queries_llm_val.jsonl --window_size 500
# python matching_window_based.py ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl --window_size 500