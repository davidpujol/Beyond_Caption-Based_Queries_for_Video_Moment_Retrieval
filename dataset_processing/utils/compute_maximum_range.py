import json
import math
import numpy as np

def compute_relevant_window_stats(jsonl_path):
    per_query_ranges = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines
            data = json.loads(line)
            relevant_windows = data.get('relevant_windows', [])
            if not relevant_windows:
                continue
            
            # Find min and max for this query's windows
            query_starts = [seg[0] for seg in relevant_windows]
            query_ends = [seg[1] for seg in relevant_windows]
            query_min = min(query_starts)
            query_max = max(query_ends)

            per_query_ranges.append(query_max - query_min)

    if not per_query_ranges:
        print("No relevant windows found in the file.")
        return None

    arr = np.array(per_query_ranges)

    min_range = arr.min()
    max_range = arr.max()
    avg_range = arr.mean()
    std_dev = arr.std()
    median = np.median(arr)
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)  # same as median
    p75 = np.percentile(arr, 75)
    p90 = np.percentile(arr, 90)

    return {
        'min_range': min_range,
        'max_range': max_range,
        'avg_range': avg_range,
        'std_dev': std_dev,
        'median': median,
        'percentiles': {
            25: p25,
            50: p50,
            75: p75,
            90: p90
        }
    }

if __name__ == "__main__":
    path = "${HOME}/Deambiguation_of_under_specified_queries/dataset_processing/datasets_augmentation_generation/matching_underspecified_queries/temp_merged.jsonl"  # Replace with your JSONL file path
    stats = compute_relevant_window_stats(path)
    if stats:
        print("Overall relevant window stats:")
        print(f"  Min range: {stats['min_range']}")
        print(f"  Max range: {stats['max_range']}")
        print(f"  Average maximum range per query: {stats['avg_range']:.4f}")
        print(f"  Standard deviation of ranges: {stats['std_dev']:.4f}")
        print(f"  Median range: {stats['median']:.4f}")
        print(f"  25th percentile: {stats['percentiles'][25]:.4f}")
        print(f"  50th percentile: {stats['percentiles'][50]:.4f}")
        print(f"  75th percentile: {stats['percentiles'][75]:.4f}")
        print(f"  90th percentile: {stats['percentiles'][90]:.4f}")
