import json
import numpy as np


def compute_relevant_window_stats(jsonl_path):
    per_query_ranges = []
    multi_moment_counts = []
    max_num_moments = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines
            data = json.loads(line)
            relevant_windows = data.get('relevant_windows', [])
            if not relevant_windows:
                continue

            # Update max number of moments seen so far
            num_moments = len(relevant_windows)
            if num_moments > max_num_moments:
                max_num_moments = num_moments

            # Track counts only for multi-moment cases (>1)
            if num_moments > 1:
                multi_moment_counts.append(num_moments)

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

    # Compute percentiles of number of relevant windows in multi-moment cases
    if multi_moment_counts:
        multi_moment_arr = np.array(multi_moment_counts)
        mm_p25 = np.percentile(multi_moment_arr, 25)
        mm_p50 = np.percentile(multi_moment_arr, 50)
        mm_p75 = np.percentile(multi_moment_arr, 75)
        mm_p90 = np.percentile(multi_moment_arr, 90)
    else:
        mm_p25 = mm_p50 = mm_p75 = mm_p90 = None  # no multi-moment cases found

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
        },
        'max_num_moments': max_num_moments,
        'multi_moment_counts_percentiles': {
            25: mm_p25,
            50: mm_p50,
            75: mm_p75,
            90: mm_p90
        }
    }


if __name__ == "__main__":
    #path = "${DATA_ROOT}/HD_EPIC/underspecified/original_annos/HD_EPIC_Narrations_train.jsonl"
    #path = "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl"
    #path = "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl"
    #path = "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl"
    path = "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl"
                
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
        print()
        print(f"Maximum number of moments in any instance: {stats['max_num_moments']}")
        if stats['multi_moment_counts_percentiles'][25] is not None:
            print("Percentiles of number of relevant windows (multi-moment cases only):")
            print(f"  25th percentile: {stats['multi_moment_counts_percentiles'][25]:.2f}")
            print(f"  50th percentile: {stats['multi_moment_counts_percentiles'][50]:.2f}")
            print(f"  75th percentile: {stats['multi_moment_counts_percentiles'][75]:.2f}")
            print(f"  90th percentile: {stats['multi_moment_counts_percentiles'][90]:.2f}")
        else:
            print("No multi-moment cases found, no percentiles computed.")


