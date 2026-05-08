# This is a simple script to count the number of instances in a JSONL file
# that have multiple annotated moments in the 'relevant_windows' field.
# It calculates the ratio of such instances to the total number of instances.
import json

def count_multiple_moments(jsonl_path):
    """
    Counts total instances and those with multiple annotated moments 
    in the 'relevant_windows' field in a given JSONL file.
    Also computes:
      - average number of GT moments per instance
      - average number of GT moments for multi-moment instances only
    """
    total = 0
    multiple_moments = 0
    total_moments = 0
    total_moments_multiple = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue  # skip empty lines
                data = json.loads(line)
                moments_count = len(data['relevant_windows']) if 'relevant_windows' in data else 1
                total += 1
                total_moments += moments_count
                if moments_count > 1:
                    multiple_moments += 1
                    total_moments_multiple += moments_count
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return None
    
    if total == 0:
        return None
    
    avg_moments_all = total_moments / total
    avg_moments_multiple = (total_moments_multiple / multiple_moments) if multiple_moments > 0 else 0
    
    return {
        'total': total,
        'multiple_moments': multiple_moments,
        'ratio': (multiple_moments / total)*100 ,
        'avg_moments_all': avg_moments_all,
        'avg_moments_multiple': avg_moments_multiple
    }


if __name__ == "__main__":
    datasets = {
        "QVHighlights_train_original": "${DATA_ROOT}/QVHighlights/underspecified/qvhighlights_train.jsonl",
        "QVHighlights_val_original": "${DATA_ROOT}/QVHighlights/underspecified/qvhighlights_val.jsonl",
        "QVHighlights_train_underspecified": "${DATA_ROOT}/QVHighlights/underspecified/llm_based/manual_reannotation/qvhighlights_train_underspecified_llm_agentic_reannotation.jsonl",
        "QVHighlights_val_underspecified": "${DATA_ROOT}/QVHighlights/underspecified/llm_based/manual_reannotation/qvhighlights_val_underspecified_full_simplification_llm_reannotated_final.jsonl",
        "HD_EPIC_train_original": "${DATA_ROOT}/HD_EPIC/underspecified/original_annos/HD_EPIC_Narrations_train.jsonl",
        "HD_EPIC_val_original": "${DATA_ROOT}/HD_EPIC/underspecified/original_annos/HD_EPIC_Narrations_val.jsonl",
        "HD_EPIC_underspecified_v1": "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl",
        "HD_EPIC_underspecified_v2": "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl",
        "HD_EPIC_underspecified_v3": "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl",
        "HD_EPIC_underspecified_v4": "${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl",
        "SynopGround_train_original": "${DATA_ROOT}/SynopGround/annotations/train.jsonl",
        "SynopGround_val_original": "${DATA_ROOT}/SynopGround/annotations/val.jsonl",
        "SynopGround_train_underspecified": "${DATA_ROOT}/SynopGround/annotations/underspecified/atttribute_modifiers_simplification/train_underspecified_attributes_modifiers_only_queries_llm.jsonl",
        "SynopGround_val_underspecified": "${DATA_ROOT}/SynopGround/annotations/underspecified/atttribute_modifiers_simplification/val_underspecified_attributes_modifiers_only_queries_llm.jsonl",
    }
    
    print("="*85)
    print("Multiple Annotated Moments Statistics per Dataset")
    print("="*85)
    
    results = []
    for name, path in datasets.items():
        stats = count_multiple_moments(path)
        if stats is None:
            print(f"Skipping dataset {name} due to errors or no data.")
            print("-"*85)
            continue
        
        results.append((name, stats))
        print(f"Dataset: {name}")
        print(f"File: {path}")
        print(f"  Total instances: {stats['total']}")
        print(f"  Instances with multiple annotated moments: {stats['multiple_moments']}")
        print(f"  Ratio: {stats['ratio']:.4f}")
        print(f"  Average GT moments (all queries): {stats['avg_moments_all']:.4f}")
        print(f"  Average GT moments (multi-moment queries only): {stats['avg_moments_multiple']:.4f}")
        print("-"*85)


    # Optionally, generate the latex table
    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Statistics of Multiple Annotated Moments and Average GT Moments per Dataset}")
    print(r"\label{tab:multi_moments_stats}")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"Dataset & Total & Multiple & Ratio (\%) & Avg. Moments (all) & Avg. Moments (multi) \\")
    print(r"\midrule")
    for name, s in results:
        # Escape underscores for LaTeX:
        latex_name = name.replace('_', r'\_')
        print(f"{latex_name} & {s['total']:,} & {s['multiple_moments']:,} & {s['ratio']:.2f} & {s['avg_moments_all']:.4f} & {s['avg_moments_multiple']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
