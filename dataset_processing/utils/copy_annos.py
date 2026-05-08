# This simple script takes a given .jsonl file, and a another .jsonl that contains some of these annotations with additional GT segments based on our manual reannotation
# All it does is copy the annotations from the second file to the first one, based on the query field, and keep the rest of the annotations intact

import json
from pathlib import Path
import argparse

def load_jsonl_to_dict(file_path, key_field):
    """
    Loads a .jsonl file into a dict mapping key_field to annotation.
    """
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            ann = json.loads(line)
            data[ann[key_field]] = ann
    return data

def merge_annotations(original_path, reannotated_path, output_path, key_field="qid"):
    # Load reannotated as a dict for fast lookup
    reannotated_dict = load_jsonl_to_dict(reannotated_path, key_field)

    with open(original_path, "r") as orig_f, open(output_path, "w") as out_f:
        for line in orig_f:
            orig_ann = json.loads(line)
            key = orig_ann[key_field]
            # Use reannotated if available, else original
            merged_ann = reannotated_dict.get(key, orig_ann)
            out_f.write(json.dumps(merged_ann) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge reannotated GT segments into original dataset.")
    parser.add_argument("original_jsonl", type=str, help="Path to the original .jsonl file")
    parser.add_argument("reannotated_jsonl", type=str, help="Path to the reannotated .jsonl file")
    parser.add_argument("output_jsonl", type=str, help="Path to the merged output .jsonl file")
    parser.add_argument("--key_field", type=str, default="qid", help="Field to use as the unique key for matching")
    args = parser.parse_args()

    merge_annotations(args.original_jsonl, args.reannotated_jsonl, args.output_jsonl, key_field=args.key_field)
    print(f"Merged file written to {args.output_jsonl}")


# Usage:
# python copy_annos.py ${DATA_ROOT}/QVHighlights/underspecified/llm_based/qvhighlights_val_underspecified_llm.jsonl ${DATA_ROOT}/QVHighlights/underspecified/llm_based/manual_reannotation/qvhighlights_val_underspecified_full_simplification_llm_reannotated.jsonl ${DATA_ROOT}/QVHighlights/underspecified/llm_based/manual_reannotation/qvhighlights_val_underspecified_full_simplification_llm_reannotated_final.jsonl --key_field qid