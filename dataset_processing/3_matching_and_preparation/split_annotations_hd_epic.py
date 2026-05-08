# This script reads a JSONL file containing video annotations, extracts unique video IDs and their associated participants, and creates a train/validation split (80/20) for each participant. The resulting video IDs are saved to separate text files.

import json
import os

def split_annotations(jsonl_path, split_dir, output_dir="."):
    # Read video split files
    with open(os.path.join(split_dir, 'train_videos.txt'), 'r') as f:
        train_videos = set(f.read().splitlines())
    
    with open(os.path.join(split_dir, 'val_videos.txt'), 'r') as f:
        val_videos = set(f.read().splitlines())
    
    # Process annotations
    train_annos = []
    val_annos = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                anno = json.loads(line)
                vid = anno['video_id']
                
                if vid in train_videos:
                    train_annos.append(anno)
                elif vid in val_videos:
                    val_annos.append(anno)
    
    # Prepare output filenames based on input
    base = os.path.basename(jsonl_path)
    if base.endswith('.jsonl'):
        base = base[:-6]
    train_out = os.path.join(output_dir, f"{base}_train.jsonl")
    val_out = os.path.join(output_dir, f"{base}_val.jsonl")

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    with open(train_out, 'w') as f:
        for anno in train_annos:
            f.write(json.dumps(anno) + '\n')
    with open(val_out, 'w') as f:
        for anno in val_annos:
            f.write(json.dumps(anno) + '\n')

    
    print(f"Created training set with {len(train_annos)} annotations")
    print(f"Created validation set with {len(val_annos)} annotations")
    print(f"Output files saved in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="Path to input JSONL annotations file")
    parser.add_argument("split_dir", help="Directory containing train_videos.txt and val_videos.txt")
    parser.add_argument("--output_dir", default=".", help="Output directory for JSONL files")
    args = parser.parse_args()
    
    split_annotations(args.jsonl_file, args.split_dir, args.output_dir)


# Usage:
# python split_annotations.py  ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl ${DATA_ROOT}/HD_EPIC/underspecified/splits_80_20_participants --output_dir ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1
# python split_annotations.py  ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl ${DATA_ROOT}/HD_EPIC/underspecified/splits_80_20_participants --output_dir ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2
# python split_annotations.py  ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl ${DATA_ROOT}/HD_EPIC/underspecified/splits_80_20_participants --output_dir ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3
# python split_annotations.py  ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl ${DATA_ROOT}/HD_EPIC/underspecified/splits_80_20_participants --output_dir ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4



# python split_annotations.py  ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_window_500.jsonl ${DATA_ROOT}/HD_EPIC/underspecified/splits_80_20_participants --output_dir ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4