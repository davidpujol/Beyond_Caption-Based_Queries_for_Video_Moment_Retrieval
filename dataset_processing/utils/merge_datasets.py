import os
import shutil
import random

def mix_npz_files(dir1, dir2, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all .npz files in both directories
    files_dir1 = [f for f in os.listdir(dir1) if f.endswith('.npz')]
    files_dir2 = [f for f in os.listdir(dir2) if f.endswith('.npz')]

    # Ensure both directories have the same files
    common_files = list(set(files_dir1) & set(files_dir2))
    common_files.sort()  # For reproducibility

    # Shuffle and split the list into two halves
    random.shuffle(common_files)
    half = len(common_files) // 2
    first_half = common_files[:half]
    second_half = common_files[half:]

    # Copy files: first half from dir1, second half from dir2
    for f in first_half:
        shutil.copy(os.path.join(dir1, f), os.path.join(output_dir, f))
    for f in second_half:
        shutil.copy(os.path.join(dir2, f), os.path.join(output_dir, f))

    print(f"Copied {len(first_half)} files from {dir1} and {len(second_half)} files from {dir2} to {output_dir}.")

# Example usage:
mix_npz_files('${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_original', '${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_no_subj_llm', '${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_merged_original_and_subj')
mix_npz_files('${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_original', '${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_no_attr_modif_llm', '${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_merged_original_and_attr')
mix_npz_files('${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_original', '${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_simp_llm', '${DATA_ROOT}/QVHighlights/internvideo_original/txt_feats_merged_original_and_full')
