#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def load_ids(path):
    """Load a list of video ids from a json file"""
    with open(path, "r") as f:
        return set(json.load(f))


def split_annotations(input_jsonl, output_dir, split_files):
    input_path = Path(input_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load splits
    split_to_vids = {name: load_ids(fpath) for name, fpath in split_files.items()}

    # prepare file handles
    writers = {name: open(output_dir / f"{name}.jsonl", "w") for name in split_files}

    counts = {name: 0 for name in split_files}

    with input_path.open("r") as fin:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            vid = ex["video_id"]

            # assign to the correct split
            for name, vids in split_to_vids.items():
                if vid in vids:
                    writers[name].write(line)
                    counts[name] += 1
                    break

    for w in writers.values():
        w.close()

    print("✅ Done splitting!")
    for name, c in counts.items():
        print(f"  {name}: {c} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .jsonl annotations file")
    parser.add_argument("--output_dir", required=True, help="Directory to save split jsonl files")
    parser.add_argument("--train_ids", required=True)
    parser.add_argument("--val_ids", required=True)
    parser.add_argument("--test_ids", required=True)
    args = parser.parse_args()

    split_files = {
        "train": args.train_ids,
        "val": args.val_ids,
        "test": args.test_ids,
    }

    split_annotations(args.input, args.output_dir, split_files)
