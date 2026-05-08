# This script reads an annotation file and computes the average word count per annotation.

import json
import sys

def count_words(text):
    return len(text.strip().split())

def main(filepath):
    narration_word_counts = []
    query_word_counts = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # skip blank lines
            obj = json.loads(line)
            try:
                narration = obj["narration"]
            except:
                narration = obj["query"]

            query = obj["query_underspecified"]

            narration_word_counts.append(count_words(narration))
            query_word_counts.append(count_words(query))

    if narration_word_counts:
        avg_narration = sum(narration_word_counts) / len(narration_word_counts)
    else:
        avg_narration = 0.0

    if query_word_counts:
        avg_query = sum(query_word_counts) / len(query_word_counts)
    else:
        avg_query = 0.0

    print(f"Average original query words: {avg_narration:.2f}")
    print(f"Average query_underspecified words: {avg_query:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input.jsonl")
        sys.exit(1)
    main(sys.argv[1])

# Usage:

# QVHighlights:
# python average_word_count.py ${DATA_ROOT}/QVHighlights/underspecified/llm_based/manual_reannotation/qvhighlights_val_underspecified_full_simplification_llm_reannotated_final.jsonl

# HD_EPIC:
# python average_word_count.py ${DATA_ROOT}/HD_EPIC/underspecified/attribute_modifiers_simplification/HD_EPIC_Narrations_underspecified_attributes_modifiers_only_queries_llm_val.jsonl
# python average_word_count.py ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification/HD_EPIC_Narrations_underspecified_full_queries_llm_val.jsonl

# Synopground:
# python average_word_count.py ${DATA_ROOT}/SynopGround/annotations/underspecified/atttribute_modifiers_simplification/val_underspecified_attributes_modifiers_only_queries_llm.jsonl



