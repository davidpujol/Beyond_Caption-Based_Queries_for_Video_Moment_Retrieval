import os
os.environ["TORCH_COMPILE"] = "0"

import pickle
import json
import copy
from pathlib import Path
from typing import Dict
import argparse
from transformers import pipeline
import torch

import torch._dynamo
torch._dynamo.config.recompile_limit = 128  # or higher
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()                        # ensure clean state

# Initialize LLM pipeline
llm = pipeline("image-text-to-text", model="google/gemma-3-12b-it")

def simplify_query(query: str) -> str:
    prompt = f"""Simplify this query by removing or generalizing unnecessary information, so that the simplified query can match other overspecific ones if possible.
    Format your output as:
    Simplified query: <your simplified query>

    Example 1:
    Query: He then bends down and grabs a ball.
    Simplified query: A person manipulates an object.
    
    Example 2: 
    Query: Then one man stands in a field holding a wooden object and begins twisting it.
    Simplified query: A person manipulates an object.
    
    Example 3:
    Query: There was a penalty and one players attempts to hit the ball into the goal from the side.
    Simplified query: People play a game.

    Example 4:
    Query: A group of people holding paintball guns and dressed in costume run into a staged setting as if in combat.
    Simplified query: People play a game.

    Now, simplify this query:
    Query: {query}
    Simplified query:"""
    
    # ---- FIX: fully disable Dynamo only for model call ----
    try:
        response = llm(prompt, max_new_tokens=80)[0]['generated_text'].strip()
    except Exception as e:
        torch._dynamo.reset()

    return response.split('Simplified query:')[-1].replace('**', '').strip()

def process_annotation(annotation: Dict) -> Dict:
    processed = copy.deepcopy(annotation)
    simplified = simplify_query(annotation["query"])
    processed["query_underspecified"] = simplified
    print("Original:", annotation["query"])
    print("Simplified:", simplified)
    print("---")
    return processed

def process_dataset(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            annotation = json.loads(line)
            processed = process_annotation(annotation)
            outfile.write(json.dumps(processed, ensure_ascii=False) + "\n")
            outfile.flush()  
            os.fsync(outfile.fileno())  # 🔒 forces OS to commit the buffer to disk



def main():
    parser = argparse.ArgumentParser(description="Simplify ActivityNet Captions queries using LLM")
    parser.add_argument("input_file", help="Path to input annotation pickle file")
    parser.add_argument("output_dir", help="Output directory for JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (input_path.stem + "_underspecified_full_queries_llm.jsonl")

    print(f"Processing {input_path} -> {output_file}")
    process_dataset(str(input_path), str(output_file))
    print("Processing complete")

if __name__ == "__main__":
    main()
    
    
# Before anything run
# python -m spacy download en_core_web_sm
    
# Login to hugging face
# pip install huggingface_hub   
# huggingface-cli login 

# Usage ActivityNet
# python dataset_processing/datasets_augmentation_generation/underspecified_queries/activitynet_captions/underspecify_queries3.py ${DATA_ROOT}/activitynet_captions/annotations/original_annos/anet_queries.jsonl ${DATA_ROOT}/activitynet_captions/annotations/underspecified/full_simplification3
