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
#torch._dynamo.config.recompile_limit = 128  # or higher
torch._dynamo.config.suppress_errors = True


# Initialize LLM pipeline
llm = pipeline("image-text-to-text", model="google/gemma-3-12b-it")

def simplify_query(query: str) -> str:
    prompt = f"""Simplify this query by removing or generalizing unnecessary information, but keeping the core sentence’s meaning.  
    Format your output as:
    Simplified query: <your simplified query>

    Example 1:
    Query: A dog in socks walks slowly out onto the floor as a lady films him.
    Simplified query: A dog moves.
    
    Example 2:
    Query: He wears the shoes over black socks and ties the shoe laces securely.
    Simplified query: A person ties his shoes.
    
    Example 3:
    Query: A group of people holding paintball guns and dressed in costume run into a staged setting as if in combat.
    Simplified query: People play a game.
    
    Now, simplify this query:
    Query: {query}
    Simplified query:"""
    
    response = llm(prompt, max_new_tokens=80)[0]['generated_text'].strip()
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
# python dataset_processing/datasets_augmentation_generation/underspecified_queries/activitynet_captions/underspecify_queries.py ${DATA_ROOT}/activitynet_captions/annotations/anet_queries.jsonl ${DATA_ROOT}/activitynet_captions/annotations/underspecified/full_simplification1
