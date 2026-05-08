import pickle
import json
import copy
from pathlib import Path
from typing import Dict
import argparse
from transformers import pipeline
import torch
import os
os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.recompile_limit = 128  # or higher

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Initialize LLM pipeline
llm = pipeline("image-text-to-text", model="google/gemma-3-12b-it")

def simplify_query(query: str) -> str:
    prompt = f"""Simplify this query by removing or generalizing unnecessary information, but keeping the sentence’s meaning. Avoid using very vague words like "something" or "items". 
    Format your output as:
    Simplified query: <your simplified query>

    Example 1:
    Query: Throw the big orange into the green recyclable bag using the left hand. With the right hand, pick up the end of the green roll and pull the roll outwards to pull one bag out of the roll using the right hand.
    Simplified query: Throw an orange away.

    Example 2:
    Query: Open the upper cupboard by holding the handle of the cupboard with the left hand.
    Simplified query: Open a cupboard.

    Example 3:
    Query: Switch the button of the socket using the left hand. This enables the power to access the food processor so as to start it.
    Simplified query: Switch a button on. 
    
    Example 4: 
    Query: Turn the dial of the food processor by turning it clockwise to switch it on. The juicer will now start rotating
    Simplied query: Turn a dial on.
    
    Example 5:
    Query: Using the left hand, remove the plastic cover of the blue scissors. This action occurs at the periphery, quite off-screen.
    Simplified query: Remove a cover.

    
    Now, simplify this query:
    Query: {query}
    Simplified query:"""
    
    response = llm(prompt, max_new_tokens=80)[0]['generated_text'].strip()
    return response.split('Simplified query:')[-1].split('.')[0].replace('**', '').strip()

def process_annotation(annotation: Dict) -> Dict:
    processed = copy.deepcopy(annotation)
    simplified = simplify_query(annotation["narration"])
    processed["query_underspecified"] = simplified
    return processed

def process_dataset(input_path: str, output_path: str):
    with open(input_path, "rb") as infile:
        data = pickle.load(infile)
    
    with open(output_path, "w") as outfile:
        for row in data.iterrows():
            processed = process_annotation(row[1].to_dict())
            outfile.write(json.dumps(processed) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Simplify HD_EPIC narrations without recursion")
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

# Usage HD-EPIC:
# python dataset_processing/datasets_augmentation_generation/underspecified_queries/hd_epic/underspecify_queries_hd_epic_v3.py ${DATA_ROOT}/HD_EPIC/HD_EPIC_Narrations.pkl ${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3