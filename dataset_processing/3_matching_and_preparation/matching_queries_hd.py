from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict
import jsonlines
import os
from transformers import pipeline
import re

os.environ["TORCH_COMPILE"] = "0"

def stage1_cluster_and_merge(input_file: str, temp_file: str, similarity_threshold: float = 0.85):
    print("Stage 1: Clustering and merging with embedding model...")
    embedding_model = SentenceTransformer('stsb-roberta-large')

    annotations = []
    with open(input_file, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    
    video_groups = defaultdict(list)
    for ann in annotations:
        video_groups[ann['video_id']].append(ann)
    
    with jsonlines.open(temp_file, mode='w') as writer:
        for video_id, anns in video_groups.items():
            print(f"Processing video {video_id}...")
            queries = [ann['query_underspecified'] for ann in anns]
            embeddings = embedding_model.encode(queries)
            sim_matrix = cosine_similarity(embeddings)

            similar_pairs = []
            for i in range(len(queries)):
                for j in range(i+1, len(queries)):
                    if sim_matrix[i][j] >= similarity_threshold:
                        similar_pairs.append((i, j))    # Mark these two indices as similar (they indicate the same fundamental meaning)
            
            graph = defaultdict(list)
            for i, j in similar_pairs:
                graph[i].append(j)
                graph[j].append(i)
            
            
            # Perform DFS to find connected components (clusters)
            visited = set()
            clusters = []
            for node in range(len(anns)):
                if node not in visited:
                    cluster = []
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            cluster.append(current)
                            stack.extend(graph[current])
                    clusters.append(cluster)
            
            for cluster in clusters:
                cluster_anns = [anns[i] for i in cluster]
                original_queries = [a['query_underspecified'] for a in cluster_anns]
                intervals = [[a['start_timestamp'], a['end_timestamp']] for a in cluster_anns]

                # Placeholder narration: just use first original query for now
                placeholder_narration = original_queries[0]

                writer.write({
                    "video_id": video_id,
                    "participant_id": cluster_anns[0]['participant_id'],
                    "narration": cluster_anns[0]['narration'],
                    "query_underspecified": placeholder_narration,
                    "relevant_windows": intervals,
                    "unique_narration_id": cluster_anns[0]['unique_narration_id'],
                    "merged_from": [a['unique_narration_id'] for a in cluster_anns],
                    "original_queries": original_queries
                })
    print(f"Stage 1 complete. Intermediate results saved to {temp_file}.")


def stage2_generate_final_narrations(temp_file: str, output_file: str):
    print("Stage 2: Generating final unified narrations with Gemma...")
    use_gemma = False

    if use_gemma:
        gemma = pipeline("image-text-to-text", model="google/gemma-3-12b-it")

    with jsonlines.open(temp_file, mode='r') as reader, \
         jsonlines.open(output_file, mode='w') as writer:
        
        for ann in reader:
            if len(ann['merged_from']) > 1:
                # Use Gemma to combine them into a new unified query
                if use_gemma:
                    print("Merging multiple queries for video:", ann['video_id'])
                    print("Original queries:", ann['original_queries'])
                    # Multiple GT merged: unify queries with Gemma
                    prompt = f"""
                        Combine these queries into a single unified description that applies to all the queries. Avoid adding unncessary ands/ors, trying to make it as concise as possible:
                        Example 1:
                        Queries: ["Hold the pan using the left hand to cook tomatoes", "Hold the pot using the right hand to cook onions"]
                        Unified description: "Hold the kitchenware using a hand to cook vegatables."   
                        Example 2:
                        Queries: ["Take off the left glove.", "Take off the right glove"]
                        Unified description: "Take off the glove."  
                        Now create a unified description for the following queries:
                        Queries: {ann['original_queries']}
                        Unified description:
                    """
                    
                    response = gemma(prompt, max_new_tokens=80)[0]['generated_text'].strip()
                    unified_query = response.split('Unified description:')[-1].split('.')[0].replace('**', '').strip()
                    
                    # Remove excessive quotes, newlines, tabs, multiple spaces
                    # Replace any sequence of: double quotes, single quotes, newlines, tabs, multiple spaces -> single space
                    unified_query = re.sub(r'["\'\n\r\t]+', ' ', unified_query)  # replace any quotes/newlines/tabs with space
                    unified_query = re.sub(r'\s+', ' ', unified_query)  # condense multiple spaces to one
                    unified_query = unified_query.strip()
                    
                    print("Unified query:", unified_query)
                    ann['query_underspecified'] = unified_query
                else:
                    # Just choose the first query as a placeholder (in case there is any error with Gemma)
                    ann['query_underspecified'] = ann['original_queries'][0]
            else:
                # Single GT: keep underspecified query as is
                ann['query_underspecified'] = ann['query_underspecified']
            
            writer.write(ann)
    print(f"Stage 2 complete. Final merged annotations saved to {output_file}.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input JSONL file with underspecified queries')
    parser.add_argument('output_file', help='Final output JSONL file with merged annotations')
    parser.add_argument('--temp_file', default='temp_merged.jsonl', help='Temporary intermediate file')
    parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold')
    args = parser.parse_args()

    stage1_cluster_and_merge(args.input_file, args.temp_file, args.threshold)
    
    # Free embedding model memory explicitly if needed
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    stage2_generate_final_narrations(args.temp_file, args.output_file)
    
    # Optionally remove temp file
    os.remove(args.temp_file)
    
# Example usage:
# python matching_queries_hd.py '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v1/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl' --threshold=0.9

# python matching_queries_hd.py '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v2/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl' --threshold=0.9

# python matching_queries_hd.py '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v3/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl' --threshold=0.9

# python matching_queries_hd.py '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm.jsonl' '${DATA_ROOT}/HD_EPIC/underspecified/full_simplification_v4/HD_EPIC_Narrations_underspecified_full_queries_llm_merged_fixed.jsonl' --threshold=0.8


# Installation:
# pip install sentence-transformers transformers jsonlines scikit-learn