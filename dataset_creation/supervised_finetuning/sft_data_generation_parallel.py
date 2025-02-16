import multiprocessing
from functools import partial
import time
import json
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from dataset_creation.persuasion_techniques.persuation_techniques import sample_technique
from dataset_creation.data_processing_functions.data_creation import extract_question, PersuasiveRevision
import os
def process_single_row(row_data: Dict[str, Any], model_name: str = "microsoft/wizardlm-2-8x22b") -> Dict[str, Any]:
    """Process a single row of the dataset"""
    try:
        row_index, row = row_data
        
        # Skip if already processed
        if ("persuation_technique" in row and 
            isinstance(row["persuation_technique"], str) and 
            len(row["revised_prompt"]) >= 10):
            return None
            
        start_time = time.time()
        persuation_technique = sample_technique()[0]
        
        result = PersuasiveRevision(
            persuation_technique, 
            row["question"], 
            model_name=model_name
        ).process_question()
        
        elapsed_time = time.time() - start_time
        print(f"Processing row {row_index} took {elapsed_time:.2f} seconds")
        
        print("====== INITIAL PROMPT ======")
        print(row["question"])
        print(f"\n====== REVISED PROMPT ====== {result['persuation_technique'].name}")
        print(result["revised_question"])
        
        return {
            "row_index": row_index,
            "data": {
                "initial_prompt": row["question"],
                "revised_prompt": result["revised_question"],
                "persuation_technique": result["persuation_technique"].name,
            }
        }
    except Exception as e:
        print(f"Error processing row {row_index}: {str(e)}")
        return None

def process_dataset_parallel(dataset: pd.DataFrame, 
                           start_index: int,
                           batch_size: int = 1000,
                           num_processes: int = 4,
                           model_name: str = "microsoft/wizardlm-2-8x22b",
                           checkpoint_frequency: int = 10) -> Dict[int, Dict]:
    """
    Process the dataset in parallel using multiple processes
    """
    # Create a pool of workers
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Prepare the data for parallel processing
    rows_to_process = dataset.iloc[start_index:start_index + batch_size].iterrows()
    
    # Create a partial function with the fixed model_name
    process_func = partial(process_single_row, model_name=model_name)
    
    # Process rows in parallel with progress bar
    results = []
    processed_count = 0
    
    for result in tqdm(pool.imap_unordered(process_func, rows_to_process), 
                      total=batch_size):
        if result is not None:
            results.append(result)
            processed_count += 1
            
            # Checkpoint save
            if processed_count % checkpoint_frequency == 0:
                save_checkpoints(dataset, results)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Final save
    save_checkpoints(dataset, results)
    
    # Convert results to dictionary format
    return {r["row_index"]: r["data"] for r in results if r is not None}

def save_checkpoints(dataset: pd.DataFrame, results: list):
    """Save intermediate results to files"""
    # Update dataset
    for result in results:
        if result is not None:
            row_index = result["row_index"]
            data = result["data"]
            dataset.loc[row_index, "persuation_technique"] = data["persuation_technique"]
            dataset.loc[row_index, "revised_prompt"] = data["revised_prompt"]
    
    # Save to pickle and JSON
    dataset.to_pickle("dataset_final.pkl")
    
    # Convert results to the format needed for JSON
    data_dict = {r["row_index"]: r["data"] for r in results if r is not None}
    with open('data.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

# Example usage
if __name__ == "__main__":
    # Load your dataset
    print("Creating dataset...")
    if not os.path.exists("dataset_final.pkl"):
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset_red_team = load_dataset("Anthropic/hh-rlhf", data_dir = "red-team-attempts")
        dataset = dataset["train"].to_pandas()
        dataset["question"] = dataset["rejected"].apply(extract_question)
    else:
        dataset = pd.read_pickle("dataset_final.pkl")
    idx = (dataset.revised_prompt.isna()) | (dataset.revised_prompt.notna() & (dataset['revised_prompt'].str.len() < 10))
    data = {}
    # Find first occurrence
    first_matching_index = idx[idx].index[0]
    batch_size = 6500
    num_processes = multiprocessing.cpu_count() - 1  # Leave one CPU free
    
    # Process the dataset in parallel
    results = process_dataset_parallel(
        dataset=dataset,
        start_index=first_matching_index,
        batch_size=batch_size,
        num_processes=num_processes,
        checkpoint_frequency=10
    )