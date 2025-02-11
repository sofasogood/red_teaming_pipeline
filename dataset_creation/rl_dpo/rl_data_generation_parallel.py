from datasets import load_dataset
import pandas as pd
import time
import os
import json
from dataset_creation.persuasion_techniques.data_creation import extract_question
from dataset_creation.rl_dpo.data_creation_rl import ai_judge, generate_response
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
from functools import partial

load_dotenv(override=True)

def process_single_row(row_data, save_interim=True):
    """
    Process a single row of the dataset.
    
    Args:
        row_data (tuple): (row_index, row_series)
        save_interim (bool): Whether to save interim results
    
    Returns:
        dict: Processed data for the row
    """
    row_index, row = row_data
    start_time = time.time()
    
    prompt = row["question"]
    # persuasion_technique = sample_technique()[0]
    # resultA = PersuasiveRevision(persuasion_technique, dataset.loc[row_index, "question"], model_name="microsoft/wizardlm-2-8x22b").process_question()

    responseA = generate_response(prompt, model_name="microsoft/wizardlm-2-8x22b")
    responseB = generate_response(prompt, model_name="microsoft/wizardlm-2-8x22b")
    winner = ai_judge(prompt, responseA, responseB)
    
    elapsed_time = time.time() - start_time
    
    print(f"Processing row {row_index} took {elapsed_time:.2f} seconds")
    print("====== INITIAL PROMPT ======")
    print(prompt)
    print("\n====== REVISED PROMPT A ====== ")
    print(responseA)
    print("\n====== REVISED PROMPT B ====== ")
    print(responseB)
    print("\n====== WINNER ========")
    print(winner)
    
    result = {
        "row_index": row_index,
        "data": {
            "initial_prompt": prompt,
            "responseA": responseA,
            "responseB": responseB,
            "winner": winner,
        }
    }
    
    if save_interim:
        save_interim_results(result)
    
    return result

def save_interim_results(result):
    """Save interim results to avoid losing progress"""
    interim_file = f"interim_results_{result['row_index']}.json"
    with open(interim_file, 'w') as f:
        json.dump(result['data'], f, indent=4)

def merge_interim_results():
    """Merge all interim result files into final output"""
    all_data = {}
    for filename in os.listdir():
        if filename.startswith("interim_results_") and filename.endswith(".json"):
            with open(filename, 'r') as f:
                row_index = int(filename.split('_')[2].split('.')[0])
                all_data[row_index] = json.load(f)
            os.remove(filename)  # Clean up interim file
    return all_data

if __name__ == "__main__":
    print("Creating dataset...")
    
    # Load or create initial dataset
    if not os.path.exists("dataset_final_rl.pkl"):
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset_red_team = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        dataset_red_team = dataset_red_team["train"].to_pandas()
        dataset_red_team["question"] = dataset_red_team["transcript"].apply(extract_question)
    else:
        dataset_red_team = pd.read_pickle("dataset_final_rl.pkl")
    
    # Prepare dataset subset
    dataset_red_team_train_subset = dataset_red_team[
        dataset_red_team["task_descripton_harmlessness_score"] < 0.5
    ].sort_values(by="task_descripton_harmlessness_score")  ##.reset_index()
    
    if "responseA" not in dataset_red_team_train_subset.columns:
        dataset_red_team_train_subset["responseA"] = ""
    if "winner" not in dataset_red_team_train_subset.columns:
        dataset_red_team_train_subset["winner"] = 100
    
    # Find rows to process
    idx = (dataset_red_team_train_subset.winner == 100)
    first_matching_index = idx[idx].index[0]
    rows_to_process = []
    
    for row_index in range(first_matching_index, first_matching_index + 10):
        row = dataset_red_team_train_subset.iloc[row_index]
        if ("winner" not in dataset_red_team_train_subset.columns or 
            not isinstance(row["responseA"], str) or 
            len(row["responseA"]) < 10):
            rows_to_process.append((row_index, row))
    
    # Process rows in parallel
    num_processes = min(cpu_count(), len(rows_to_process))  # Don't create more processes than rows
    print(f"Starting parallel processing with {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(partial(process_single_row, save_interim=True), rows_to_process)
    
    # Merge all results
    data = merge_interim_results()
    
    # Update the main dataset with results
    for result in results:
        row_index = result["row_index"]
        row_data = result["data"]
        dataset_red_team_train_subset.loc[row_index, "responseA"] = row_data["responseA"]
        dataset_red_team_train_subset.loc[row_index, "responseB"] = row_data["responseB"]
        dataset_red_team_train_subset.loc[row_index, "winner"] = row_data["winner"]
    
    # Save final results
    dataset_red_team_train_subset.to_pickle("dataset_final_rl.pkl")
    with open('data_rl.json', 'w') as f:
        json.dump(data, f, indent=4)
