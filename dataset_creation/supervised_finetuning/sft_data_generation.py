from datasets import load_dataset, Dataset
import pandas as pd
import time
from dataset_creation.persuasion_techniques.persuation_techniques import sample_technique
from dataset_creation.data_processing_functions.data_creation import extract_question, PersuasiveRevision
import os
import json

if __name__ == "__main__":
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
    for row_indexer in range(first_matching_index,first_matching_index+1000):
        if "persuation_technique" not in dataset.columns or not isinstance(dataset["persuation_technique"][row_indexer],str) or len(dataset["revised_prompt"][row_indexer])<10:
            print(f"Processing row {row_indexer}")
            start_time = time.time()
            persuation_technique = sample_technique()[0]
            result = PersuasiveRevision(persuation_technique, dataset.loc[row_indexer, "question"], model_name="microsoft/wizardlm-2-8x22b").process_question()

            dataset.loc[row_indexer, "persuation_technique"] = result["persuation_technique"].name
            dataset.loc[row_indexer, "revised_prompt"] = result["revised_question"]
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing row {row_indexer} took {elapsed_time:.2f} seconds")
            print("====== INITIAL PROMPT ======")
            print(dataset.loc[row_indexer,"question"])
            print(f"\n====== REVISED PROMPT ====== {result["persuation_technique"].name}")
            print(result["revised_question"])
            data[row_indexer]={
                "initial_prompt": dataset.loc[row_indexer,"question"],
                "revised_prompt": result["revised_question"],
                "persuation_technique": result["persuation_technique"].name,
            }
        dataset.to_pickle("dataset_final.pkl")
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=4)



