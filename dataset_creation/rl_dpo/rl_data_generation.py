from datasets import load_dataset, Dataset
import pandas as pd
import time
import os
import json
from dataset_creation.rl_dpo.data_creation_rl import extract_question, generate_response, ai_judge
from dotenv import load_dotenv

load_dotenv(override=True)

if __name__ == "__main__":
    print("Creating dataset...")
    if not os.path.exists("dataset_final_rl.pkl"):
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset_red_team = load_dataset("Anthropic/hh-rlhf", data_dir = "red-team-attempts")
        dataset_red_team = dataset_red_team["train"].to_pandas()
        dataset_red_team["question"] = dataset_red_team["transcript"].apply(extract_question)
    else:
        dataset = pd.read_pickle("dataset_final_rl.pkl")
    dataset_red_team_train_subset = dataset_red_team[dataset_red_team["task_descripton_harmlessness_score"]<0.5].sort_values(by = "task_descripton_harmlessness_score").reset_index()
    if "responseA" not in dataset_red_team_train_subset.columns:
        dataset_red_team_train_subset["responseA"] = ""
    if "winner" not in dataset_red_team_train_subset.columns:
        dataset_red_team_train_subset["winner"] = 100
    idx = (dataset_red_team_train_subset.winner==100)
    data = {}
    # Find first occurrence
    first_matching_index = idx[idx].index[0]
    for row_indexer in range(first_matching_index,first_matching_index+10):
        if "winner" not in dataset_red_team_train_subset.columns or not isinstance(dataset_red_team_train_subset["responseA"][row_indexer],str) or len(dataset_red_team_train_subset["responseA"][row_indexer])<10:
            print(f"Processing row {row_indexer}")
            start_time = time.time()
            prompt = dataset_red_team_train_subset.loc[row_indexer, "question"]
            responseA = generate_response(prompt, model_name = "microsoft/wizardlm-2-8x22b")
            responseB = generate_response(prompt, model_name = "microsoft/wizardlm-2-8x22b")
            dataset_red_team_train_subset.loc[row_indexer, "responseA"] = responseA
            dataset_red_team_train_subset.loc[row_indexer, "responseB"] = responseB
            winner = ai_judge(prompt, responseA, responseB)
            dataset_red_team_train_subset.loc[row_indexer, "winner"] = winner
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing row {row_indexer} took {elapsed_time:.2f} seconds")
            print("====== INITIAL PROMPT ======")
            print(dataset_red_team_train_subset.loc[row_indexer,"question"])
            print("\n====== REVISED PROMP A ====== ")
            print(responseA)
            print("\n====== REVISED PROMP B ====== ")
            print(responseB)
            print("\n====== WINNER ========")
            print(winner)

            data[row_indexer]={
                "initial_prompt": dataset_red_team_train_subset.loc[row_indexer,"question"],
                "responseA": responseA,
                "responseB": responseB,
                "winner": winner,
            }
        dataset_red_team_train_subset.to_pickle("dataset_final_rl.pkl")
        with open('data_rl.json', 'w') as f:
            json.dump(data, f, indent=4)



