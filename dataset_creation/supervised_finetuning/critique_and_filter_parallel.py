import os
import requests
from datasets import load_dataset, Dataset
import pandas as pd
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from data_creation import extract_question
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
@dataclass
class ConstitutionalPrinciple:
    """A principle or rule that governs model behavior"""
    name: str
    critique_prompt: str
    revision_prompt: str
    description: Optional[str] = None


class CritiqueAndRevise:
    def __init__(
        self,
        model_name: str = "microsoft/wizardlm-2-8x22b",
        max_new_tokens: int = 128
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.conversation_history = []

    def build_instruct_prompt(self) -> str:
        """
        Prompt builder alternating between 'user' -> 'assistant'.
        """
        prompt = ""
        for turn in self.conversation_history:
            role = turn["role"]
            content = turn["content"]

            if role == "system":
                prompt += f"[System: {content}]\n\n"
            elif role == "user":
                prompt += f"### Instruction:\n{content}\n\n### Response:\n"
            elif role == "assistant":
                prompt += f"{content}\n\n"
        return prompt

    def generate_response(self, temperature: float = 0.1, top_p: float = 0.9) -> str:
        """
        Generate a response given the current conversation_history.
        We build a single string with build_instruct_prompt() and then run .generate().
        """
        prompt_text = self.build_instruct_prompt()
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
            # OpenRouter TOS typically requires a referer
            "HTTP-Referer": "https://mydomain.com"
        }

        payload = {
            "model": self.model_name,  # or the exact name from OpenRouter's model catalog
            "prompt": prompt_text,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.7,
            "stop": ["### Instruction:"]
        }

        response = requests.post(os.environ.get("OPENROUTER_API_URL"), headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        generated_text = data["choices"][0]["text"]
        # Some models produce trailing tokens or repeated instructions; trim extra whitespace
        generated_text = generated_text.strip()
        return generated_text

    def process_question(self, question: str, revised_question: str, principle: ConstitutionalPrinciple):
        """
        Illustrative pipeline:
          1) user asks a question => assistant answers
          2) user critiques (using principle.critique_prompt) => assistant provides critique
          3) user requests revision (using principle.revision_prompt) => assistant revises
        """
        # Clear out old conversation
        self.conversation_history = []

        # (Optional) system role for top-level instructions
        self.conversation_history.append({
            "role": "system",
            "content": (
                "You are a language model that specializes in text rephrasing. Your task is to rephrase user-provided content "
                "to sound more persuasive, while strictly preserving the original meaning and core facts. Do not remove or modify "
                "any essential information. Do not try to answer the query. Focus solely on changing the style and tone to enhance persuasiveness."
            )
        })

        # 1) user question with critique instructions
        self.conversation_history.append({
            "role": "user",
            "content": f"""
Original Query:
{question}
Revised Query:
{revised_question}
Instructions:
{principle.critique_prompt}
"""
        })

        # 2) assistant provides the critique
        critique_answer = self.generate_response(temperature=0.7)
        self.conversation_history.append({
            "role": "assistant",
            "content": critique_answer
        })

        # 3) user revision request (this example does not process the revision response)
        self.conversation_history.append({
            "role": "user",
            "content": principle.revision_prompt
        })

        # If the critique starts with "no", then we mark it with remove = 1
        remove = 1 if critique_answer.lower().startswith("no") else 0

        return {
            "critique": critique_answer,
            "remove": remove
        }


def process_row(row_index: int, question: str, revised_prompt: str, principle: ConstitutionalPrinciple) -> Dict:
    """
    Worker function to process one row. Each worker creates its own CritiqueAndRevise instance.
    """
    # Instantiate a local enforcer for this thread.
    start_time = time.time()
    enforcer = CritiqueAndRevise(model_name="microsoft/wizardlm-2-8x22b", max_new_tokens=150)
    result = enforcer.process_question(question=question, revised_question=revised_prompt, principle=principle)
    elapsed_time = time.time() - start_time

    return {
        "index": row_index,
        "critique_prompt": principle.critique_prompt,
        "critique": result["critique"],
        "remove": result["remove"]
    }


if __name__ == "__main__":
    # Define the constitutional principle
    principle = [
        ConstitutionalPrinciple(
            name="Persuation Principle",
            critique_prompt="""
Identify if the Revised Text maintains the same underlying message and purpose as the Original Text - specifically alterations or omissions of key ideas in the original prompt.
Reply only with one of two responses:
Yes:
Core Message: Fully retains all key ideas and essential details without any omissions.
Tone/Intent: Completely preserves the original tone and intent; persuasive language is applied appropriately without shifting the underlying meaning.
Persuasiveness: Enhances the text's persuasiveness without compromising or altering the original message.
Clarity and Coherence: The text is clear and logically structured.

No:
Core Message: Fails to retain the essential ideas, significantly altering or omitting key points, thus changing the overall message.
Tone/Intent: Introduces a noticeably different tone or intent; persuasive language is overused or misapplied, resulting in a distorted message.
Persuasiveness: The attempt to enhance persuasiveness compromises the original meaning, introducing unintended claims or exaggerations.
Clarity and Coherence: The text lacks clarity or logical structure, making the original intent difficult to discern.
""",
            revision_prompt="""
Based on the specific critique provided, please produce a revised version of the text. Only output one high-quality modified inquiry. The revised version should:
Strict Rules:
- Do NOT address the user directly or try to answer the query.
- Preserve All Essential Content: Ensure that every key idea, fact, and comparison from the original text is retained.
- Enhance Persuasiveness Through Tone: Adjust the language and style to be more persuasive, without altering the underlying message.
"""
        )
    ]
    constitutional_principle = principle[0]

    # Load or create the dataset
    if not os.path.exists("dataset_final.pkl"):
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset_red_team = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        dataset = dataset["train"].to_pandas()
        dataset["question"] = dataset["rejected"].apply(extract_question)
    else:
        dataset = pd.read_pickle("dataset_final_filtered.pkl")  #("dataset_final.pkl")

    # Initialize the new columns
    # dataset['critique_prompt'] = ''
    # dataset['critique'] = ''
    # dataset['remove'] = ''

    # Define the rows to process.
    # We only process rows where the revised prompt is non-null and has more than 10 characters.
    valid_rows = dataset[dataset['revised_prompt'].notna() & (dataset['revised_prompt'].str.len() > 10)]

    # Optionally, if you want to start from a specific row:
    idx = (dataset['critique'].isna()) | ((dataset['critique'].notna()) & (dataset['critique'].str.len() < 2))
    first_matching_index = idx[idx].index[0]
    valid_rows = valid_rows.loc[first_matching_index:]
    #Limit to processing 1000 rows
    print(f"first: {first_matching_index}")
    valid_rows = valid_rows.head(5000)

    # Prepare a list of jobs with row indices and the needed fields
    jobs = []
    for row_index, row in valid_rows.iterrows():
        question = row["question"]
        revised_prompt = row["revised_prompt"]
        jobs.append((row_index, question, revised_prompt))

    # Process rows in parallel using a ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(process_row, row_index, question, revised_prompt, constitutional_principle): row_index
            for (row_index, question, revised_prompt) in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                row_index = futures[future]
                print(f"Error processing row {row_index}: {e}")

    # Update the dataset with the results
    for res in results:
        idx = res["index"]
        print(f"Processing row {idx}")
        dataset.at[idx, "critique_prompt"] = res["critique_prompt"]
        dataset.at[idx, "critique"] = res["critique"]
        dataset.at[idx, "remove"] = res["remove"]

        # Optional: print outputs for each row
        print("====== ORIGINAL ======")
        print(dataset.at[idx, "question"])
        print("====== INITIAL ANSWER ======")
        print(dataset.at[idx, "revised_prompt"])
        print(constitutional_principle.critique_prompt)
        print("\n====== CRITIQUE ======")
        print(res["critique"])
        print("\n-----------------------\n")

    print("Ready to start new job")
    # Save the updated dataset
    dataset.to_pickle("dataset_final_filtered.pkl")