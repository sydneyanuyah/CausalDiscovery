import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from prompts import prompts
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import numpy as np
import torch

SEED = 4000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def build_prompt(template, sentence):
    return (
        template['system']
        + "\n\n"
        + template['user'].format(sentence=sentence)
    )

def extract_strict_json(text):
    try:
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace == -1 or last_brace == -1:
            return None, f"No braces found in output: {text}"
        json_str = text[first_brace:last_brace+1]
        obj = json.loads(json_str)
        if (
            isinstance(obj, dict)
            and "answer" in obj
            and obj["answer"] in ["causal", "noncausal"]
        ):
            return obj["answer"], ""
        else:
            return None, f"JSON missing 'answer' key or invalid value: {json_str}"
    except Exception as e:
        return None, f"Malformed JSON: {e} | Raw: {text}"

def compute_metrics(true_labels, pred_labels):
    label_map = {'causal': 1, 'noncausal': 0}
    
    # Map labels to numeric values; use -1 for invalid/missing labels
    true_numeric = [label_map.get(l, -1) for l in true_labels]
    pred_numeric = [label_map.get(l, -1) for l in pred_labels]

    # Determine valid indices (where predictions are valid)
    valid_idx = [i for i, (t, p) in enumerate(zip(true_numeric, pred_numeric)) if t != -1 and p != -1]

    # Filtered valid labels
    true_valid = [true_numeric[i] for i in valid_idx]
    pred_valid = [pred_numeric[i] for i in valid_idx]

    total_samples = len(true_labels)
    valid_samples = len(true_valid)

    # Calculate accuracy based on ALL original samples
    correct_predictions = sum(1 for t, p in zip(true_numeric, pred_numeric) if t != -1 and t == p)
    accuracy = correct_predictions / total_samples if total_samples else 0

    # Handle the special case of no valid predictions
    if valid_samples == 0:
        precision = recall = f1 = 0
        conf_matrix = [[0,0],[0,0]]
    else:
        precision = precision_score(true_valid, pred_valid, zero_division=0)
        recall = recall_score(true_valid, pred_valid, zero_division=0)
        f1 = f1_score(true_valid, pred_valid, zero_division=0)
        conf_matrix = confusion_matrix(true_valid, pred_valid, labels=[0,1]).tolist()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
        "total_samples": total_samples,
        "valid_predictions": valid_samples,
        "missing_or_invalid_predictions": total_samples - valid_samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to CSV data file")
    parser.add_argument('--model_name', type=str, required=True, help="Huggingface model name or path")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_col', type=str, default="Label", help="Column containing ground truth labels")
    args = parser.parse_args()

    model_folder = args.model_name.split('/')[-1]
    os.makedirs(model_folder, exist_ok=True)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer
    )
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(args.data)
    if 'sentence' not in df.columns:
        raise ValueError("Input CSV must have a 'sentence' column.")
    sentences = df['sentence'].tolist()

    total_start = time.time()
    for prompt_key, template in prompts.items():
        print(f"Running inference for prompt: {prompt_key}")
        preds, errors, outputs, batch_times = [], [], [], []

        # Subfolder per prompt key
        prompt_folder = os.path.join(model_folder, prompt_key)
        os.makedirs(prompt_folder, exist_ok=True)

        for i in tqdm(range(0, len(sentences), args.batch_size)):
            batch = sentences[i:i+args.batch_size]
            prompts_batch = [build_prompt(template, s) for s in batch]
            batch_start = time.time()
            gen_outputs = pipe(
                prompts_batch,
                max_new_tokens=512,
                batch_size=args.batch_size,
                truncation=True,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            batch_end = time.time()
            per_instance_time = (batch_end - batch_start) / len(batch)
            for j, out in enumerate(gen_outputs):
                generated = out[0]['generated_text'].strip()
                label, error = extract_strict_json(generated)
                preds.append(label)
                errors.append(error)
                outputs.append(generated)
                batch_times.append(per_instance_time)

        # Results DataFrame
        out_df = df.copy()
        out_df[f"llm_prediction"] = preds
        out_df[f"llm_error"] = errors
        out_df[f"llm_raw_output"] = outputs

        llm_results_path = os.path.join(prompt_folder, 'llm_results.csv')
        out_df.to_csv(llm_results_path, index=False)

        # Compute metrics
        if args.label_col in out_df.columns:
            label_map = {0: "noncausal", 1: "causal"}
            true_labels_mapped = out_df[args.label_col].map(label_map).tolist()
            metrics = compute_metrics(true_labels_mapped, preds)
        else:
            metrics = {}
        
        metrics["avg_inference"] = float(np.mean(batch_times)) if batch_times else 0
        metrics["min_inference"] = float(np.min(batch_times)) if batch_times else 0
        metrics["max_inference"] = float(np.max(batch_times)) if batch_times else 0
        metrics["num_samples"] = len(sentences)
        metrics["total_runtime"] = float(sum(batch_times))

        results_json_path = os.path.join(prompt_folder, 'results.json')
        with open(results_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved outputs to {llm_results_path} and {results_json_path}")

    print(f"All prompts completed for model {args.model_name}.")

if __name__ == "__main__":
    main()
