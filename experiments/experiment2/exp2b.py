import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import json
import time
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompts import PROMPTS2

SEED = 4000

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_model_name_last(model_path):
    return model_path.rstrip('/').split('/')[-1]

def quantize_4bit(model_path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    )

def run_llm_batch(
    model,
    tokenizer,
    prompts,
    batch_size=64,
    top_k=40,
    temperature=0.7,
    max_new_tokens=256
):
    all_outputs = []
    timings = []
    n = len(prompts)
    for i in tqdm(range(0, n, batch_size)):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_time = time.time() - start_time
        for prompt, out in zip(batch, outputs):
            timings.append({"inference_time": batch_time / len(batch)})
            clean_out = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
            all_outputs.append(clean_out)
    return all_outputs, timings

def extract_json_from_response(response):
    try:
        start = response.find('{')
        end = response.rfind('}')
        if start == -1 or end == -1:
            return {}
        return json.loads(response[start:end+1])
    except Exception:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model repo (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument('--data', type=str, required=True, help="Path to CSV file")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for inference")
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=4000)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--exp_folder', type=str, default="exp2i", help="Base experiment folder name")
    args = parser.parse_args()

    set_seed(args.seed)

    model_name_last = get_model_name_last(args.model)
    base_folder = os.path.join(args.exp_folder, model_name_last)
    ensure_dir(base_folder)

    print(f"Loading tokenizer and 4-bit model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    # Padding fix:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = quantize_4bit(args.model)
    model.eval()

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    all_sentences = df['sentence'].tolist()

    for prompt_name, prompt_template in PROMPTS2.items():
        print(f"Processing prompt: {prompt_name}")

        prompt_folder = os.path.join(base_folder, prompt_name, "llm_results")
        ensure_dir(prompt_folder)

        prompts = [
            prompt_template.format(input_sentence=sent)
            for sent in all_sentences
        ]

        outputs, timings = run_llm_batch(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
            top_k=args.top_k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )

        parsed_results = []
        for out in outputs:
            parsed = extract_json_from_response(out)
            parsed_results.append(parsed)

        # Determine superset of all output fields (cause, cause_2, ..., effect, effect_2, ...)
        all_keys = set()
        for parsed in parsed_results:
            all_keys.update(parsed.keys())
        # Focus on causal extraction fields
        base_fields = ['cause', 'effect', 'causal_markers', 'explicitness', 'sentential_scope']
        desired_keys = []
        for base in base_fields:
            # Add base (un-numbered) field first
            if base in all_keys:
                desired_keys.append(base)
            # Add all _2, _3, ... for this base
            numbered = [k for k in all_keys if k.startswith(base + "_")]
            # Sort for natural order: base_2, base_3, ...
            numbered.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
            desired_keys.extend(numbered)

        out_rows = []
        for orig_row, parsed in zip(df.to_dict(orient='records'), parsed_results):
            row = dict(orig_row)
            for key in desired_keys:
                row[key] = parsed.get(key, "")
            out_rows.append(row)

        out_df = pd.DataFrame(out_rows)
        out_csv_path = os.path.join(prompt_folder, "results.csv")
        out_df.to_csv(out_csv_path, index=False)

        timings_path = os.path.join(prompt_folder, "timings.json")
        with open(timings_path, 'w') as f:
            json.dump(timings, f, indent=2)

        print(f"Saved results for {prompt_name} in {prompt_folder}")

if __name__ == "__main__":
    main()
