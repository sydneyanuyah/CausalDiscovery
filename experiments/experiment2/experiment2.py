import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import json
import time
import random
import numpy as np
from itertools import zip_longest
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
    parser.add_argument('--model',       type=str, required=True,
                        help="Model repo (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument('--data',        type=str, default="data/Experiment2.csv")
    parser.add_argument('--batch_size',  type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument('--top_k',       type=int, default=40)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed',        type=int, default=SEED)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--exp_folder',  type=str, default="exp2i",
                        help="Base experiment folder name")
    parser.add_argument(
        '--skip_prompts',
        nargs='+',
        default=[],
        help="Names of PROMPTS2 keys to skip (e.g. --skip_prompts prompt1 prompt3)"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model_name_last = get_model_name_last(args.model)
    base_folder = os.path.join(args.exp_folder, model_name_last)
    ensure_dir(base_folder)

    prompts_to_run = {
        name: tmpl
        for name, tmpl in PROMPTS2.items()
        if name not in args.skip_prompts
    }

    print(f"Loading tokenizer and 4-bit model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
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

    for prompt_name, prompt_template in prompts_to_run.items():
        print(f"\n=== Processing prompt: {prompt_name} ===")
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

        parsed_results = [extract_json_from_response(out) for out in outputs]

        for p, raw_out, parsed in zip(prompts, outputs, parsed_results):
            print("PROMPT:")
            print(p)
            print("RAW MODEL OUTPUT:")
            print(raw_out)
            print("PARSED JSON:")
            print(json.dumps(parsed, indent=2))
            print("-" * 80)

        ce_lists = []
        modelcm_lists = []

        for idx, parsed in enumerate(parsed_results):
            try:
                # CE pairs: keep only numeric-suffixed keys
                numeric_ck = [
                    k for k in parsed
                    if k.startswith('cause_') and k.rsplit('_', 1)[1].isdigit()
                ]
                cause_keys = ['cause'] + sorted(
                    numeric_ck,
                    key=lambda x: int(x.rsplit('_', 1)[1])
                )
                numeric_ek = [
                    k for k in parsed
                    if k.startswith('effect_') and k.rsplit('_', 1)[1].isdigit()
                ]
                effect_keys = ['effect'] + sorted(
                    numeric_ek,
                    key=lambda x: int(x.rsplit('_', 1)[1])
                )

                ce = []
                for ck, ek in zip_longest(cause_keys, effect_keys, fillvalue=None):
                    c = parsed.get(ck, "") or ""
                    e = parsed.get(ek, "") or ""
                    if c or e:
                        ce.append((c, e))
                ce_lists.append(ce)

                # ModelCM: handle plural list, singular, and numeric keys
                markers = []
                if 'causal_markers' in parsed and isinstance(parsed['causal_markers'], list):
                    markers = parsed['causal_markers']
                else:
                    if 'causal_marker' in parsed:
                        singular = parsed['causal_marker']
                        if isinstance(singular, (list, tuple)):
                            markers.extend(singular)
                        elif singular:
                            markers.append(singular)
                    numeric_mk = [
                        k for k in parsed
                        if k.startswith('causal_marker_') and k.rsplit('_', 1)[1].isdigit()
                    ]
                    for k in sorted(numeric_mk, key=lambda x: int(x.rsplit('_', 1)[1])):
                        val = parsed.get(k)
                        if isinstance(val, list):
                            markers.extend(val)
                        elif val:
                            markers.append(val)
                mc = [(m,) for m in markers]
                modelcm_lists.append(mc)

            except Exception as e:
                print(f"[WARN] skipping row {idx} due to: {e}")
                ce_lists.append([])
                modelcm_lists.append([])
                continue

        df_out = df.copy().reset_index(drop=True)
        df_out['CE'] = [lst if lst else np.nan for lst in ce_lists]
        df_out['ModelCM'] = [lst if lst else np.nan for lst in modelcm_lists]

        def norm_exp(val, n):
            v = (val or "").strip().lower()
            if v in ('explicit', 'implicit'):
                return [v] * n
            return ['Invalid'] * n

        df_out['EXPorIMP'] = [
            norm_exp(df.loc[i, 'implicit_vs_explicit'], len(ce_lists[i]))
            for i in range(len(df_out))
        ]

        def norm_scope(val, n):
            v = (val or "").strip().lower()
            if v.startswith('intra'):
                tag = 'intrasentential'
            elif v.startswith('inter'):
                tag = 'intersentential'
            else:
                tag = 'Invalid'
            return [tag] * n

        df_out['sentential_scope'] = [
            norm_scope(df.loc[i, 'IntervsIntra'], len(ce_lists[i]))
            for i in range(len(df_out))
        ]

        out_csv_path = os.path.join(prompt_folder, "results.csv")
        df_out.to_csv(out_csv_path, index=False, na_rep='null')

        timings_path = os.path.join(prompt_folder, "timings.json")
        with open(timings_path, 'w') as f:
            json.dump(timings, f, indent=2)

        print(f"Saved results for {prompt_name} in {prompt_folder}")

if __name__ == "__main__":
    main()
