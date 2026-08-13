import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def normalize_list_field(value):
    """Extracts first string from list-like fields like ['explicit']"""
    if isinstance(value, list):
        return value[0].strip().lower()
    elif isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return parsed[0].strip().lower()
        except:
            pass
    return str(value).strip().lower()

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "TN": 0,
        "FP": 0,
        "FN": y_true.count(1) - sum(y_pred),
        "TP": sum(y_pred)
    }

def evaluate_per_tag(path, model, prompt_type):
    try:
        df = pd.read_csv(path, encoding='utf-8', engine='python')
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

    df.columns = [c.strip() for c in df.columns]
    df["EXPorIMP_norm"] = df["EXPorIMP"].apply(normalize_list_field)
    df["sentential_scope_norm"] = df["sentential_scope"].apply(normalize_list_field)
    df["EvalEXPorIMP_norm"] = df["EvalEXPorIMP"].astype(str).str.strip().str.lower()
    df["Evalsentential_scope_norm"] = df["Evalsentential_scope"].astype(str).str.strip().str.lower()

    results = []
    for tag in sorted(df["ExpTag"].dropna().unique()):
        tag_df = df[df["ExpTag"] == tag]
        num_samples = len(tag_df)

        if num_samples == 0:
            continue

        # EvalCE
        y_true = [1] * num_samples
        y_pred = tag_df["EvalCE"].fillna(0).astype(int).tolist()
        metrics_ce = compute_metrics(y_true, y_pred)
        metrics_ce = {f"{k}_EvalCE": v for k, v in metrics_ce.items()}

        # EvalEXPorIMP & explicit
        subset_explicit = tag_df[
            (tag_df["EvalEXPorIMP_norm"] == "correct") &
            (tag_df["EXPorIMP_norm"] == "explicit")
        ]
        y_true_exp = [1] * len(tag_df)
        y_pred_exp = [1 if i in subset_explicit.index else 0 for i in tag_df.index]
        metrics_exp = compute_metrics(y_true_exp, y_pred_exp)
        metrics_exp = {f"{k}_EvalEXPorIMP": v for k, v in metrics_exp.items()}

        # EvalSentential_scope & intrasentential
        subset_intra = tag_df[
            (tag_df["Evalsentential_scope_norm"] == "correct") &
            (tag_df["sentential_scope_norm"] == "intrasentential")
        ]
        y_true_sent = [1] * len(tag_df)
        y_pred_sent = [1 if i in subset_intra.index else 0 for i in tag_df.index]
        metrics_sent = compute_metrics(y_true_sent, y_pred_sent)
        metrics_sent = {f"{k}_EvalSentScope": v for k, v in metrics_sent.items()}

        # Combine results
        row = {
            "Model": model,
            "PromptType": prompt_type,
            "ExpTag": tag,
            "NumSamples": num_samples
        }
        row.update(metrics_ce)
        row.update(metrics_exp)
        row.update(metrics_sent)
        results.append(row)

    return results

def process_all(base_folder, output_path):
    prompt_types = ["zero_shot", "few_shot", "chain_of_thought", "least_to_most", "react_style"]
    all_results = []

    for model in os.listdir(base_folder):
        model_path = os.path.join(base_folder, model)
        nested_model_path = os.path.join(model_path, model)
        if not os.path.isdir(nested_model_path):
            continue

        for prompt in prompt_types:
            file_path = os.path.join(nested_model_path, prompt, "llm_results", "results_evaluated.csv")
            if not os.path.isfile(file_path):
                print(f"Missing: {file_path}")
                continue

            result_rows = evaluate_per_tag(file_path, model, prompt)
            all_results.extend(result_rows)

    if not all_results:
        print("No evaluation results generated.")
        return

    final_df = pd.DataFrame(all_results)
    final_df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")

# Run the script
if __name__ == "__main__":
    base_folder = "."  
    output_file = "tagged_evaluation_output.xlsx"
    process_all(base_folder, output_file)
