import os
import pandas as pd
import ast
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# === CUSTOM STOPWORDS ===
custom_stopwords = set([
    "so", "thus", "then", "therefore", "also", "however",
    "moreover", "furthermore", "indeed", "consequently", "additionally"
])

# === KEYWORDS TO TRIGGER STRICT 0 ===
causal_keywords = ["cause", "caused", "causing", "because", "resulting in"]

# === CLEAN TEXT FUNCTIONS ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def clean_text_strict(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.text not in custom_stopwords]

# === EVALUATION HELPERS ===
def evaluate_ce(row):
    try:
        gt_list = ast.literal_eval(row['CauseEffect'])
        pred_list_raw = ast.literal_eval(row['CE'])
        sentence = row.get('sentence', '')
    except:
        return 0

    if not pred_list_raw or row['CE'] == 'null' or row['CE'] is None:
        return 0

    pred_list = []
    for item in pred_list_raw:
        if isinstance(item, dict):
            pred_list.append((item.get('cause', ''), item.get('effect', '')))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pred_list.append((item[0], item[1]))

    if not pred_list or any('...' in str(cause) or '...' in str(effect) for cause, effect in pred_list):
        return 0

    sent_len = len(sentence.split())
    for cause, effect in pred_list:
        if len(str(cause).split()) >= sent_len - 3 or len(str(effect).split()) >= sent_len - 3:
            return 0

    try:
        cleaned_gt = set((clean_text(c), clean_text(e)) for c, e in gt_list)
        cleaned_pred = set((clean_text(c), clean_text(e)) for c, e in pred_list)
    except:
        return 0

    return 1 if cleaned_gt == cleaned_pred else 0.5

def strict_check(row):
    if row["EvalCE"] != 0.5:
        return row["EvalCE"]

    try:
        ce_raw = ast.literal_eval(row["CE"])
        sentence_len = len(str(row.get("sentence", "")).split())
        ce = []
        for item in ce_raw:
            if isinstance(item, dict):
                ce.append((item.get('cause', ''), item.get('effect', '')))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                ce.append((item[0], item[1]))

        for cause, effect in ce:
            cause_clean = clean_text_strict(cause)
            effect_clean = clean_text_strict(effect)
            if len(cause_clean) > 2 or len(effect_clean) > 2:
                return 0
            if len(cause_clean) >= sentence_len - 3 or len(effect_clean) >= sentence_len - 3:
                return 0

        ce_text = " ".join([f"{c} {e}" for c, e in ce]).lower()
        model_cm = ast.literal_eval(row["ModelCM"]) if pd.notna(row["ModelCM"]) else []
        gold_cm = ast.literal_eval(row["causal_marker"]) if pd.notna(row["causal_marker"]) else []
        marker_text = " ".join(model_cm + gold_cm).lower()

        if any(k in ce_text for k in causal_keywords) and any(k in marker_text for k in causal_keywords):
            return 0

    except:
        return 0

    return 0.5

def evaluate_model_cm(row):
    try:
        gt = ast.literal_eval(row['causal_marker']) if pd.notna(row['causal_marker']) else []
        pred = ast.literal_eval(row['ModelCM']) if pd.notna(row['ModelCM']) else []
        if not pred:
            return 'incorrect'
        return 'correct' if any(p in gt for p in pred) else 'incorrect'
    except:
        return 'incorrect'

def evaluate_explicit(row):
    gt = str(row['implicit_vs_explicit']).strip().lower()
    pred_list = ast.literal_eval(row['EXPorIMP']) if pd.notna(row['EXPorIMP']) else []
    if not pred_list:
        return 'incorrect'
    return 'correct' if pred_list[0].strip().lower() == gt else 'incorrect'

def evaluate_scope(row):
    gt = str(row['IntervsIntra']).strip().lower()
    pred_list = ast.literal_eval(row['sentential_scope']) if pd.notna(row['sentential_scope']) else []
    pred = pred_list[0].strip().lower() if pred_list else ""
    pred = pred.replace("intrasentential", "intrasentencial").replace("intersentential", "intersentencial")
    return 'correct' if gt == pred else 'incorrect'

# === MAIN EVALUATION ===
def evaluate_result_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    df.columns = [col.strip() for col in df.columns]

    df["EvalCE"] = df.apply(evaluate_ce, axis=1)
    df["EvalCE"] = df.apply(strict_check, axis=1)
    df["EvalModelCM"] = df.apply(evaluate_model_cm, axis=1)
    df["EvalEXPorIMP"] = df.apply(evaluate_explicit, axis=1)
    df["Evalsentential_scope"] = df.apply(evaluate_scope, axis=1)

    output_path = os.path.join(os.path.dirname(csv_path), "results_evaluated.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Evaluated: {csv_path}")
    print(f"→ Saved: {output_path}\n")

# === WALK THROUGH MODELS ===
def evaluate_all_models(done_folder):
    for model_folder in os.listdir(done_folder):
        model_path = os.path.join(done_folder, model_folder, model_folder)
        if not os.path.isdir(model_path):
            continue
        for prompt_type in ["chain_of_thought", "few_shot", "least_to_most", "react_style", "zero_shot"]:
            csv_path = os.path.join(model_path, prompt_type, "llm_results", "results.csv")
            if os.path.isfile(csv_path):
                evaluate_result_file(csv_path)
            else:
                print(f"Missing: {csv_path}")

# === RUN ===
if __name__ == "__main__":
    base_done_folder = os.path.dirname(os.path.abspath(__file__))
    evaluate_all_models(base_done_folder)
