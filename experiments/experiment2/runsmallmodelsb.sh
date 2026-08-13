#!/bin/bash
export HF_HOME=/data0/projects/Causal_and_Agentic_AI/hf_home
export TRANSFORMERS_CACHE=/data0/projects/Causal_and_Agentic_AI/hf_cache
export HF_DATASETS_CACHE=/data0/projects/Causal_and_Agentic_AI/hf_cache
export HF_METRICS_CACHE=/data0/projects/Causal_and_Agentic_AI/hf_cache
export CUDA_VISIBLE_DEVICES=6

DATA="data/Final2b.csv"

models=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-3.2-1B-Instruct"
)

for model in "${models[@]}"; do
    model_last="${model##*/}"
    outdir="exp2b/$model_last"
    logfile="$outdir/nohup.log"

    mkdir -p "$outdir"
    echo "Launching $model -> $logfile"

    nohup python3 exp2.py --model "$model" --data "$DATA" --exp_folder exp2b > "$logfile" 2>&1 &
    pid=$!
    wait $pid
done


