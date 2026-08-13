# Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts

Repository for the paper:

**Sydney Anuyah, Sneha Shajee-Mohan, Ankit-Singh Chauhan, and Sunandan Chakraborty.**

*Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts.*

arXiv: [2601.15479v1](https://arxiv.org/abs/2601.15479)

The study evaluates 13 open-source large language models on pairwise causal discovery from text. It contains two tasks:

1. **Causal Detection** - identify whether text contains a factual causal relationship.
2. **Causal Extraction** - extract cause-effect pairs from causal text.

## Repository contents

```text
.
├── analysis/
├── experiments/
│   ├── experiment1/
│   └── experiment2/
├── paper/
│   ├── 2601.15479v1.pdf
│   └── source/
└── results/
    └── experiment2/
```

- `experiments/experiment1/` contains the causal-detection data, prompt definitions, inference script, and model outputs. The dataset file contains 3,008 rows.
- `experiments/experiment2/` contains the causal-extraction data, task slices 2a-2h, prompt definitions, inference scripts, and shell launch files. The consolidated dataset file contains 8,925 rows.
- `results/experiment2/` contains evaluated outputs and timing files grouped by model and prompting strategy.
- `analysis/` contains exact-match, similarity-scoring, problem-case, and error-bucket files.
- `paper/` contains the arXiv PDF and the available LaTeX source package.

## Dataset families

The experiment files contain examples from the following dataset families:

- CausalNet
- CausalProbe
- SemEval2010Task8
- MedCaus
- CauseNet
- FinCausal
- CausalBench
- CRASS
- Coling22
- ECI-B
- PubMed
- COPA

## Models

The paper evaluates:

- DeepSeek-R1-0528-Qwen3-8B
- DeepSeek-R1-Distill-Llama-70B
- DeepSeek-R1-Distill-Qwen-32B
- Llama-3-8B
- Llama-3.1-8B-Instruct
- Llama-3.2-3B
- Llama-3.2-3B-Instruct
- Meta-Llama-3.1-8B
- Meta-Llama-3.3-70B-Instruct
- Mistral-7B-Instruct-v0.3
- Mixtral-8x7B-Instruct-v0.1
- Qwen2.5-7B-Instruct
- Qwen2.5-Coder-32B-Instruct

## Prompting strategies

Experiment 1 includes:

- Instruction-only
- Few-shot in-context learning
- Chain-of-thought
- Hybrid chain-of-thought with few-shot in-context learning

Experiment 2 includes:

- General instruction-only / zero-shot
- Few-shot in-context learning
- Chain-of-thought
- Least-to-Most
- ReAct

## Experiment scripts

The Experiment 1 inference script accepts a CSV path, Hugging Face model name or path, batch size, and label column:

```bash
cd experiments/experiment1
python main.py \
  --data data/exp1.csv \
  --model_name MODEL_NAME_OR_PATH
```

The Experiment 2 inference script accepts a model, CSV path, batch size, top-k value, temperature, seed, maximum new-token count, and output folder:

```bash
cd experiments/experiment2
python exp2.py \
  --model MODEL_NAME_OR_PATH \
  --data data/Experiment2.csv \
  --exp_folder OUTPUT_FOLDER
```

The scripts use seed `4000`. The Experiment 1 script uses 4-bit model loading. The Experiment 2 script exposes defaults of batch size `64`, top-k `40`, temperature `0.7`, and maximum new-token count `256`.

## Contact

Sydney Anuyah - sanuyah@iu.edu
