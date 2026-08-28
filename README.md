# Debate-of-Thoughts: Resolving Knowledge Conflicts in LLMs Through Internal Deliberation

Official implementation of the ACL 2026 paper: **"Debate-of-Thoughts: Resolving Knowledge Conflicts in LLMs Through Internal Deliberation"**

## Overview

Debate-of-Thoughts (DoT) is a framework that transforms knowledge conflict resolution into an active deliberation process. Instead of forcing models into a binary choice between context and memory, DoT guides a single model through three phases:

1. **Multi-Hypothesis Generation**: Forms competing perspectives from conflicting sources
2. **Multi-Role Adversarial Internal Debate**: The model acts as both Proponent and Critic to stress-test each hypothesis
3. **Evidential Adjudication**: The model acts as a Judge to evaluate arguments based on evidential support, logical consistency, and source reliability

## Framework

```
User Query + Conflicting Contexts
        |
        v
[Phase 1] Multi-Hypothesis Generation
        |  Generate H1, H2, ..., Hn from different perspectives
        v
[Phase 2] Adversarial Internal Debate
        |  For each Hi: Proponent argues FOR, Critic argues AGAINST
        v
[Phase 3] Evidential Adjudication
        |  Judge scores each hypothesis on:
        |  - Evidential Support (S_evid)
        |  - Logical Consistency (S_logic)
        |  - Source Reliability (S_source)
        v
    Final Answer (or "Uncertain" if all scores < 0.6)
```

## Installation

```bash
git clone https://github.com/cong03/DoT.git
cd DoT
pip install -r requirements.txt
```

## Data Preparation

Download and organize the datasets as follows:

```
data/
├── faitheval/
│   ├── counterfactual/
│   │   ├── train.json
│   │   └── test.json
│   ├── inconsistent/
│   │   ├── train.json
│   │   └── test.json
│   └── unanswerable/
│       ├── train.json
│       └── test.json
├── musique/
│   ├── train.json
│   └── test.json
└── squad/
    ├── train.json
    └── test.json
```

Each data file should contain a JSON array with objects having the fields:
- `question`: The question string
- `context`: Combined context (for main experiments, concatenation of negative and golden contexts)
- `golden_context`: The original correct context
- `negative_context`: The context with fabricated/conflicting information
- `choices` (optional): List of answer choices for multiple-choice questions
- `answer`: The ground truth answer

## Usage

### DoT-Prompting (Inference-Time Prompt Chaining)

No training required. Directly applicable to any LLM:

```bash
python run_evaluation.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --method dot_prompting \
    --dataset faitheval \
    --subtask counterfactual \
    --data_dir ./data \
    --output_dir ./results \
    --tensor_parallel_size 1
```

### DoT-Tuning (Supervised Fine-Tuning)

**Step 1**: Construct training data using a teacher model:

```bash
python run_data_construction.py \
    --teacher_model Qwen/Qwen3-72B-Instruct \
    --tensor_parallel_size 4 \
    --data_dir ./data \
    --output_path ./data/sft_train.jsonl
```

**Step 2**: Fine-tune with LoRA:

```bash
python -m dot.tuning.train \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --train_data_path ./data/sft_train.jsonl \
    --output_dir ./output/dot_tuning \
    --num_train_epochs 4 \
    --learning_rate 5e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

**Step 3**: Evaluate the fine-tuned model:

```bash
python run_evaluation.py \
    --model_name ./output/dot_tuning \
    --method dot_tuning \
    --dataset faitheval \
    --subtask counterfactual \
    --data_dir ./data \
    --output_dir ./results
```

### Baseline Methods

```bash
# Available baselines: no_context, full_context, cot, cot_sc, opin_instr, kre
python run_evaluation.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --method cot \
    --dataset musique \
    --data_dir ./data \
    --output_dir ./results
```

## Supported Models

- Llama-3.1-8B-Instruct
- Qwen3-8B
- Qwen3-14B
- Any model supported by vLLM or OpenAI-compatible APIs

## Supported Datasets

| Dataset | Type | Subtasks |
|---------|------|----------|
| FaithEval | Context-Memory / Context-Context / Boundary | Counterfactual, Inconsistent, Unanswerable |
| MuSiQue | Context-Context (KRE-based) | - |
| SQuAD | Context-Context (KRE-based) | - |

## Key Results

| Method | FaithEval (Unans.) | FaithEval (Incons.) | FaithEval (Counter.) | MuSiQue | SQuAD |
|--------|--------------------|--------------------|---------------------|---------|-------|
| DoT-Prompting (Llama-3.1-8B) | 58.2 | 82.3 | 64.9 | 80.4 | 82.8 |
| DoT-Tuning (Llama-3.1-8B) | **64.7** | **86.9** | **69.2** | **83.6** | **87.9** |
| DoT-Prompting (Qwen3-8B) | 79.4 | 87.0 | 76.3 | 72.8 | 81.9 |
| DoT-Tuning (Qwen3-8B) | **82.8** | **89.1** | **79.2** | **75.6** | **85.3** |

## Project Structure

```
DoT/
├── dot/
│   ├── prompting/
│   │   └── dot_prompting.py      # DoT-Prompting pipeline
│   ├── tuning/
│   │   ├── data_construction.py   # SFT data construction
│   │   ├── train.py               # LoRA training script
│   │   └── inference.py           # DoT-Tuning inference
│   ├── data/
│   │   └── dataset.py             # Data loading utilities
│   ├── evaluation/
│   │   └── evaluator.py           # Evaluation metrics
│   ├── baselines/
│   │   └── baselines.py           # Baseline implementations
│   └── llm_engine.py              # Unified LLM inference engine
├── prompts/
│   └── dot_prompts.py             # Prompt templates (Figures 4-6)
├── configs/                        # Configuration files
├── scripts/
│   └── run_experiments.md         # Shell scripts for all experiments
├── run_evaluation.py              # Main evaluation entry point
├── run_data_construction.py       # SFT data construction entry point
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{li2026debate,
  title={Debate-of-Thoughts: Resolving Knowledge Conflicts in LLMs Through Internal Deliberation},
  author={Li, Guocong and Hu, Qirui and Wang, Ping and Zhang, Guofeng and Wu, Jian and Xu, Hongxia},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={35674--35696},
  year={2026}
}
```

## License

This project is released under the MIT License.
