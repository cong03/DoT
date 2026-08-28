# Debate-of-Thoughts (DoT) - Shell Scripts for Reproducing Experiments

## Quick Start Scripts

### 1. DoT-Prompting Evaluation

```bash
# Evaluate DoT-Prompting on FaithEval (all subtasks)
for subtask in counterfactual inconsistent unanswerable; do
    python run_evaluation.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --method dot_prompting \
        --dataset faitheval \
        --subtask $subtask \
        --data_dir ./data \
        --output_dir ./results \
        --tensor_parallel_size 1
done

# Evaluate DoT-Prompting on MuSiQue
python run_evaluation.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --method dot_prompting \
    --dataset musique \
    --data_dir ./data \
    --output_dir ./results

# Evaluate DoT-Prompting on SQuAD
python run_evaluation.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --method dot_prompting \
    --dataset squad \
    --data_dir ./data \
    --output_dir ./results
```

### 2. DoT-Tuning (Two-Step Process)

```bash
# Step 1: Construct SFT data using a teacher model
python run_data_construction.py \
    --teacher_model Qwen/Qwen3-72B-Instruct \
    --tensor_parallel_size 4 \
    --data_dir ./data \
    --output_path ./data/sft_train.jsonl

# Step 2: Train with LoRA
python -m dot.tuning.train \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --train_data_path ./data/sft_train.jsonl \
    --output_dir ./output/dot_tuning_llama \
    --num_train_epochs 4 \
    --learning_rate 5e-4

# Step 3: Evaluate DoT-Tuning
python run_evaluation.py \
    --model_name ./output/dot_tuning_llama \
    --method dot_tuning \
    --dataset faitheval \
    --subtask counterfactual \
    --data_dir ./data \
    --output_dir ./results
```

### 3. Baseline Evaluations

```bash
# Run all baselines on all datasets
for method in no_context full_context cot cot_sc opin_instr kre; do
    for dataset in faitheval musique squad; do
        if [ "$dataset" == "faitheval" ]; then
            for subtask in counterfactual inconsistent unanswerable; do
                python run_evaluation.py \
                    --model_name meta-llama/Llama-3.1-8B-Instruct \
                    --method $method \
                    --dataset $dataset \
                    --subtask $subtask \
                    --data_dir ./data \
                    --output_dir ./results
            done
        else
            python run_evaluation.py \
                --model_name meta-llama/Llama-3.1-8B-Instruct \
                --method $method \
                --dataset $dataset \
                --data_dir ./data \
                --output_dir ./results
        fi
    done
done
```
