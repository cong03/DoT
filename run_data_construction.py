# Debate-of-Thoughts (DoT) - SFT Data Construction Script
# Uses a teacher model (e.g., Qwen3-72B, GPT-4) to generate DoT training data

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot.llm_engine import LLMEngine, GenerationConfig
from dot.data.dataset import load_dataset
from dot.tuning.data_construction import construct_sft_data, save_sft_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct SFT training data using DoT-Prompting with a teacher model"
    )
    parser.add_argument("--teacher_model", type=str, required=True,
                        help="Teacher model name (e.g., Qwen/Qwen3-72B-Instruct)")
    parser.add_argument("--engine_type", type=str, default="vllm",
                        choices=["vllm", "openai"])
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./data/sft_train.jsonl")
    parser.add_argument("--max_samples_per_dataset", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["faitheval", "musique", "squad"],
                        help="Datasets to use for training data construction")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize teacher model
    print(f"Initializing teacher model: {args.teacher_model}")
    engine = LLMEngine(
        model_name=args.teacher_model,
        engine_type=args.engine_type,
        api_base=args.api_base,
        api_key=args.api_key,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    all_sft_samples = []

    for dataset_name in args.datasets:
        print(f"\nProcessing dataset: {dataset_name}")

        if dataset_name == "faitheval":
            # Load all FaithEval subtasks
            for subtask in ["counterfactual", "inconsistent", "unanswerable"]:
                print(f"  Loading FaithEval/{subtask} train split...")
                samples = load_dataset(dataset_name, args.data_dir,
                                       subtask, split="train")
                sft_samples = construct_sft_data(
                    engine, samples, args.max_samples_per_dataset
                )
                all_sft_samples.extend(sft_samples)
        else:
            samples = load_dataset(dataset_name, args.data_dir, split="train")
            sft_samples = construct_sft_data(
                engine, samples, args.max_samples_per_dataset
            )
            all_sft_samples.extend(sft_samples)

    # Save combined SFT data
    save_sft_data(all_sft_samples, args.output_path)
    print(f"\nTotal SFT samples: {len(all_sft_samples)}")


if __name__ == "__main__":
    main()
