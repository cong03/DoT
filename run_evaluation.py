# Debate-of-Thoughts (DoT) - Main Evaluation Script
# Run DoT-Prompting, DoT-Tuning, or baseline methods on conflict QA benchmarks

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot.llm_engine import LLMEngine, GenerationConfig
from dot.data.dataset import load_dataset
from dot.prompting.dot_prompting import DoTPrompting
from dot.tuning.inference import DoTTuning
from dot.baselines.baselines import BASELINE_REGISTRY
from dot.evaluation.evaluator import (
    evaluate_batch, evaluate_single, save_results,
    EvalResult, EvalSummary, check_answer_match,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DoT: Debate-of-Thoughts for Knowledge Conflict Resolution"
    )

    # Model settings
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--engine_type", type=str, default="vllm",
                        choices=["vllm", "openai"],
                        help="LLM engine type")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)

    # Method settings
    parser.add_argument("--method", type=str, required=True,
                        choices=["dot_prompting", "dot_tuning"] +
                                list(BASELINE_REGISTRY.keys()),
                        help="Method to evaluate")
    parser.add_argument("--confidence_threshold", type=float, default=0.6,
                        help="Confidence threshold for uncertain output")

    # Data settings
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["faitheval", "musique", "squad"],
                        help="Dataset to evaluate on")
    parser.add_argument("--subtask", type=str, default=None,
                        choices=["counterfactual", "inconsistent", "unanswerable"],
                        help="Subtask for FaithEval")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")

    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize LLM engine
    print(f"Initializing {args.engine_type} engine with model: {args.model_name}")
    engine = LLMEngine(
        model_name=args.model_name,
        engine_type=args.engine_type,
        api_base=args.api_base,
        api_key=args.api_key,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}" +
          (f" ({args.subtask})" if args.subtask else ""))
    samples = load_dataset(
        args.dataset, args.data_dir, args.subtask, args.split
    )
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples")

    # Initialize method
    gen_config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.method == "dot_prompting":
        method = DoTPrompting(engine, gen_config, args.confidence_threshold)
        method_name = "DoT-Prompting"
    elif args.method == "dot_tuning":
        method = DoTTuning(engine, gen_config, args.confidence_threshold)
        method_name = "DoT-Tuning"
    elif args.method in BASELINE_REGISTRY:
        method = BASELINE_REGISTRY[args.method](engine, gen_config)
        method_name = args.method
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Run evaluation
    print(f"Running {method_name} on {len(samples)} samples...")

    if args.method in ("dot_prompting", "dot_tuning"):
        # DoT methods return DoTOutput objects
        outputs = method.run_batch(samples)
        results = [evaluate_single(out, s) for out, s in zip(outputs, samples)]
    else:
        # Baseline methods return answer strings
        predictions = method.run_batch(samples)
        results = []
        for pred, sample in zip(predictions, samples):
            is_correct = check_answer_match(pred, sample.answer, sample.choices)
            results.append(EvalResult(
                question=sample.question,
                gold_answer=sample.answer,
                predicted_answer=pred,
                is_correct=is_correct,
                is_uncertain=False,
                confidence=0.0,
                dataset=sample.dataset,
                subtask=sample.subtask,
            ))

    # Compute summary
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    uncertain = sum(1 for r in results if r.is_uncertain)
    non_uncertain = total - uncertain
    correct_non_uncertain = sum(
        1 for r in results if r.is_correct and not r.is_uncertain
    )

    summary = EvalSummary(
        dataset=args.dataset,
        subtask=args.subtask,
        total=total,
        correct=correct,
        uncertain_count=uncertain,
        accuracy=correct / total if total > 0 else 0.0,
        accuracy_excluding_uncertain=(
            correct_non_uncertain / non_uncertain if non_uncertain > 0 else 0.0
        ),
    )

    # Save results
    save_results(results, summary, args.output_dir, method_name)


if __name__ == "__main__":
    main()
