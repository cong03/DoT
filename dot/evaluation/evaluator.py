# Debate-of-Thoughts (DoT) - Evaluation Module

import re
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from ..data.dataset import DoTSample
from ..prompting.dot_prompting import DoTOutput


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    question: str
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    is_uncertain: bool
    confidence: float
    dataset: str
    subtask: Optional[str] = None


@dataclass
class EvalSummary:
    """Summary of evaluation results."""
    dataset: str
    subtask: Optional[str]
    total: int
    correct: int
    uncertain_count: int
    accuracy: float
    accuracy_excluding_uncertain: float


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if not answer:
        return ""
    # Remove extra whitespace, lowercase
    answer = answer.strip().lower()
    # Remove trailing punctuation
    answer = re.sub(r'[.,;:!?]+$', '', answer)
    # Remove articles
    answer = re.sub(r'^(the|a|an)\s+', '', answer)
    return answer


def check_answer_match(predicted: str, gold: str, choices: Optional[List[str]] = None) -> bool:
    """Check if predicted answer matches gold answer.

    Supports:
    - Exact match (normalized)
    - Choice matching for multiple-choice questions
    - Partial containment for short answers
    """
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if not pred_norm or not gold_norm:
        return False

    # Exact match
    if pred_norm == gold_norm:
        return True

    # For multiple-choice: check if the predicted answer matches any choice
    if choices:
        for choice in choices:
            choice_norm = normalize_answer(choice)
            if pred_norm == choice_norm and gold_norm == choice_norm:
                return True
            # Check if predicted contains the choice text
            if choice_norm in pred_norm and gold_norm == choice_norm:
                return True

    # Containment match (for short answers)
    if len(gold_norm) > 2:
        if gold_norm in pred_norm or pred_norm in gold_norm:
            return True

    return False


def evaluate_single(output: DoTOutput, sample: DoTSample) -> EvalResult:
    """Evaluate a single DoT output against ground truth."""
    is_correct = check_answer_match(
        output.final_answer, sample.answer, sample.choices
    )

    return EvalResult(
        question=sample.question,
        gold_answer=sample.answer,
        predicted_answer=output.final_answer,
        is_correct=is_correct,
        is_uncertain=output.adjudication.is_uncertain,
        confidence=output.adjudication.confidence,
        dataset=sample.dataset,
        subtask=sample.subtask,
    )


def evaluate_batch(outputs: List[DoTOutput], samples: List[DoTSample],
                   dataset_name: str, subtask: Optional[str] = None) -> EvalSummary:
    """Evaluate a batch of outputs."""
    results = []
    for output, sample in zip(outputs, samples):
        results.append(evaluate_single(output, sample))

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    uncertain = sum(1 for r in results if r.is_uncertain)
    non_uncertain = total - uncertain
    correct_non_uncertain = sum(1 for r in results if r.is_correct and not r.is_uncertain)

    return EvalSummary(
        dataset=dataset_name,
        subtask=subtask,
        total=total,
        correct=correct,
        uncertain_count=uncertain,
        accuracy=correct / total if total > 0 else 0.0,
        accuracy_excluding_uncertain=(
            correct_non_uncertain / non_uncertain if non_uncertain > 0 else 0.0
        ),
    )


def save_results(results: List[EvalResult], summary: EvalSummary,
                 output_dir: str, method_name: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(output_dir, f"{method_name}_results.jsonl")
    with open(results_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Save summary
    summary_file = os.path.join(output_dir, f"{method_name}_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    print(f"\n=== Results for {method_name} ===")
    print(f"Dataset: {summary.dataset}" +
          (f" ({summary.subtask})" if summary.subtask else ""))
    print(f"Total: {summary.total}, Correct: {summary.correct}, "
          f"Uncertain: {summary.uncertain_count}")
    print(f"Accuracy: {summary.accuracy:.4f} ({summary.accuracy*100:.1f}%)")
    if summary.uncertain_count > 0:
        print(f"Accuracy (excl. uncertain): "
              f"{summary.accuracy_excluding_uncertain:.4f} "
              f"({summary.accuracy_excluding_uncertain*100:.1f}%)")
