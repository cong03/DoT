# Debate-of-Thoughts (DoT) - Data Loading and Preprocessing
# Supports: FaithEval, MuSiQue (KRE-based), SQuAD (KRE-based)

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DoTSample:
    """A single sample for DoT evaluation."""
    question: str
    context: str  # Combined context (negative + golden)
    golden_context: str
    negative_context: str
    choices: Optional[List[str]]  # For multiple-choice datasets
    answer: str
    dataset: str
    subtask: Optional[str] = None  # e.g., "counterfactual", "inconsistent", "unanswerable"


def load_faitheval(data_dir: str, subtask: str, split: str = "test") -> List[DoTSample]:
    """Load FaithEval dataset.

    Args:
        data_dir: Root directory containing FaithEval data files.
        subtask: One of "counterfactual", "inconsistent", "unanswerable".
        split: "train" or "test".

    Returns:
        List of DoTSample objects.
    """
    filepath = os.path.join(data_dir, "faitheval", subtask, f"{split}.json")
    samples = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # FaithEval format: question, context (with [DOC][TLE][PAR] markers), choices, answer
        context = item.get("context", "")
        question = item["question"]
        choices = item.get("choices", None)
        answer = item["answer"]

        sample = DoTSample(
            question=question,
            context=context,
            golden_context=item.get("golden_context", ""),
            negative_context=item.get("negative_context", context),
            choices=choices,
            answer=answer,
            dataset="faitheval",
            subtask=subtask,
        )
        samples.append(sample)

    return samples


def load_musique(data_dir: str, split: str = "test") -> List[DoTSample]:
    """Load MuSiQue dataset (KRE-based construction).

    The KRE-based MuSiQue contains fact-level knowledge conflicts
    with negative and golden contexts.
    """
    filepath = os.path.join(data_dir, "musique", f"{split}.json")
    samples = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Combine negative and golden contexts for the main experiment
        negative_ctx = item.get("negative_context", "")
        golden_ctx = item.get("golden_context", "")
        combined = _combine_contexts(negative_ctx, golden_ctx)

        sample = DoTSample(
            question=item["question"],
            context=combined,
            golden_context=golden_ctx,
            negative_context=negative_ctx,
            choices=item.get("choices", None),
            answer=item["answer"],
            dataset="musique",
        )
        samples.append(sample)

    return samples


def load_squad(data_dir: str, split: str = "test") -> List[DoTSample]:
    """Load SQuAD dataset (KRE-based construction)."""
    filepath = os.path.join(data_dir, "squad", f"{split}.json")
    samples = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        negative_ctx = item.get("negative_context", "")
        golden_ctx = item.get("golden_context", "")
        combined = _combine_contexts(negative_ctx, golden_ctx)

        sample = DoTSample(
            question=item["question"],
            context=combined,
            golden_context=golden_ctx,
            negative_context=negative_ctx,
            choices=item.get("choices", None),
            answer=item["answer"],
            dataset="squad",
        )
        samples.append(sample)

    return samples


def _combine_contexts(negative: str, golden: str) -> str:
    """Combine negative and golden contexts for the main conflict experiment."""
    parts = []
    if negative:
        parts.append(negative)
    if golden:
        parts.append(golden)
    return "\n\n".join(parts)


def load_dataset(dataset_name: str, data_dir: str, subtask: Optional[str] = None,
                 split: str = "test") -> List[DoTSample]:
    """Unified dataset loading interface.

    Args:
        dataset_name: One of "faitheval", "musique", "squad".
        data_dir: Root data directory.
        subtask: Subtask name for FaithEval (counterfactual/inconsistent/unanswerable).
        split: "train" or "test".

    Returns:
        List of DoTSample objects.
    """
    loaders = {
        "faitheval": lambda: load_faitheval(data_dir, subtask, split),
        "musique": lambda: load_musique(data_dir, split),
        "squad": lambda: load_squad(data_dir, split),
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(loaders.keys())}")

    return loaders[dataset_name]()


def format_context_for_prompt(sample: DoTSample) -> str:
    """Format the context for inclusion in prompts."""
    return sample.context


def format_choices_for_prompt(sample: DoTSample) -> str:
    """Format choices for multiple-choice questions."""
    if sample.choices:
        return "\nChoices: " + ", ".join(f'"{c}"' for c in sample.choices)
    return ""
