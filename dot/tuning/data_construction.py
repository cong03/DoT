# Debate-of-Thoughts (DoT) - DoT-Tuning: SFT Data Construction
# Uses DoT-Prompting with a teacher model to generate training data

import json
import os
from typing import List, Optional
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from ..llm_engine import LLMEngine, GenerationConfig
from ..data.dataset import DoTSample, load_dataset
from ..prompting.dot_prompting import DoTPrompting, DoTOutput
from ..evaluation.evaluator import check_answer_match


@dataclass
class SFTSample:
    """A single SFT training sample."""
    input_text: str   # Query + context
    output_text: str  # Complete DoT deliberation process
    dataset: str


def construct_sft_data(
    teacher_engine: LLMEngine,
    samples: List[DoTSample],
    max_samples: Optional[int] = None,
) -> List[SFTSample]:
    """Construct SFT training data using DoT-Prompting with a teacher model.

    Only trajectories leading to correct final answers are kept as positive examples.

    Args:
        teacher_engine: LLM engine for the teacher model (e.g., Qwen3-72B, GPT-4).
        samples: List of training samples.
        max_samples: Maximum number of samples to process.

    Returns:
        List of SFTSample objects for training.
    """
    dot = DoTPrompting(teacher_engine)
    sft_samples = []

    if max_samples:
        samples = samples[:max_samples]

    for sample in tqdm(samples, desc="Constructing SFT data"):
        try:
            output = dot.run(sample)

            # Only keep trajectories with correct final answers
            if check_answer_match(output.final_answer, sample.answer, sample.choices):
                input_text = _format_sft_input(sample)
                output_text = _format_sft_output(output)

                sft_samples.append(SFTSample(
                    input_text=input_text,
                    output_text=output_text,
                    dataset=sample.dataset,
                ))
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    print(f"Constructed {len(sft_samples)} SFT samples from {len(samples)} inputs "
          f"({len(sft_samples)/max(len(samples),1)*100:.1f}% retention rate)")
    return sft_samples


def _format_sft_input(sample: DoTSample) -> str:
    """Format input for SFT training."""
    parts = [
        f"**Context**: {sample.context}",
        f"**Question**: {sample.question}",
    ]
    if sample.choices:
        parts.append(f"**Choices**: {', '.join(sample.choices)}")
    return "\n".join(parts)


def _format_sft_output(output: DoTOutput) -> str:
    """Format the complete DoT deliberation process as SFT target output."""
    parts = []

    # Step 1: Conflicts
    parts.append("Step 1 - Conflicts:")
    parts.append(output.conflicts)

    # Step 2: Hypotheses
    parts.append("\nStep 2 - Hypotheses:")
    for hyp in output.hypotheses:
        parts.append(
            f"Hypothesis {hyp.label} ({hyp.answer}): {hyp.description} "
            f"[Evidence: {hyp.evidence}]"
        )

    # Step 3: Debate
    parts.append("\nStep 3 - Debate:")
    for record in output.debate_records:
        parts.append(f"Hypothesis {record.hypothesis.label} Analysis:")
        parts.append(f"  Supporting: {record.supporting}")
        parts.append(f"  Opposing: {record.opposing}")

    # Step 4: Scores
    parts.append("\nStep 4 - Scores:")
    for label, score in output.adjudication.all_scores.items():
        parts.append(f"  Hypothesis {label}: {score:.2f}")

    # Step 5: Final Answer
    parts.append(f"\nStep 5 - Final Answer: {output.final_answer} "
                 f"(Confidence: {output.adjudication.confidence*100:.0f}%)")

    return "\n".join(parts)


def save_sft_data(sft_samples: List[SFTSample], output_path: str):
    """Save SFT data in JSONL format for training."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            entry = {
                "input": sample.input_text,
                "output": sample.output_text,
                "dataset": sample.dataset,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(sft_samples)} SFT samples to {output_path}")
