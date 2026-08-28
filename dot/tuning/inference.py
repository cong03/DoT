# Debate-of-Thoughts (DoT) - DoT-Tuning: Inference with Fine-tuned Model
# Uses the fine-tuned model for direct deliberation without multi-step prompting

import re
from typing import Optional, List
from dataclasses import dataclass

from ..llm_engine import LLMEngine, GenerationConfig
from ..data.dataset import DoTSample, format_choices_for_prompt
from ..prompting.dot_prompting import DoTOutput, Hypothesis, DebateRecord, AdjudicationResult
from ..prompts import DOT_TUNING_INFERENCE


class DoTTuning:
    """DoT-Tuning: Inference with fine-tuned model.

    The fine-tuned model internalizes the deliberation process and can
    produce structured DoT output in a single forward pass.
    """

    def __init__(
        self,
        engine: LLMEngine,
        generation_config: Optional[GenerationConfig] = None,
        confidence_threshold: float = 0.6,
    ):
        self.engine = engine
        self.config = generation_config or GenerationConfig(
            temperature=0.0, max_tokens=1024
        )
        self.confidence_threshold = confidence_threshold

    def run(self, sample: DoTSample) -> DoTOutput:
        """Run inference with the fine-tuned DoT model."""
        context = sample.context
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = DOT_TUNING_INFERENCE.format(
            context=context, question=question
        )

        output_text = self.engine.generate(prompt, self.config)

        # Parse the structured output
        return self._parse_output(output_text, sample)

    def run_batch(self, samples: List[DoTSample]) -> List[DoTOutput]:
        """Run batch inference."""
        prompts = []
        for sample in samples:
            question = sample.question
            choices_str = format_choices_for_prompt(sample)
            if choices_str:
                question = question + choices_str
            prompts.append(DOT_TUNING_INFERENCE.format(
                context=sample.context, question=question
            ))

        outputs = self.engine.generate_batch(prompts, self.config)

        return [self._parse_output(out, sample)
                for out, sample in zip(outputs, samples)]

    def _parse_output(self, output_text: str, sample: DoTSample) -> DoTOutput:
        """Parse the structured output from the fine-tuned model."""
        # Extract final answer
        final_answer = ""
        confidence = 0.0
        is_uncertain = False

        ans_match = re.search(
            r"Final\s+Answer:\s*(.+?)(?:\s*\(Confidence:\s*([\d.]+)%?\))?",
            output_text
        )
        if ans_match:
            final_answer = ans_match.group(1).strip()
            if ans_match.group(2):
                confidence = float(ans_match.group(2))
                if confidence > 1:
                    confidence /= 100.0

        if "uncertain" in final_answer.lower():
            is_uncertain = True
            final_answer = "uncertain"

        # Parse hypotheses
        hypotheses = []
        hyp_pattern = r"Hypothesis\s+([A-Z])\s*\(([^)]+)\)\s*:\s*(.+?)(?:\[Evidence:\s*(.+?)\])?$"
        for match in re.finditer(hyp_pattern, output_text, re.MULTILINE):
            hypotheses.append(Hypothesis(
                label=match.group(1),
                answer=match.group(2).strip(),
                description=match.group(3).strip(),
                evidence=match.group(4).strip() if match.group(4) else "",
            ))

        # Parse scores
        all_scores = {}
        score_pattern = r"Hypothesis\s+([A-Z]):\s*([\d.]+)"
        for match in re.finditer(score_pattern, output_text):
            all_scores[match.group(1)] = float(match.group(2))

        adjudication = AdjudicationResult(
            selected_hypothesis=None,
            selected_score=max(all_scores.values()) if all_scores else 0.0,
            all_scores=all_scores,
            justification="",
            final_answer=final_answer,
            confidence=confidence,
            is_uncertain=is_uncertain,
        )

        return DoTOutput(
            question=sample.question,
            conflicts="",
            hypotheses=hypotheses,
            debate_records=[],
            adjudication=adjudication,
            final_answer=final_answer,
            raw_outputs={"tuning_output": output_text},
        )
