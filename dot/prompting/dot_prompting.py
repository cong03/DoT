# Debate-of-Thoughts (DoT) - DoT-Prompting Pipeline
# Three-phase prompt chaining: Hypothesis Generation -> Adversarial Debate -> Evidential Adjudication

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..llm_engine import LLMEngine, GenerationConfig
from ..data.dataset import DoTSample, format_choices_for_prompt
from ..prompts import (
    HYPOTHESIS_GENERATION,
    ADVERSARIAL_DEBATE,
    EVIDENTIAL_ADJUDICATION,
)


@dataclass
class Hypothesis:
    """A single hypothesis generated in Phase 1."""
    label: str  # e.g., "A", "B", "C"
    answer: str
    description: str
    evidence: str


@dataclass
class DebateRecord:
    """Debate record for a single hypothesis."""
    hypothesis: Hypothesis
    supporting: str  # Proponent arguments
    opposing: str  # Critic arguments


@dataclass
class AdjudicationResult:
    """Result from the Judge's adjudication."""
    selected_hypothesis: Optional[str]  # Label of selected hypothesis
    selected_score: float
    all_scores: Dict[str, float]
    justification: str
    final_answer: str
    confidence: float
    is_uncertain: bool = False


@dataclass
class DoTOutput:
    """Complete output of the DoT pipeline."""
    question: str
    conflicts: str
    hypotheses: List[Hypothesis]
    debate_records: List[DebateRecord]
    adjudication: AdjudicationResult
    final_answer: str
    raw_outputs: Dict[str, str] = field(default_factory=dict)


class DoTPrompting:
    """DoT-Prompting: Three-phase prompt chaining implementation.

    Phase 1: Multi-Hypothesis Generation (M_gen)
    Phase 2: Multi-Role Adversarial Internal Debate (Proponent + Critic)
    Phase 3: Evidential Adjudication (Judge)
    """

    def __init__(
        self,
        engine: LLMEngine,
        generation_config: Optional[GenerationConfig] = None,
        confidence_threshold: float = 0.6,
    ):
        self.engine = engine
        self.config = generation_config or GenerationConfig(
            temperature=0.0, max_tokens=2048
        )
        self.confidence_threshold = confidence_threshold

    def run(self, sample: DoTSample) -> DoTOutput:
        """Execute the full DoT pipeline on a single sample.

        Args:
            sample: Input DoTSample with question and context.

        Returns:
            DoTOutput containing all intermediate results and final answer.
        """
        context = sample.context
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        # Phase 1: Multi-Hypothesis Generation
        phase1_output = self._phase1_generate(question, context)
        conflicts, hypotheses = self._parse_hypotheses(phase1_output)

        # Phase 2: Adversarial Debate
        phase2_output = self._phase2_debate(question, context, conflicts, hypotheses)
        debate_records = self._parse_debate(phase2_output, hypotheses)

        # Phase 3: Evidential Adjudication
        phase3_output = self._phase3_adjudicate(
            question, context, conflicts, phase2_output
        )
        adjudication = self._parse_adjudication(phase3_output, hypotheses)

        final_answer = adjudication.final_answer

        return DoTOutput(
            question=sample.question,
            conflicts=conflicts,
            hypotheses=hypotheses,
            debate_records=debate_records,
            adjudication=adjudication,
            final_answer=final_answer,
            raw_outputs={
                "phase1": phase1_output,
                "phase2": phase2_output,
                "phase3": phase3_output,
            },
        )

    def run_batch(self, samples: List[DoTSample]) -> List[DoTOutput]:
        """Run DoT pipeline on a batch of samples."""
        return [self.run(sample) for sample in samples]

    def _phase1_generate(self, question: str, context: str) -> str:
        """Phase 1: Multi-Hypothesis Generation."""
        prompt = HYPOTHESIS_GENERATION.format(
            context=context, question=question
        )
        return self.engine.generate(prompt, self.config)

    def _phase2_debate(
        self, question: str, context: str,
        conflicts: str, hypotheses: List[Hypothesis]
    ) -> str:
        """Phase 2: Multi-Role Adversarial Internal Debate."""
        hypotheses_str = self._format_hypotheses(hypotheses)
        prompt = ADVERSARIAL_DEBATE.format(
            context=context,
            question=question,
            conflicts=conflicts,
            hypotheses=hypotheses_str,
        )
        return self.engine.generate(prompt, self.config)

    def _phase3_adjudicate(
        self, question: str, context: str,
        conflicts: str, debate_output: str
    ) -> str:
        """Phase 3: Evidential Adjudication."""
        prompt = EVIDENTIAL_ADJUDICATION.format(
            context=context,
            question=question,
            conflicts=conflicts,
            debate_records=debate_output,
        )
        return self.engine.generate(prompt, self.config)

    def _parse_hypotheses(self, output: str) -> Tuple[str, List[Hypothesis]]:
        """Parse Phase 1 output into conflicts and hypotheses."""
        lines = output.strip().split("\n")
        conflicts = ""
        hypotheses = []
        in_conflicts = False
        in_hypotheses = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "conflict" in line.lower() and not in_hypotheses:
                in_conflicts = True
                if ":" in line:
                    conflicts += line.split(":", 1)[1].strip() + "\n"
                continue

            # Match hypothesis patterns like "Hypothesis A (answer): description [Evidence: source]"
            match = re.match(
                r"Hypothesis\s+([A-Z])\s*\(([^)]+)\)\s*:\s*(.+?)(?:\[Evidence:\s*(.+?)\])?$",
                line, re.IGNORECASE
            )
            if match:
                in_hypotheses = True
                in_conflicts = False
                hypotheses.append(Hypothesis(
                    label=match.group(1),
                    answer=match.group(2).strip(),
                    description=match.group(3).strip(),
                    evidence=match.group(4).strip() if match.group(4) else "",
                ))
            elif in_conflicts:
                conflicts += line + "\n"

        # Fallback: if no structured hypotheses found, try to extract from raw text
        if not hypotheses:
            hypotheses = self._fallback_parse_hypotheses(output)

        return conflicts.strip(), hypotheses

    def _fallback_parse_hypotheses(self, output: str) -> List[Hypothesis]:
        """Fallback parser for hypotheses when structured parsing fails."""
        hypotheses = []
        # Try to find any pattern like "A (answer)" or "Hypothesis A"
        pattern = r"(?:Hypothesis\s+)?([A-Z])\s*[\(:\s]\s*(.+?)[\):\s]\s*(.+?)(?=(?:Hypothesis\s+)?[A-Z]\s*[\(:\s]|$)"
        matches = re.findall(pattern, output, re.DOTALL)
        for i, (label, answer, desc) in enumerate(matches[:4]):
            hypotheses.append(Hypothesis(
                label=label.strip(),
                answer=answer.strip(),
                description=desc.strip()[:200],
                evidence="",
            ))
        return hypotheses

    def _parse_debate(
        self, output: str, hypotheses: List[Hypothesis]
    ) -> List[DebateRecord]:
        """Parse Phase 2 output into debate records."""
        records = []
        for hyp in hypotheses:
            supporting = ""
            opposing = ""

            # Find the section for this hypothesis
            pattern = rf"Hypothesis\s+{hyp.label}.*?Analysis.*?Supporting:(.*?)(?:Opposing:(.*?))?(?=Hypothesis\s+[A-Z]|$)"
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                supporting = match.group(1).strip() if match.group(1) else ""
                opposing = match.group(2).strip() if match.group(2) else ""

            records.append(DebateRecord(
                hypothesis=hyp,
                supporting=supporting,
                opposing=opposing,
            ))
        return records

    def _parse_adjudication(
        self, output: str, hypotheses: List[Hypothesis]
    ) -> AdjudicationResult:
        """Parse Phase 3 output into adjudication result."""
        selected = None
        selected_score = 0.0
        all_scores = {}
        justification = ""
        final_answer = ""
        confidence = 0.0
        is_uncertain = False

        # Parse selected hypothesis
        sel_match = re.search(
            r"Selected\s+Hypothesis:\s*([A-Z])\s*\(Score:\s*([\d.]+)", output
        )
        if sel_match:
            selected = sel_match.group(1)
            selected_score = float(sel_match.group(2))

        # Parse all scores
        score_pattern = r"([A-Z])\s*\(Score:\s*([\d.]+)"
        for match in re.finditer(score_pattern, output):
            all_scores[match.group(1)] = float(match.group(2))

        # Parse justification
        just_match = re.search(r"Justification:\s*(.+?)(?=Final\s+Answer|$)", output, re.DOTALL)
        if just_match:
            justification = just_match.group(1).strip()

        # Parse final answer
        ans_match = re.search(r"Final\s+Answer:\s*(.+?)(?:\s*\(Confidence:\s*([\d.]+)%?\))?", output)
        if ans_match:
            final_answer = ans_match.group(1).strip()
            if ans_match.group(2):
                confidence = float(ans_match.group(2))
                if confidence > 1:
                    confidence /= 100.0

        # Check for uncertain output
        if "uncertain" in output.lower() or selected_score < self.confidence_threshold:
            is_uncertain = True
            if not final_answer or final_answer.lower() == "uncertain":
                final_answer = "uncertain"

        # Map selected hypothesis to answer
        if selected and not final_answer:
            for hyp in hypotheses:
                if hyp.label == selected:
                    final_answer = hyp.answer
                    break

        return AdjudicationResult(
            selected_hypothesis=selected,
            selected_score=selected_score,
            all_scores=all_scores,
            justification=justification,
            final_answer=final_answer,
            confidence=confidence,
            is_uncertain=is_uncertain,
        )

    def _format_hypotheses(self, hypotheses: List[Hypothesis]) -> str:
        """Format hypotheses for inclusion in prompts."""
        parts = []
        for hyp in hypotheses:
            parts.append(
                f"Hypothesis {hyp.label} ({hyp.answer}): {hyp.description} "
                f"[Evidence: {hyp.evidence}]"
            )
        return "\n".join(parts)
