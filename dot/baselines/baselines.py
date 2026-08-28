# Debate-of-Thoughts (DoT) - Baseline Methods
# Implements: No-Context, Full-Context, CoT, CoT-SC, Opin(Instr), KRE

import re
from typing import Optional, List
from collections import Counter

from ..llm_engine import LLMEngine, GenerationConfig
from ..data.dataset import DoTSample, format_choices_for_prompt


class BaselineMethod:
    """Base class for baseline methods."""

    def __init__(self, engine: LLMEngine, config: Optional[GenerationConfig] = None):
        self.engine = engine
        self.config = config or GenerationConfig(temperature=0.0, max_tokens=512)

    def run(self, sample: DoTSample) -> str:
        """Run the baseline method on a single sample. Returns predicted answer."""
        raise NotImplementedError

    def run_batch(self, samples: List[DoTSample]) -> List[str]:
        return [self.run(s) for s in samples]


class NoContextBaseline(BaselineMethod):
    """No-Context: Model answers using only parametric knowledge."""

    def run(self, sample: DoTSample) -> str:
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = f"""Answer the following question based on your knowledge.

Question: {question}

Answer:"""
        return self.engine.generate(prompt, self.config)


class FullContextBaseline(BaselineMethod):
    """Full-Context: Model answers using the provided context."""

    def run(self, sample: DoTSample) -> str:
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = f"""Answer the following question based on the given context.

Context: {sample.context}

Question: {question}

Answer:"""
        return self.engine.generate(prompt, self.config)


class CoTBaseline(BaselineMethod):
    """Chain-of-Thought (CoT) reasoning."""

    def run(self, sample: DoTSample) -> str:
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = f"""Answer the following question based on the given context. Think step by step before providing your final answer.

Context: {sample.context}

Question: {question}

Let's think step by step.

At the end, state your final answer in the format: "Final Answer: [answer]"

"""
        output = self.engine.generate(prompt, self.config)
        return self._extract_answer(output)

    def _extract_answer(self, output: str) -> str:
        match = re.search(r"Final\s+Answer:\s*(.+?)$", output, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # Fallback: return last line
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        return lines[-1] if lines else output


class CoTSCBaseline(BaselineMethod):
    """Chain-of-Thought with Self-Consistency (CoT-SC).

    Samples k=5 reasoning paths and selects answer by majority voting.
    """

    def __init__(self, engine: LLMEngine, config: Optional[GenerationConfig] = None,
                 k: int = 5):
        super().__init__(engine, config)
        self.k = k
        # Use higher temperature for diverse sampling
        self.sc_config = GenerationConfig(
            temperature=0.7,
            max_tokens=self.config.max_tokens,
            top_p=0.95,
        )

    def run(self, sample: DoTSample) -> str:
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = f"""Answer the following question based on the given context. Think step by step before providing your final answer.

Context: {sample.context}

Question: {question}

Let's think step by step.

At the end, state your final answer in the format: "Final Answer: [answer]"

"""
        # Sample k reasoning paths
        answers = []
        for _ in range(self.k):
            output = self.engine.generate(prompt, self.sc_config)
            answer = self._extract_answer(output)
            answers.append(answer)

        # Majority voting
        if answers:
            counter = Counter(answers)
            return counter.most_common(1)[0][0]
        return ""

    def _extract_answer(self, output: str) -> str:
        match = re.search(r"Final\s+Answer:\s*(.+?)$", output, re.MULTILINE)
        if match:
            return match.group(1).strip()
        lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
        return lines[-1] if lines else output


class OpinInstrBaseline(BaselineMethod):
    """Opinion-based Instruction (Opin(Instr)) baseline.

    Uses opinionated questioning to encourage model reliance on context.
    """

    def run(self, sample: DoTSample) -> str:
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = f"""Based on the following documents, what is your opinion on the answer to the question? Please rely on the information provided in the documents.

Documents: {sample.context}

Question: {question}

In my opinion, based on the documents, the answer is:"""
        return self.engine.generate(prompt, self.config)


class KREBaseline(BaselineMethod):
    """KRE (Knowledge conflict Resolution Evaluation) baseline.

    Tests whether the model trusts internal memory or external context.
    """

    def run(self, sample: DoTSample) -> str:
        question = sample.question
        choices_str = format_choices_for_prompt(sample)
        if choices_str:
            question = question + choices_str

        prompt = f"""You are given a question and some context documents. The context may contain conflicting information. Please carefully analyze the context and your own knowledge to provide the most accurate answer.

Context: {sample.context}

Question: {question}

Analyze the reliability of each source and provide your answer.

Answer:"""
        return self.engine.generate(prompt, self.config)


# Registry of all baseline methods
BASELINE_REGISTRY = {
    "no_context": NoContextBaseline,
    "full_context": FullContextBaseline,
    "cot": CoTBaseline,
    "cot_sc": CoTSCBaseline,
    "opin_instr": OpinInstrBaseline,
    "kre": KREBaseline,
}
