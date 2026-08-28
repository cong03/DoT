# Debate-of-Thoughts (DoT) - LLM Inference Engine
# Supports vLLM for local models and OpenAI-compatible APIs

import os
import json
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    n: int = 1  # Number of samples (for CoT-SC)


class LLMEngine:
    """Unified interface for LLM inference.

    Supports:
    - vLLM for local model serving
    - OpenAI-compatible API endpoints
    """

    def __init__(
        self,
        model_name: str,
        engine_type: str = "vllm",  # "vllm" or "openai"
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
    ):
        self.model_name = model_name
        self.engine_type = engine_type

        if engine_type == "vllm":
            self._init_vllm(model_name, tensor_parallel_size,
                            gpu_memory_utilization, max_model_len)
        elif engine_type == "openai":
            self._init_openai(api_base, api_key)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

    def _init_vllm(self, model_name, tensor_parallel_size,
                   gpu_memory_utilization, max_model_len):
        """Initialize vLLM engine."""
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams

    def _init_openai(self, api_base, api_key):
        """Initialize OpenAI-compatible API client."""
        import openai
        self.client = openai.OpenAI(
            base_url=api_base or os.environ.get("OPENAI_API_BASE"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from a single prompt.

        Args:
            prompt: Input prompt string.
            config: Generation configuration.

        Returns:
            Generated text string.
        """
        if config is None:
            config = GenerationConfig()

        if self.engine_type == "vllm":
            return self._generate_vllm(prompt, config)
        else:
            return self._generate_openai(prompt, config)

    def generate_batch(self, prompts: List[str],
                       config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate text for a batch of prompts."""
        if config is None:
            config = GenerationConfig()

        if self.engine_type == "vllm":
            return self._generate_vllm_batch(prompts, config)
        else:
            return [self._generate_openai(p, config) for p in prompts]

    def _generate_vllm(self, prompt: str, config: GenerationConfig) -> str:
        sampling_params = self.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
            n=config.n,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _generate_vllm_batch(self, prompts: List[str],
                              config: GenerationConfig) -> List[str]:
        sampling_params = self.SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
            n=config.n,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def _generate_openai(self, prompt: str, config: GenerationConfig) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
        )
        return response.choices[0].message.content.strip()
