# Debate-of-Thoughts (DoT) - DoT-Tuning: LoRA SFT Training Script
# Fine-tunes a base model to internalize the DoT deliberation process

import json
import os
import torch
from typing import Optional, Dict
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


@dataclass
class DoTTrainingConfig:
    """Configuration for DoT-Tuning training."""
    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
    # Training
    output_dir: str = "./output/dot_tuning"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 16  # global batch = 4 * 16 = 64
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    max_seq_length: int = 4096
    bf16: bool = True
    # Data
    train_data_path: str = "./data/sft_train.jsonl"


def load_sft_dataset(data_path: str, tokenizer, max_length: int) -> Dataset:
    """Load and tokenize SFT dataset."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    def tokenize_function(example):
        # Format as instruction-following
        prompt = (
            "You are a rigorous research analyst. Analyze the following information, "
            "identify knowledge conflicts, and resolve them through structured internal "
            "deliberation.\n\n"
            f"{example['input']}\n\n"
            "Please perform the following steps:\n"
            "1. Identify knowledge conflicts in the given information\n"
            "2. Generate competing hypotheses from different perspectives\n"
            "3. For each hypothesis, provide supporting and opposing arguments\n"
            "4. Score each hypothesis on evidential support, logical consistency, "
            "and source reliability\n"
            "5. Select the best hypothesis or output \"uncertain\" if confidence is low\n\n"
        )

        full_text = prompt + example["output"]

        # Tokenize
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=max_length, add_special_tokens=True
        )["input_ids"]
        full_ids = tokenizer(
            full_text, truncation=True, max_length=max_length, add_special_tokens=True
        )["input_ids"]

        # Labels: mask prompt tokens with -100
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    return dataset


def train(config: DoTTrainingConfig):
    """Run DoT-Tuning LoRA training."""
    print(f"Loading model: {config.model_name_or_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    target_modules = config.lora_target_modules.split(",")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    train_dataset = load_sft_dataset(
        config.train_data_path, tokenizer, config.max_seq_length
    )
    print(f"Training samples: {len(train_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DoT-Tuning: LoRA SFT Training")
    parser.add_argument("--model_name_or_path", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/dot_tuning")
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=4096)

    args = parser.parse_args()

    config = DoTTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
    )

    train(config)
