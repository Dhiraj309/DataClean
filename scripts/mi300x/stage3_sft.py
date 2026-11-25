#!/usr/bin/env python3
"""Stage 3 MI300X SFT on verified HuggingFaceTB/smoltalk2 SFT no_think splits."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from trainlm import (
    TrainLMFineTuningConfig,
    TrainLMTrainingArguments,
    run_finetuning,
)


# Verified against HuggingFaceTB/smoltalk2 config="SFT" via datasets-server API.
VERIFIED_SMOLTALK2_SFT_NO_THINK_SPLITS = (
    "smoltalk_smollm3_everyday_conversations_no_think",
    "smoltalk_smollm3_systemchats_30k_no_think",
    "smoltalk_smollm3_smol_magpie_ultra_no_think",
    "smoltalk_smollm3_smol_summarize_no_think",
    "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_explore_instruct_rewriting_no_think",
    "OpenHermes_2.5_no_think",
    "tulu_3_sft_personas_instruction_following_no_think",
    "Mixture_of_Thoughts_science_no_think",
    "hermes_function_calling_v1_no_think",
    "xlam_traces_no_think",
    "table_gpt_no_think",
    "LongAlign_64k_context_lang_annotated_lang_6_no_think",
)


def tokenize_with_assistant_mask(
    batch: dict[str, list[Any]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[Any]]:
    """Render ChatML turns and compute assistant-only labels + loss_mask."""
    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_loss_mask: list[list[int]] = []

    for messages in batch["messages"]:
        if not messages:
            continue
        full_ids: list[int] = []
        loss_mask: list[int] = []
        prefix_len = 0
        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            upto = tokenizer.apply_chat_template(
                messages[: idx + 1],
                tokenize=True,
                add_generation_prompt=False,
            )
            if role == "assistant":
                prompt_prefix = tokenizer.apply_chat_template(
                    messages[:idx],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                header_len = max(prefix_len, len(prompt_prefix))
                full_ids = list(upto)
                # Mask prompt/header tokens (0) and supervise assistant tokens + <|im_end|> (1).
                loss_mask.extend([0] * max(0, header_len - len(loss_mask)))
                loss_mask.extend([1] * max(0, len(full_ids) - len(loss_mask)))
            else:
                full_ids = list(upto)
                loss_mask.extend([0] * max(0, len(full_ids) - len(loss_mask)))
            prefix_len = len(full_ids)

        if not full_ids:
            continue
        full_ids = full_ids[:max_length]
        loss_mask = loss_mask[:max_length]
        if sum(loss_mask) == 0:
            continue
        labels = [
            tok_id if m == 1 else -100 for tok_id, m in zip(full_ids, loss_mask)
        ]
        all_input_ids.append(full_ids)
        all_attention_mask.append([1] * len(full_ids))
        all_labels.append(labels)
        all_loss_mask.append(loss_mask)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
        "loss_mask": all_loss_mask,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-checkpoint", default="/mnt/scratch/runs/laughlm-stage2-5b/final")
    parser.add_argument("--output-dir", default="/mnt/scratch/runs/laughlm-stage3-sft")
    parser.add_argument("--splits", nargs="+", default=list(VERIFIED_SMOLTALK2_SFT_NO_THINK_SPLITS))
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", use_fast=True)
    if tokenizer.eos_token_id != 2 or len(tokenizer) != 49152:
        raise ValueError(
            f"Unexpected instruct tokenizer geometry: len={len(tokenizer)}, eos={tokenizer.eos_token_id}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_parts = [
        load_dataset("HuggingFaceTB/smoltalk2", name="SFT", split=split_name)
        for split_name in args.splits
    ]
    merged = concatenate_datasets(raw_parts).shuffle(seed=42)
    tokenized = merged.map(
        lambda b: tokenize_with_assistant_mask(b, tokenizer, args.sequence_length),
        batched=True,
        num_proc=min(16, os.cpu_count() or 4),
        remove_columns=merged.column_names,
    )
    split_ds = tokenized.train_test_split(test_size=0.01, seed=42)

    cfg = TrainLMFineTuningConfig(
        model_name_or_path=str(Path(args.stage2_checkpoint).resolve()),
        tokenizer_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        data_format="tokens",
        args=TrainLMTrainingArguments(
            output_dir=Path(args.output_dir),
            max_steps=None,
            num_train_epochs=args.epochs,
            sequence_length=args.sequence_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=200,
            weight_decay=0.05,
            bf16=True,
            accelerator="cuda",
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to=("jsonl",),
        ),
    )

    result = run_finetuning(
        cfg,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["test"],
        tokenizer=tokenizer,
    )
    print(f"Stage 3 SFT complete. Saved model to: {result.model_path}", flush=True)


if __name__ == "__main__":
    main()
