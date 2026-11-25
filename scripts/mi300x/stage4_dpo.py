#!/usr/bin/env python3
"""Stage 4 MI300X DPO alignment on HuggingFaceTB/smoltalk2 Preference no_think split."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


VERIFIED_PREFERENCE_NO_THINK_SPLIT = "llama_3.1_tulu_3_8b_preference_mixture_no_think"


def _format_pair(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    chosen_msgs = list(example["chosen"])
    rejected_msgs = list(example["rejected"])
    # Extract shared prompt turns before final assistant response.
    prompt_msgs = chosen_msgs[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen_full = tokenizer.apply_chat_template(
        chosen_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    rejected_full = tokenizer.apply_chat_template(
        rejected_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    chosen_resp = chosen_full[len(prompt_text) :] if chosen_full.startswith(prompt_text) else chosen_msgs[-1]["content"] + tokenizer.eos_token
    rejected_resp = rejected_full[len(prompt_text) :] if rejected_full.startswith(prompt_text) else rejected_msgs[-1]["content"] + tokenizer.eos_token
    return {
        "prompt": prompt_text,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-model", default="/mnt/scratch/runs/laughlm-stage3-sft/model")
    parser.add_argument("--output-dir", default="/mnt/scratch/runs/laughlm-stage4-dpo")
    parser.add_argument("--hub-repo", default="Dhiraj309/laughlm-135m-deep-narrow-v2")
    parser.add_argument("--split", default=VERIFIED_PREFERENCE_NO_THINK_SPLIT)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("HuggingFaceTB/smoltalk2", name="Preference", split=args.split)
    formatted = raw.map(
        lambda ex: _format_pair(ex, tokenizer),
        num_proc=min(16, os.cpu_count() or 4),
        remove_columns=raw.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.stage3_model,
        torch_dtype=torch.bfloat16,
    )

    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=formatted,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Stage 4 DPO complete. Saved model to: {final_dir}", flush=True)

    if not args.no_push:
        model.push_to_hub(args.hub_repo, private=True)
        tokenizer.push_to_hub(args.hub_repo, private=True)
        print(f"Pushed aligned model to https://huggingface.co/{args.hub_repo}", flush=True)


if __name__ == "__main__":
    main()
