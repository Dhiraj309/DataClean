#!/usr/bin/env python3
"""Stage 2 MI300X 4K annealing: 5B tokens @ 4096 context, rope_theta=100000, LR decay 3e-3 -> 0.0."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trainlm import (
    PackedBinDataset,
    TrainLMTrainer,
    TrainLMTrainingArguments,
)
from trainlm.training import TokenWSD


def load_stage2_dataset(shards_path: Path, sequence_length: int, seed: int) -> PackedBinDataset:
    root = shards_path.expanduser().resolve()
    if (root / "stage2").is_dir() and list((root / "stage2").glob("*.manifest.json")):
        return PackedBinDataset.from_directory(
            root,
            manifest_pattern="stage2/*.manifest.json",
            sequence_length=sequence_length,
            split="train",
            seed=seed,
        )
    if root.name == "stage2" and list(root.glob("*.manifest.json")):
        first_text = next(root.glob("*.manifest.json")).read_text(encoding="utf-8")
        if '"data_path": "stage2/' in first_text or '"data_path":"stage2/' in first_text:
            return PackedBinDataset.from_directory(
                root.parent,
                manifest_pattern="stage2/*.manifest.json",
                sequence_length=sequence_length,
                split="train",
                seed=seed,
            )
    return PackedBinDataset.from_directory(
        root,
        manifest_pattern="*.manifest.json",
        sequence_length=sequence_length,
        split="train",
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", default="/mnt/scratch/runs/laughlm-stage1-30b/final")
    parser.add_argument("--shards-dir", default="/mnt/scratch/shards/stage2")
    parser.add_argument("--output-dir", default="/mnt/scratch/runs/laughlm-stage2-5b")
    parser.add_argument("--max-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = load_stage2_dataset(
        Path(args.shards_dir),
        sequence_length=4096,
        seed=142,
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)

    # Update rope_theta BEFORE instantiating model so rotary inv_freq buffers use 100000.0.
    config = AutoConfig.from_pretrained(args.stage1_checkpoint)
    config.rope_theta = 100000.0
    config.max_position_embeddings = 4096
    model = AutoModelForCausalLM.from_pretrained(
        args.stage1_checkpoint,
        config=config,
        torch_dtype=torch.float32,
    )

    training_args = TrainLMTrainingArguments(
        output_dir=output_dir,
        max_steps=None,
        max_tokens=args.max_tokens,
        sequence_length=4096,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,  # 64 * 2 * 4096 = 524,288 tokens/step
        learning_rate=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        lr_scheduler_type="linear",
        warmup_steps=0,
        bf16=True,
        accelerator="cuda",
        torch_compile=args.torch_compile,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        eval_strategy="no",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=("jsonl",),
    )

    trainer = TrainLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    # Token-indexed linear decay 3e-3 -> 0.0 over 5B tokens with zero second warmup.
    scheduler = TokenWSD(
        trainer.optimizer,
        horizon_tokens=args.max_tokens,
        warmup_fraction=0.0,
        stable_fraction=0.0,
        min_lr_ratio=0.0,
    )
    trainer.scheduler = scheduler
    trainer.engine.scheduler = scheduler

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        export_dir = trainer.save_model(output_dir / "final")
        print(f"Stage 2 annealing complete. Exported model to: {export_dir}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
