#!/usr/bin/env python3
"""Stage 1 MI300X base pretraining: 30B tokens @ 2048 context (WSD flat at 3e-3)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

from trainlm import (
    PackedBinDataset,
    TrainLMTrainer,
    TrainLMTrainingArguments,
)
from trainlm.config import ModelSourceConfig
from trainlm.training import TokenWSD
from transformers import AutoTokenizer


def resolve_stage1_manifest_root(shards_path: Path) -> tuple[Path, str]:
    root = shards_path.expanduser().resolve()
    sub = root / "stage1"
    if sub.is_dir():
        manifests = sorted(sub.glob("*.manifest.json"))
        if manifests:
            data_path = json.loads(manifests[0].read_text(encoding="utf-8")).get("data_path", "")
            if str(data_path).startswith("stage1/"):
                return root, "stage1/*.manifest.json"
            return sub, "*.manifest.json"
    if root.name == "stage1":
        manifests = sorted(root.glob("*.manifest.json"))
        if manifests:
            data_path = json.loads(manifests[0].read_text(encoding="utf-8")).get("data_path", "")
            if str(data_path).startswith("stage1/"):
                return root.parent, "stage1/*.manifest.json"
    return root, "*.manifest.json"


def load_stage1_dataset(shards_path: Path, sequence_length: int, seed: int) -> PackedBinDataset:
    base_dir, pattern = resolve_stage1_manifest_root(shards_path)
    return PackedBinDataset.from_directory(
        base_dir,
        manifest_pattern=pattern,
        sequence_length=sequence_length,
        split="train",
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", default="/mnt/scratch/shards/stage1")
    parser.add_argument("--output-dir", default="/mnt/scratch/runs/laughlm-stage1-30b")
    parser.add_argument("--max-tokens", type=int, default=30_000_000_000)
    parser.add_argument("--horizon-tokens", type=int, default=35_000_000_000)
    parser.add_argument("--save-steps", type=int, default=2500)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = load_stage1_dataset(
        Path(args.shards_dir),
        sequence_length=2048,
        seed=42,
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)

    model_source = ModelSourceConfig(
        provider="huggingface",
        initialization="config",
        name_or_path="HuggingFaceTB/SmolLM2-135M",
        dtype="float32",  # FP32 master weights; TrainLM uses BF16 autocast when bf16=True.
        config_overrides={
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "vocab_size": 49152,
            "tie_word_embeddings": True,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
        },
    )

    training_args = TrainLMTrainingArguments(
        output_dir=output_dir,
        max_steps=None,
        max_tokens=args.max_tokens,
        sequence_length=2048,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=2,  # 128 * 2 * 2048 = 524,288 tokens/step
        learning_rate=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        lr_scheduler_type="wsd",
        bf16=True,
        accelerator="cuda",
        torch_compile=args.torch_compile,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        eval_strategy="no",
        # Checkpointing an IterableDataset (PackedBinDataset) requires num_workers=0 in TrainLM.
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to=("jsonl",),
    )

    trainer = TrainLMTrainer(
        model=model_source,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    # TrainLMTrainer.train() does not rebuild trainer.scheduler or trainer.engine;
    # checkpoint resume reloads state into this exact TokenWSD instance.
    scheduler = TokenWSD(
        trainer.optimizer,
        horizon_tokens=args.horizon_tokens,
        warmup_fraction=0.02,
        stable_fraction=0.84,
        min_lr_ratio=0.0,
    )
    trainer.scheduler = scheduler
    trainer.engine.scheduler = scheduler

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        export_dir = trainer.save_model(output_dir / "final")
        print(f"Stage 1 complete. Exported FP32/HF checkpoint to: {export_dir}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
