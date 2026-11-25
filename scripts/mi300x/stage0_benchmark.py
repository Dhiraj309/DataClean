#!/usr/bin/env python3
"""Stage 0 MI300X throughput & MFU benchmark on local stage1 packed shards."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

import torch
from trainlm import (
    PackedBinDataset,
    TrainLMTrainer,
    TrainLMTrainingArguments,
)
from trainlm.config import ModelSourceConfig


import json


def resolve_stage_manifest_root(
    shards_path: Path, stage_prefix: str
) -> tuple[Path, str]:
    root = shards_path.expanduser().resolve()
    sub = root / stage_prefix
    if sub.is_dir():
        manifests = sorted(sub.glob("*.manifest.json"))
        if manifests:
            data_path = json.loads(manifests[0].read_text(encoding="utf-8")).get("data_path", "")
            if str(data_path).startswith(f"{stage_prefix}/"):
                return root, f"{stage_prefix}/*.manifest.json"
            return sub, "*.manifest.json"
    if root.name == stage_prefix:
        manifests = sorted(root.glob("*.manifest.json"))
        if manifests:
            data_path = json.loads(manifests[0].read_text(encoding="utf-8")).get("data_path", "")
            if str(data_path).startswith(f"{stage_prefix}/"):
                return root.parent, f"{stage_prefix}/*.manifest.json"
    return root, "*.manifest.json"


def load_local_stage_dataset(
    shards_path: Path,
    *,
    stage_prefix: str,
    sequence_length: int,
    seed: int = 42,
) -> PackedBinDataset:
    base_dir, pattern = resolve_stage_manifest_root(shards_path, stage_prefix)
    return PackedBinDataset.from_directory(
        base_dir,
        manifest_pattern=pattern,
        sequence_length=sequence_length,
        split="train",
        seed=seed,
    )


def run_case(
    *,
    shards_dir: Path,
    output_root: Path,
    batch_size: int,
    torch_compile: bool,
    steps: int,
    sequence_length: int,
) -> dict[str, float]:
    dataset = load_local_stage_dataset(
        shards_dir,
        stage_prefix="stage1",
        sequence_length=sequence_length,
        seed=42,
    )
    model_source = ModelSourceConfig(
        provider="huggingface",
        initialization="config",
        name_or_path="HuggingFaceTB/SmolLM2-135M",
        dtype="float32",
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
    run_dir = output_root / f"bs{batch_size}_compile{int(torch_compile)}"
    args = TrainLMTrainingArguments(
        output_dir=run_dir,
        max_steps=steps,
        sequence_length=sequence_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        bf16=True,
        accelerator="cuda",
        torch_compile=torch_compile,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        logging_verbosity="normal",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = TrainLMTrainer(
        model=model_source,
        args=args,
        train_dataset=dataset,
    )
    try:
        trainer.train()
        perf = trainer.performance_summary()
        peak_vram_gib = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        tps = float(perf.get("tokens_per_second") or 0.0)
        mfu = float(perf.get("non_embedding_mfu") or 0.0)
        print(
            f"[BENCH] bs={batch_size:<3} compile={str(torch_compile):<5} | "
            f"tok/s={tps:,.1f} | non_embedding_mfu={mfu:.2%} | "
            f"peak_vram={peak_vram_gib:.2f} GiB",
            flush=True,
        )
        return {"tok_s": tps, "non_embedding_mfu": mfu, "peak_vram_gib": peak_vram_gib}
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", default="/mnt/scratch/shards")
    parser.add_argument("--output-dir", default="/mnt/scratch/runs/stage0_benchmark")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    output_root = Path(args.output_dir)
    results = []
    for bs in (64, 128):
        for compile_flag in (False, True):
            metrics = run_case(
                shards_dir=shards_dir,
                output_root=output_root,
                batch_size=bs,
                torch_compile=compile_flag,
                steps=args.steps,
                sequence_length=args.sequence_length,
            )
            results.append((bs, compile_flag, metrics))

    print("\n=== Stage 0 MI300X Benchmark Summary ===", flush=True)
    for bs, compile_flag, m in results:
        print(
            f"BS={bs:<3} torch_compile={str(compile_flag):<5} -> "
            f"{m['tok_s']:,.1f} tok/s | MFU={m['non_embedding_mfu']:.2%} | "
            f"VRAM={m['peak_vram_gib']:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
