# DataClean

Single-pass, resumable streaming dataset curation, filtering, deficit-weighted mixing, and packed binary tokenization pipeline for **LaughLM / TrainLM** pretraining at **30B–35B token** scale.

`DataClean` streams raw Parquet and `.jsonl.zst` corpora directly from Hugging Face Hub on CPU, applies per-source quality filters, tokenizes with the `HuggingFaceTB/SmolLM2-135M` tokenizer (`49,152` vocab), shuffles documents per shard, and emits zero-header packed binary `.bin` shards + `explicit_v1` manifests consumed directly by [`Dhiraj309/TrainLM`](https://github.com/Dhiraj309/TrainLM) at **`73,240.1 tok/s` (`~63.6%` total MFU) with `0.0%` host data-loading wait** on Google Cloud TPUs.

---

## Packed Binary Shard Specification (`Dhiraj45/laughlm-135m-shards-v2`)

Shards produced by `smol_stream_pack.py` are hosted at [`Dhiraj45/laughlm-135m-shards-v2`](https://huggingface.co/datasets/Dhiraj45/laughlm-135m-shards-v2):

- **Path pattern:**
  - Stage 1 (base pretraining): `stage1/stream_stage1_w00_shard_{index:05d}.bin` (plus companion `.manifest.json`)
  - Stage 2 (4K context annealing): `stage2/stream_stage2_w00_shard_{index:05d}.bin` (plus companion `.manifest.json`)
  - Resumable `O(1)` worker state: `_state/{stage}_w{worker_id:02d}.json`
- **Tokenizer & vocabulary:** `HuggingFaceTB/SmolLM2-135M` with a **`49,152`-token `SmolLM2` vocabulary** (`eos_token_id=0`).
- **Binary layout:** `header_bytes=0`, contiguous packed token IDs (`uint16` / `uint32`; `smol_stream_pack.py` emits little-endian `uint16` matching `49,152 <= 65,536`, and `TrainLM` supports both `uint16` and `uint32`), engineered for zero-copy sequential slicing (`mmap`) at `seq_len=2048` and `seq_len=4096` with zero runtime tokenization overhead.
- **Manifest sidecar (`explicit_v1`):** Each `.bin` file is accompanied by a `PackedBinaryShardManifest` JSON (`schema_version=1`, `manifest_type="packed_binary_shard"`, `compatibility_profile="explicit_v1"`, `header_bytes=0`, `token_dtype="uint16"`, `byte_order="little"`, `vocab_size=49152`, `token_count`, `token_id_min`, `token_id_max`, `file_size_bytes`, and `sha256`).

---

## Production Validation on Cloud TPU (`TrainLM`)

Validated end-to-end on Google Cloud **`TPU v5e-1`** running [`Dhiraj309/TrainLM`](https://github.com/Dhiraj309/TrainLM) for a 1,000-step Stage-1 pilot (`131.07M` tokens, `micro_batch_size=4`, `grad_accum=16`, `global_batch=64`, `seq_len=2048` = `131,072` tokens/step):

- **Zero Host Data Starvation:** Streaming [`Dhiraj45/laughlm-135m-shards-v2`](https://huggingface.co/datasets/Dhiraj45/laughlm-135m-shards-v2) via `PackedBinDataset` sustained **`73,240.1 tok/s` (`53.73%` non-logit MFU / `~63.6%` total MFU)** with **`wait=0.0%` host wait** across all 1,000 optimizer steps.
- **Clean Train / Held-Out Shard Separation & Convergence:**
  - **Training shard** (`stage1/stream_stage1_w00_shard_00000.bin`, `131.07M` tokens): cross-entropy loss converged smoothly from `10.9748` (`58,384` PPL) to **`4.2332` (`68.93` PPL)**.
  - **Held-out validation shard** (`stage1/stream_stage1_w00_shard_00029.bin`, `131,072` tokens, zero overlap): achieved **`4.2198` loss (`68.02` PPL)** at $R=4$ (`20` effective layers) on [`Dhiraj45/laughlm-loop-135m-pilot`](https://huggingface.co/Dhiraj45/laughlm-loop-135m-pilot), confirming clean out-of-sample generalization and leak-free shard boundaries.

| Metric | Pilot Result (`TPU v5e-1`, 1,000 steps) |
| :--- | :--- |
| Global Batch Geometry | `micro_batch_size=4`, `grad_accum=16`, `global_batch=64`, `seq_len=2048` (`131,072` tok/step) |
| Throughput & MFU | **`73,240.1 tok/s`** · `53.73%` non-logit MFU / **`~63.6%` total MFU** |
| Host Data-Loading Wait | **`wait=0.0%`** across all 1,000 optimizer steps (`PackedBinDataset`) |
| Train Loss / PPL (`shard_00000.bin`) | `10.9748` (`58,384` PPL) -> **`4.2332` (`68.93` PPL)** |
| Held-Out Val Loss / PPL (`shard_00029.bin`, $R=4$) | **`4.2198` (`68.02` PPL)** (`131,072` eval tokens, zero overlap) |

---

## Multi-Chip `v6e-8` / `v5e-8` Same-Region GCS Staging Workflow (35B-Token Scale)

For multi-chip Cloud TPU training on **`v6e-8`** or **`v5e-8`** slices at full **35B-token scale** (Stage 1: 30B tokens @ 2K + Stage 2: 5B tokens @ 4K), stage `DataClean` shards into a **single-region Google Cloud Storage (`gs://`) bucket in the exact same GCP region as the TPU slice** to eliminate internet bandwidth bottlenecks and guarantee **`$0.00` network egress**:

1. **Mirror shards from HF / packing scratch to your same-region GCS bucket:**

   ```bash
   # Bucket must be Single-Region in the exact same region as your TPU v6e-8 / v5e-8 slice ($0.00 egress)
   hf download Dhiraj45/laughlm-135m-shards-v2 --repo-type dataset --include "stage1/*" --local-dir /tmp/shards_v2
   gcloud storage rsync -r /tmp/shards_v2/stage1 gs://<your-same-region-bucket>/laughlm-135m-shards-v2/stage1
   ```

2. **Consume deterministically across all 8 TPU chips via `PackedBinDataset.from_gcs` / `stage_gcs_prefix` in `TrainLM`:**

   ```python
   from trainlm.data import PackedBinDataset

   train_ds = PackedBinDataset.from_gcs(
       "gs://<your-same-region-bucket>/laughlm-135m-shards-v2/stage1",
       sequence_length=2048,
       split="train",
       seed=42,
       epoch=0,
       world_size=8,   # e.g. v6e-8 / v5e-8 chip count
       rank=rank,      # JAX/PyTorch TPU process index in [0, 7]
       header_bytes=0,
       vocab_size=49152,
   )
   ```

   Under the hood in `TrainLM`:
   - `stage_gcs_prefix` elects the node-local leader (`local_rank == 0` / `rank == 0`) to copy manifests and `.bin` shards (plus any `.sha256` sidecars) onto local SSD once, verifies file integrity, writes an atomic `.staged.ok` marker, and synchronizes all workers via `dist.barrier()` before any rank reads.
   - `PackedBinDataset` constructs a deterministic global batch schedule from shard fingerprints and assigns disjoint slices per `(rank, world_size)`, ensuring every chip reads strictly non-overlapping token windows across the full 35B-token run.

---

## Repository Structure

```text
DataClean/
├── smol_stream_pack.py                 # Single-pass streaming filter -> mix -> tokenize -> .bin + manifest packer
├── smol_pipeline.py                    # Shared row normalization, language/score/length filters, config loader
├── configs/smol/
│   ├── stream_stage1_30b.yaml          # Stage 1: 30B tokens @ 2K context (DCLM, FineWeb-Edu, Cosmopedia-v2, StarCoderData, FineMath)
│   ├── stream_stage2_anneal_5b.yaml    # Stage 2: 5B tokens @ 4K context WSD annealing mix (9 streams, reverse_files=true)
│   ├── stage1/*.yaml                   # Legacy Stage 1 per-source filter configs
│   ├── stage2/*.yaml                   # Legacy Stage 2 mixture configs
│   └── stage3/tokenize.yaml            # Legacy Stage 3 offline tokenizer config
├── notebooks/
│   └── colab_stream_pack.ipynb         # Colab-ready Stage 1 + Stage 2 CPU packing notebook (24-vCPU / 8-vCPU auto-tuned)
├── scripts/mi300x/                     # Reference MI300X training & Hub sync scripts (stage0..stage4)
├── tests/                              # Unit & integration tests (pytest)
├── smol_stage1_filter.py               # Legacy 3-stage pipeline: Stage 1 row-group Parquet filter
├── smol_stage1_buffered.py             # Legacy 3-stage pipeline: Stage 1 rolling Parquet buffer
├── smol_stage2_mix.py                  # Legacy 3-stage pipeline: Stage 2 weighted Parquet mixer
├── smol_stage3_tokenize.py             # Legacy 3-stage pipeline: Stage 3 Parquet-to-.bin tokenizer
├── requirements-smol.txt               # Minimal runtime dependencies
├── SMOL_PIPELINE.md                    # Legacy 2/3-stage Parquet pipeline notes
└── IMPLEMENTATION_ROADMAP.md           # Project milestones & verification gates
```

---

## Streaming Mixtures (`smol_stream_pack.py`)

Both streaming configs target `HuggingFaceTB/SmolLM2-135M` (`vocab_size=49152`, `eos_token_id=0`) and `500M`-token (`~1 GiB`) shards:

- **Stage 1 Base Pretraining (`configs/smol/stream_stage1_30b.yaml`, `target_tokens=30B`):**
  - `dclm` (`HuggingFaceFW/dclm_100BT-shuffled`): `40%`
  - `fineweb_edu` (`HuggingFaceFW/fineweb_edu_100BT-shuffled`): `35%`
  - `cosmopedia_v2` (`HuggingFaceTB/smollm-corpus`): `10%`
  - `starcoderdata` (`bigcode/starcoderdata`, Python `4%` / JS `2%` / C++ `2%`): `8%`
  - `finemath-4plus` (`HuggingFaceTB/finemath`): `7%`
- **Stage 2 4K Annealing (`configs/smol/stream_stage2_anneal_5b.yaml`, `target_tokens=5B`, `reverse_files=true`):**
  - Web-anchored long-context mix (`>= 1024` tokens / `>= 4000` chars where applicable, reading Parquet file lists in reverse so Stage 2 never repeats Stage 1's opening files): `fineweb_edu` (`20%`), `finepdfs_edu` (`20%`), `cosmopedia_v2` (`20%`), `dclm` (`15%`), `starcoderdata` Python/JS/C++ (`15%` split `7%`/`4%`/`4%`), `finemath-4plus` (`5%`), and `megamath-web-pro` (`5%`).

### Running Locally or on Colab CPU

Interactive Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dhiraj309/DataClean/blob/main/notebooks/colab_stream_pack.ipynb)

```bash
pip install -r requirements-smol.txt
export HF_TOKEN=hf_your_write_token
export HF_XET_HIGH_PERFORMANCE=1

# Stage 1 (30B tokens -> stage1/)
python3 -u smol_stream_pack.py \
  --config configs/smol/stream_stage1_30b.yaml \
  --worker-id 0 \
  --num-workers 1 \
  --workers 18 \
  --batch-size 1024 \
  --max-runtime-minutes 660 \
  --resume

# Stage 2 (5B tokens -> stage2/)
python3 -u smol_stream_pack.py \
  --config configs/smol/stream_stage2_anneal_5b.yaml \
  --worker-id 0 \
  --num-workers 1 \
  --workers 18 \
  --batch-size 1024 \
  --max-runtime-minutes 660 \
  --resume
```

---

## Running Tests

```bash
pytest -v
```
