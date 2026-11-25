#!/usr/bin/env python3
"""Single-pass, multi-core, resumable streaming shard packer for TrainLM.

Streams Parquet (local single-file prefetch for <= 2 GiB, HfFileSystem range
reads for > 2 GiB) or .jsonl.zst datasets, applies smol_pipeline.accepts(),
tokenizes with SmolLM2-135M across a ProcessPoolExecutor, mixes by token
deficit, shuffles documents per shard, emits headerless little-endian uint16
.bin shards with TrainLM explicit_v1 manifests, and commits atomically to
Hugging Face Hub via a double-buffered async uploader.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import random
import re
import shutil
import sys
import threading
import time
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import zstandard as zstd

from smol_pipeline import (
    accepts,
    as_int,
    as_string,
    get_path,
    load_config,
    normalize_row,
)


TWO_GIB = 2 * 1024 * 1024 * 1024
DEFAULT_SHARD_TOKENS = 500_000_000
DEFAULT_BATCH_SIZE = 1024
DEFAULT_MAX_RUNTIME_MINUTES = 680.0
DEFAULT_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_VOCAB_SIZE = 49152
DEFAULT_MAX_DOC_TOKENS = 8192
DEFAULT_HARD_DOC_LIMIT = 65536


def default_tmp_dir() -> Path:
    kaggle_temp = Path("/kaggle/temp")
    if kaggle_temp.exists():
        return kaggle_temp / "smol_pack"
    return Path("/tmp/smol_pack")


def _preparse_tmp_dir(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tmp-dir", default=None)
    known, _ = parser.parse_known_args(argv)
    if known.tmp_dir:
        return Path(known.tmp_dir).expanduser()
    return default_tmp_dir()


def configure_environment(tmp_dir: Path) -> None:
    """Route HF caches into scratch tmp_dir before importing HF libraries."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = tmp_dir / "hf_cache"
    xet_cache = hf_cache / "xet"
    hf_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    if os.environ.get("HF_XET_HIGH_PERFORMANCE") != "1":
        print(
            "WARNING: HF_XET_HIGH_PERFORMANCE=1 is not set; Hugging Face Xet "
            "downloads/uploads may run slower on Kaggle/Colab.",
            file=sys.stderr,
            flush=True,
        )


def optional_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    return token.strip() if token and token.strip() else None


# ---------------------------------------------------------------------------
# Worker process state & tokenization
# ---------------------------------------------------------------------------

_WORKER_TOKENIZER: Any = None
_WORKER_EOS_ID: int = 0
_WORKER_VOCAB_SIZE: int = DEFAULT_VOCAB_SIZE
_WORKER_MAX_DOC_TOKENS: int = DEFAULT_MAX_DOC_TOKENS
_WORKER_HARD_DOC_LIMIT: int = DEFAULT_HARD_DOC_LIMIT


def _worker_init(
    tokenizer_id: str,
    vocab_size: int,
    max_doc_tokens: int,
    hard_doc_limit: int,
    expected_eos_id: int | None = None,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOS_ID, _WORKER_VOCAB_SIZE
    global _WORKER_MAX_DOC_TOKENS, _WORKER_HARD_DOC_LIMIT

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        model_max_length=10_000_000,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Tokenizer {tokenizer_id!r} must be a fast tokenizer")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {tokenizer_id!r} has no eos_token_id")
    if expected_eos_id is not None and int(eos_id) != int(expected_eos_id):
        raise ValueError(
            f"Tokenizer {tokenizer_id!r} eos_token_id={eos_id} != expected {expected_eos_id}"
        )
    if int(eos_id) >= int(vocab_size) or int(vocab_size) > 65536:
        raise ValueError(
            f"Tokenizer vocab_size={vocab_size}, eos_id={eos_id} incompatible with uint16"
        )
    _WORKER_TOKENIZER = tokenizer
    _WORKER_EOS_ID = int(eos_id)
    _WORKER_VOCAB_SIZE = int(vocab_size)
    _WORKER_MAX_DOC_TOKENS = int(max_doc_tokens)
    _WORKER_HARD_DOC_LIMIT = int(hard_doc_limit)


def tokenize_and_filter_rows(
    rows: list[dict[str, Any]],
    source_cfg: dict[str, Any],
    tokenizer: Any,
    eos_id: int,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_doc_tokens: int = DEFAULT_MAX_DOC_TOKENS,
    hard_doc_limit: int = DEFAULT_HARD_DOC_LIMIT,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Normalize, filter with smol_pipeline.accepts(), and pack uint16 docs."""
    source_name = str(source_cfg["name"])
    filters = source_cfg.get("filters") or {}
    norm_cfg = {
        "source": {
            "repo_id": source_cfg.get("repo_id", ""),
            "config": source_cfg.get("config"),
            "split": source_cfg.get("split", "train"),
        },
        "columns": source_cfg.get("columns") or {},
    }

    texts: list[str] = []
    rejected = 0
    for raw in rows:
        norm = normalize_row(raw, norm_cfg, source_name)
        ok, _ = accepts(norm, filters)
        if not ok:
            rejected += 1
            continue
        texts.append(norm["text"])

    if not texts:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.int32),
            {"seen": len(rows), "accepted": 0, "rejected": rejected, "tokens": 0},
        )

    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=hard_doc_limit,
    )

    flat_tokens: list[int] = []
    doc_lengths: list[int] = []
    for ids in encoded["input_ids"]:
        if not ids:
            continue
        for start in range(0, len(ids), max_doc_tokens):
            chunk = ids[start : start + max_doc_tokens]
            if not chunk:
                continue
            flat_tokens.extend(int(v) for v in chunk)
            flat_tokens.append(eos_id)
            doc_lengths.append(len(chunk) + 1)

    if not flat_tokens:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.int32),
            {
                "seen": len(rows),
                "accepted": len(texts),
                "rejected": rejected,
                "tokens": 0,
            },
        )

    arr = np.asarray(flat_tokens, dtype=np.int64)
    if int(arr.min()) < 0 or int(arr.max()) >= vocab_size:
        raise ValueError(
            f"Token ID out of range [0, {vocab_size}): min={arr.min()}, max={arr.max()}"
        )
    tokens_u16 = arr.astype(np.uint16)
    lengths_i32 = np.asarray(doc_lengths, dtype=np.int32)
    return (
        tokens_u16,
        lengths_i32,
        {
            "seen": len(rows),
            "accepted": len(texts),
            "rejected": rejected,
            "tokens": int(tokens_u16.size),
        },
    )


def _worker_process_batch(
    task: tuple[list[dict[str, Any]], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Worker tokenizer was not initialized")
    rows, source_cfg = task
    return tokenize_and_filter_rows(
        rows,
        source_cfg,
        _WORKER_TOKENIZER,
        _WORKER_EOS_ID,
        vocab_size=_WORKER_VOCAB_SIZE,
        max_doc_tokens=_WORKER_MAX_DOC_TOKENS,
        hard_doc_limit=_WORKER_HARD_DOC_LIMIT,
    )


# ---------------------------------------------------------------------------
# File discovery, partitioning & O(1) resumable readers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RemoteFileEntry:
    path: str
    size: int
    revision: str
    local_override: Path | None = None


def _matches_pattern(path: str, pattern: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern)


def discover_source_files(
    api: Any,
    source_cfg: dict[str, Any],
    *,
    global_reverse: bool,
    worker_id: int,
    num_workers: int,
    token: str | None,
) -> tuple[str, list[RemoteFileEntry]]:
    """Enumerate sorted source files (with sizes), optionally reverse, and shard across workers."""
    # Local directory mode for offline tests / local benchmarks
    if source_cfg.get("local_files"):
        entries = [
            RemoteFileEntry(
                path=str(Path(p)),
                size=Path(p).stat().st_size,
                revision="local",
                local_override=Path(p),
            )
            for p in sorted(str(x) for x in source_cfg["local_files"])
        ]
        reverse = bool(source_cfg.get("reverse_files", global_reverse))
        if reverse:
            entries.reverse()
        return "local", entries[worker_id::num_workers]

    repo_id = str(source_cfg["repo_id"])
    requested_rev = str(source_cfg.get("revision", "main"))
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
        revision=requested_rev,
        token=token,
    )
    revision = str(getattr(info, "sha", None) or requested_rev)

    prefixes = source_cfg.get("path_prefixes")
    if not prefixes:
        single = str(source_cfg.get("path_prefix", "")).strip("/")
        prefixes = [single] if single else [""]
    else:
        prefixes = [str(p).strip("/") for p in prefixes]

    pattern = str(source_cfg.get("file_pattern", "*.parquet"))
    discovered: list[RemoteFileEntry] = []
    seen_paths: set[str] = set()
    for prefix in prefixes:
        tree_iter = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=prefix or None,
            recursive=True,
            revision=revision,
            token=token,
        )
        for item in tree_iter:
            item_path = getattr(item, "path", None)
            item_size = getattr(item, "size", None)
            if not item_path or item_size is None:
                continue
            norm_path = str(item_path).strip("/")
            if norm_path in seen_paths:
                continue
            if not _matches_pattern(norm_path, pattern):
                continue
            seen_paths.add(norm_path)
            discovered.append(
                RemoteFileEntry(
                    path=norm_path,
                    size=int(item_size),
                    revision=revision,
                )
            )

    discovered.sort(key=lambda entry: entry.path)
    reverse = bool(source_cfg.get("reverse_files", global_reverse))
    if reverse:
        discovered.reverse()
    partitioned = discovered[worker_id::num_workers]
    return revision, partitioned


def _source_read_columns(source_cfg: dict[str, Any], available: list[str]) -> list[str]:
    columns = source_cfg.get("columns") or {}
    requested: list[str] = []
    logical_fields = (
        "text",
        "id",
        "url",
        "language",
        "dataset",
        "language_score",
        "fasttext_score",
        "score",
        "answer_count",
        "accepted_answer_id",
        "int_score",
        "token_count",
    )
    for logical in logical_fields:
        col = columns.get(logical, logical)
        if col:
            requested.append(str(col).split(".", 1)[0])
    for col in source_cfg.get("required_columns", []):
        requested.append(str(col).split(".", 1)[0])
    available_set = set(available)
    cols = list(dict.fromkeys(c for c in requested if c in available_set))
    return cols or available


class SingleFilePrefetcher:
    """Background prefetcher keeping at most active + 1 next <=2GiB file on disk."""

    def __init__(self, prefetch_root: Path, token: str | None) -> None:
        self.prefetch_root = prefetch_root
        self.prefetch_root.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.lock = threading.Lock()
        self.in_flight: dict[tuple[str, int], tuple[threading.Event, dict[str, Any]]] = {}

    def _file_dir(self, domain: str, file_idx: int) -> Path:
        return self.prefetch_root / domain / f"f{file_idx:05d}"

    def _download_one(
        self, domain: str, file_idx: int, repo_id: str, entry: RemoteFileEntry
    ) -> Path:
        if entry.local_override is not None:
            return entry.local_override
        from huggingface_hub import hf_hub_download

        target_dir = self._file_dir(domain, file_idx)
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=entry.path,
            repo_type="dataset",
            revision=entry.revision,
            token=self.token,
            local_dir=str(target_dir),
        )
        return Path(local_path)

    def submit(
        self, domain: str, repo_id: str, file_idx: int, entry: RemoteFileEntry
    ) -> None:
        if entry.local_override is not None or entry.size > TWO_GIB:
            return
        if entry.path.lower().endswith(".jsonl.zst"):
            return
        key = (domain, file_idx)
        with self.lock:
            if key in self.in_flight:
                return
            done = threading.Event()
            box: dict[str, Any] = {}
            self.in_flight[key] = (done, box)

        def _worker() -> None:
            try:
                box["path"] = self._download_one(domain, file_idx, repo_id, entry)
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

        t = threading.Thread(
            target=_worker, name=f"prefetch-{domain}-{file_idx}", daemon=True
        )
        t.start()

    def get(
        self, domain: str, repo_id: str, file_idx: int, entry: RemoteFileEntry
    ) -> Path:
        if entry.local_override is not None:
            return entry.local_override
        key = (domain, file_idx)
        with self.lock:
            item = self.in_flight.pop(key, None)
        if item is None:
            return self._download_one(domain, file_idx, repo_id, entry)
        done, box = item
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["path"]

    def cleanup_file(self, domain: str, file_idx: int, entry: RemoteFileEntry) -> None:
        if entry.local_override is not None:
            return
        shutil.rmtree(self._file_dir(domain, file_idx), ignore_errors=True)

    def close(self) -> None:
        # Daemon prefetch threads exit immediately on process termination.
        pass


class DomainBatchReader:
    """Sequential O(1)-resumable reader yielding (rows, end_cursor) for one domain."""

    def __init__(
        self,
        source_cfg: dict[str, Any],
        files: list[RemoteFileEntry],
        cursor: dict[str, Any],
        batch_size: int,
        prefetcher: SingleFilePrefetcher,
        hfs: Any,
        token: str | None,
    ) -> None:
        self.source_cfg = source_cfg
        self.name = str(source_cfg["name"])
        self.repo_id = str(source_cfg.get("repo_id", ""))
        self.files = files
        self.batch_size = max(1, int(batch_size))
        self.prefetcher = prefetcher
        self.hfs = hfs
        self.token = token

        self.file_idx = int(cursor.get("file_idx", 0))
        self.row_group = int(cursor.get("row_group", 0))
        self.row_offset = int(cursor.get("row_offset", 0))
        self.exhausted = bool(cursor.get("exhausted", False)) or (
            self.file_idx >= len(self.files)
        )
        self._iter: Iterator[tuple[list[dict[str, Any]], dict[str, Any]]] | None = None
        if not self.exhausted:
            self._iter = self._stream_from_cursor()

    def _next_small_parquet_file_prefetch(self, idx: int) -> None:
        if idx < len(self.files):
            nxt = self.files[idx]
            if (
                nxt.local_override is None
                and nxt.size <= TWO_GIB
                and nxt.path.lower().endswith(".parquet")
            ):
                self.prefetcher.submit(self.name, self.repo_id, idx, nxt)

    def _iter_parquet(
        self, f_idx: int, entry: RemoteFileEntry, start_rg: int, start_offset: int
    ) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
        use_range = (
            entry.local_override is None
            and entry.size > TWO_GIB
        )
        local_path: Path | None = None
        remote_handle: Any = None
        prefetched_next = False
        try:
            if use_range:
                hf_uri = f"datasets/{self.repo_id}@{entry.revision}/{entry.path}"
                remote_handle = self.hfs.open(hf_uri, "rb")
                pf = pq.ParquetFile(remote_handle)
            else:
                local_path = self.prefetcher.get(self.name, self.repo_id, f_idx, entry)
                pf = pq.ParquetFile(local_path)

            available = list(pf.schema_arrow.names)
            required = self.source_cfg.get(
                "required_columns",
                [(self.source_cfg.get("columns") or {}).get("text", "text")],
            )
            missing = [
                str(col).split(".", 1)[0]
                for col in required
                if str(col).split(".", 1)[0] not in set(available)
            ]
            if missing:
                raise RuntimeError(
                    f"{self.name}:{entry.path}: missing required columns {missing}; available={available}"
                )
            read_cols = _source_read_columns(self.source_cfg, available)
            num_rgs = pf.num_row_groups

            for rg in range(start_rg, num_rgs):
                if (
                    not use_range
                    and not prefetched_next
                    and rg >= max(0, num_rgs - 2)
                ):
                    self._next_small_parquet_file_prefetch(f_idx + 1)
                    prefetched_next = True
                rg_rows = int(pf.metadata.row_group(rg).num_rows)
                skip = start_offset if rg == start_rg else 0
                if skip >= rg_rows:
                    continue
                emitted_until = skip
                for attempt in range(5):
                    try:
                        row_pos = 0
                        for batch in pf.iter_batches(
                            row_groups=[rg],
                            columns=read_cols,
                            batch_size=self.batch_size,
                            use_threads=False,
                        ):
                            b_len = int(batch.num_rows)
                            if row_pos + b_len <= emitted_until:
                                row_pos += b_len
                                continue
                            if row_pos < emitted_until:
                                offset_in_batch = emitted_until - row_pos
                                batch = batch.slice(offset_in_batch)
                                row_pos = emitted_until
                            rows = batch.to_pylist()
                            row_pos += len(rows)
                            emitted_until = row_pos
                            if row_pos < rg_rows:
                                end_cursor = {
                                    "file_idx": f_idx,
                                    "row_group": rg,
                                    "row_offset": row_pos,
                                    "exhausted": False,
                                }
                            elif rg + 1 < num_rgs:
                                end_cursor = {
                                    "file_idx": f_idx,
                                    "row_group": rg + 1,
                                    "row_offset": 0,
                                    "exhausted": False,
                                }
                            else:
                                is_last = f_idx + 1 >= len(self.files)
                                end_cursor = {
                                    "file_idx": f_idx + 1,
                                    "row_group": 0,
                                    "row_offset": 0,
                                    "exhausted": is_last,
                                }
                            yield rows, end_cursor
                        break
                    except Exception as exc:
                        if not use_range or attempt >= 4:
                            raise
                        delay = float(2 ** attempt) * random.uniform(0.7, 1.3)
                        print(
                            f"Range-read retry {attempt + 1}/5 for {self.name}:{entry.path} "
                            f"rg={rg} offset={emitted_until}: {exc} (sleeping {delay:.1f}s)",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(delay)
                        if remote_handle is not None:
                            try:
                                remote_handle.close()
                            except Exception:
                                pass
                        hf_uri = f"datasets/{self.repo_id}@{entry.revision}/{entry.path}"
                        remote_handle = self.hfs.open(hf_uri, "rb")
                        pf = pq.ParquetFile(remote_handle)
        finally:
            if remote_handle is not None:
                try:
                    remote_handle.close()
                except Exception:
                    pass
            if local_path is not None:
                self.prefetcher.cleanup_file(self.name, f_idx, entry)

    def _iter_jsonl_zst(
        self, f_idx: int, entry: RemoteFileEntry, start_offset: int
    ) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
        raw_handle: Any = None
        if entry.local_override is not None:
            raw_handle = entry.local_override.open("rb")
        else:
            hf_uri = f"datasets/{self.repo_id}@{entry.revision}/{entry.path}"
            raw_handle = self.hfs.open(hf_uri, "rb")

        dctx = zstd.ZstdDecompressor()
        try:
            with dctx.stream_reader(raw_handle) as zf:
                text_stream = io.TextIOWrapper(zf, encoding="utf-8", errors="replace")
                line_idx = 0
                while line_idx < start_offset:
                    skipped = text_stream.readline()
                    if not skipped:
                        return
                    line_idx += 1

                batch: list[dict[str, Any]] = []
                for line in text_stream:
                    line_idx += 1
                    stripped = line.strip()
                    if not stripped:
                        continue
                    batch.append(json.loads(stripped))
                    if len(batch) >= self.batch_size:
                        yield batch, {
                            "file_idx": f_idx,
                            "row_group": 0,
                            "row_offset": line_idx,
                            "exhausted": False,
                        }
                        batch = []
                is_last = f_idx + 1 >= len(self.files)
                if batch:
                    yield batch, {
                        "file_idx": f_idx + 1,
                        "row_group": 0,
                        "row_offset": 0,
                        "exhausted": is_last,
                    }
        finally:
            try:
                raw_handle.close()
            except Exception:
                pass

    def _stream_from_cursor(self) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
        start_f = self.file_idx
        start_rg = self.row_group
        start_off = self.row_offset

        if start_f < len(self.files):
            first_entry = self.files[start_f]
            if (
                first_entry.local_override is None
                and first_entry.size <= TWO_GIB
                and first_entry.path.lower().endswith(".parquet")
            ):
                self.prefetcher.submit(self.name, self.repo_id, start_f, first_entry)

        for f_idx in range(start_f, len(self.files)):
            entry = self.files[f_idx]
            rg = start_rg if f_idx == start_f else 0
            off = start_off if f_idx == start_f else 0
            lower = entry.path.lower()
            if lower.endswith(".jsonl.zst") or lower.endswith(".json.zst"):
                yield from self._iter_jsonl_zst(f_idx, entry, off)
            else:
                yield from self._iter_parquet(f_idx, entry, rg, off)

    def next_batch(self) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        if self.exhausted or self._iter is None:
            self.exhausted = True
            return None
        try:
            return next(self._iter)
        except StopIteration:
            self.exhausted = True
            return None

    def close(self) -> None:
        if self._iter is not None and hasattr(self._iter, "close"):
            try:
                self._iter.close()
            except Exception:
                pass
        self._iter = None


# ---------------------------------------------------------------------------
# Manifest, Shard Writer, State & Double-Buffered Async Uploader
# ---------------------------------------------------------------------------

def build_explicit_v1_manifest(
    *,
    shard_id: str,
    data_path: str,
    token_count: int,
    token_id_min: int,
    token_id_max: int,
    vocab_size: int,
    file_size_bytes: int,
    sha256: str,
) -> dict[str, Any]:
    """Build a TrainLM PackedBinaryShardManifest (explicit_v1) JSON dict."""
    return {
        "schema_version": 1,
        "manifest_type": "packed_binary_shard",
        "shard_id": shard_id,
        "data_path": data_path,
        "compatibility_profile": "explicit_v1",
        "header_bytes": 0,
        "token_dtype": "uint16",
        "byte_order": "little",
        "token_count": int(token_count),
        "token_id_min": int(token_id_min),
        "token_id_max": int(token_id_max),
        "vocab_size": int(vocab_size),
        "file_size_bytes": int(file_size_bytes),
        "sha256": sha256,
        "documents": {"storage": "unavailable"},
    }


def write_shuffled_shard(
    *,
    chunks: list[np.ndarray],
    doc_lengths: list[np.ndarray],
    shard_path: Path,
    shard_idx: int,
    worker_id: int,
) -> tuple[int, int, int, int, str]:
    """Deterministically shuffle documents in RAM and write headerless little-endian uint16 .bin."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(".bin.tmp")
    tmp_path.unlink(missing_ok=True)

    flat_tokens = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    all_lengths = np.concatenate(doc_lengths) if len(doc_lengths) > 1 else doc_lengths[0]
    num_docs = int(all_lengths.size)
    token_count = int(flat_tokens.size)
    if token_count <= 0 or num_docs <= 0:
        raise ValueError("Cannot write an empty shard")

    token_id_min = int(flat_tokens.min())
    token_id_max = int(flat_tokens.max())

    # Compute document start/end offsets and shuffle order deterministically
    ends = np.cumsum(all_lengths, dtype=np.int64)
    starts = np.empty_like(ends)
    starts[0] = 0
    if num_docs > 1:
        starts[1:] = ends[:-1]

    rng = np.random.default_rng(seed=int(shard_idx) + int(worker_id) * 100_003)
    perm = rng.permutation(num_docs)

    digest = hashlib.sha256()
    # Stream-write shuffled docs in ~16M-token (~32 MiB) buffer blocks to avoid
    # duplicating the full 1 GiB array in RAM.
    block_budget = 16 * 1024 * 1024
    buf_slices: list[np.ndarray] = []
    buf_tokens = 0
    with tmp_path.open("wb") as handle:
        for doc_i in perm:
            s = int(starts[doc_i])
            e = int(ends[doc_i])
            sl = flat_tokens[s:e]
            buf_slices.append(sl)
            buf_tokens += e - s
            if buf_tokens >= block_budget:
                block = np.concatenate(buf_slices)
                if block.dtype != np.dtype("<u2"):
                    block = block.astype("<u2", copy=False)
                raw = block.tobytes()
                handle.write(raw)
                digest.update(raw)
                buf_slices.clear()
                buf_tokens = 0
        if buf_slices:
            block = np.concatenate(buf_slices) if len(buf_slices) > 1 else buf_slices[0]
            if block.dtype != np.dtype("<u2"):
                block = block.astype("<u2", copy=False)
            raw = block.tobytes()
            handle.write(raw)
            digest.update(raw)
            buf_slices.clear()
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_path, shard_path)
    file_size = shard_path.stat().st_size
    if file_size != token_count * 2:
        raise RuntimeError(
            f"Written shard size {file_size} != expected {token_count * 2}"
        )
    return token_count, token_id_min, token_id_max, file_size, digest.hexdigest()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _is_retryable_hf_error(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (412, 429) or (isinstance(status, int) and 500 <= status < 600):
        return True
    text = str(exc)
    return any(code in text for code in ("412", "429", "500", "502", "503", "504", "Precondition", "Too Many Requests", "timed out"))


def commit_with_backoff(
    api: Any,
    *,
    repo_id: str,
    operations: list[Any],
    commit_message: str,
    token: str,
    max_attempts: int = 8,
) -> None:
    for attempt in range(max_attempts):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
                revision="main",
                token=token,
            )
            return
        except Exception as exc:
            if attempt + 1 >= max_attempts or not _is_retryable_hf_error(exc):
                raise
            delay = min(180.0, float(2 ** attempt)) * random.uniform(0.7, 1.3)
            print(
                f"Upload commit retry {attempt + 1}/{max_attempts} after error: {exc} "
                f"(sleeping {delay:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def prune_xet_cache(tmp_dir: Path, max_age_seconds: float = 30.0) -> None:
    xet_cache = tmp_dir / "hf_cache" / "xet"
    if not xet_cache.exists():
        return
    now = time.time()
    for root, _, files in os.walk(xet_cache):
        for fname in files:
            fpath = Path(root) / fname
            try:
                if now - fpath.stat().st_mtime >= max_age_seconds:
                    fpath.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class UploadJob:
    shard_id: str
    bin_path: Path
    manifest_path: Path
    remote_manifest_path: str
    manifest_dict: dict[str, Any]
    state_path: Path
    remote_state_path: str
    state_snapshot: dict[str, Any]


class AsyncShardUploader:
    """Single background uploader thread consuming queue.Queue(maxsize=1)."""

    def __init__(
        self,
        *,
        api: Any,
        repo_id: str,
        token: str | None,
        tmp_dir: Path,
        no_upload: bool,
    ) -> None:
        self.api = api
        self.repo_id = repo_id
        self.token = token
        self.tmp_dir = tmp_dir
        self.no_upload = no_upload
        self.queue: queue.Queue[UploadJob | None] = queue.Queue(maxsize=1)
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="shard-uploader", daemon=True
        )
        self.thread.start()

    def _process_job(self, job: UploadJob) -> None:
        atomic_write_json(job.manifest_path, job.manifest_dict)
        atomic_write_json(job.state_path, job.state_snapshot)
        if self.no_upload:
            print(
                f"[no-upload] Wrote {job.shard_id} "
                f"({job.manifest_dict['token_count']:,} tokens, "
                f"{job.manifest_dict['file_size_bytes'] / 1024**2:.1f} MiB)",
                flush=True,
            )
            return

        if not self.token:
            raise RuntimeError("HF_TOKEN is required when uploading shards")
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(
                path_in_repo=job.manifest_dict["data_path"],
                path_or_fileobj=str(job.bin_path),
            ),
            CommitOperationAdd(
                path_in_repo=job.remote_manifest_path,
                path_or_fileobj=str(job.manifest_path),
            ),
            CommitOperationAdd(
                path_in_repo=job.remote_state_path,
                path_or_fileobj=str(job.state_path),
            ),
        ]
        commit_with_backoff(
            self.api,
            repo_id=self.repo_id,
            operations=operations,
            commit_message=f"Add {job.shard_id} ({job.manifest_dict['token_count']:,} tokens)",
            token=self.token,
        )
        job.bin_path.unlink(missing_ok=True)
        job.manifest_path.unlink(missing_ok=True)
        prune_xet_cache(self.tmp_dir)
        print(
            f"Uploaded {job.shard_id} "
            f"({job.manifest_dict['token_count']:,} tokens, "
            f"{job.manifest_dict['file_size_bytes'] / 1024**2:.1f} MiB)",
            flush=True,
        )

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                self._process_job(job)
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def submit(self, job: UploadJob) -> None:
        if self.error is not None:
            raise self.error
        self.queue.put(job)
        if self.error is not None:
            raise self.error

    def finish(self) -> None:
        self.queue.put(None)
        self.queue.join()
        self.thread.join()
        if self.error is not None:
            raise self.error


# ---------------------------------------------------------------------------
# Deficit-weighted mixing & state loading
# ---------------------------------------------------------------------------

def _tie_rank(seed: int, name: str) -> int:
    digest = hashlib.blake2b(
        f"{int(seed)}:{name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def choose_deficit_domain(
    sources: list[dict[str, Any]],
    tokens_by_domain: dict[str, int],
    exhausted: set[str],
    seed: int,
) -> dict[str, Any]:
    total_weight = sum(float(s["weight"]) for s in sources)
    candidates = []
    for src in sources:
        name = str(src["name"])
        if name in exhausted:
            continue
        norm_w = float(src["weight"]) / total_weight
        emitted = int(tokens_by_domain.get(name, 0))
        candidates.append((emitted / norm_w, _tie_rank(seed, name), src))
    if not candidates:
        raise StopIteration
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def compute_worker_target_tokens(
    global_target: int, worker_id: int, num_workers: int
) -> int:
    """Split global_target across workers so sum(all workers) == global_target."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")
    base, rem = divmod(int(global_target), int(num_workers))
    return base + (1 if int(worker_id) < rem else 0)


def load_resume_state(
    *,
    local_state_path: Path,
    remote_repo: str | None,
    remote_state_path: str,
    api: Any,
    token: str | None,
    no_upload: bool,
    resume: bool,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    default_cursors = {
        str(src["name"]): {
            "file_idx": 0,
            "row_group": 0,
            "row_offset": 0,
            "tokens_emitted": 0,
            "exhausted": False,
        }
        for src in sources
    }
    fresh = {
        "version": 1,
        "next_shard_idx": 0,
        "total_tokens": 0,
        "cursors": default_cursors,
    }
    if not resume:
        return fresh

    loaded: dict[str, Any] | None = None
    if not no_upload and remote_repo:
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id=remote_repo,
                filename=remote_state_path,
                repo_type="dataset",
                revision="main",
                token=token,
            )
            loaded = json.loads(Path(cached).read_text(encoding="utf-8"))
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 404 and "Entry Not Found" not in str(exc) and "404" not in str(exc):
                print(
                    f"Warning: remote state fetch failed ({exc}); checking local fallback.",
                    file=sys.stderr,
                    flush=True,
                )

    if loaded is None and local_state_path.is_file():
        loaded = json.loads(local_state_path.read_text(encoding="utf-8"))

    if loaded is None:
        return fresh

    cursors = dict(default_cursors)
    for name, c in (loaded.get("cursors") or {}).items():
        if name in cursors:
            cursors[name] = {
                "file_idx": int(c.get("file_idx", 0)),
                "row_group": int(c.get("row_group", 0)),
                "row_offset": int(c.get("row_offset", 0)),
                "tokens_emitted": int(c.get("tokens_emitted", 0)),
                "exhausted": bool(c.get("exhausted", False)),
            }
    return {
        "version": 1,
        "next_shard_idx": int(loaded.get("next_shard_idx", 0)),
        "total_tokens": int(loaded.get("total_tokens", 0)),
        "cursors": cursors,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Local ProcessPoolExecutor worker count",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument("--path-prefix", default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--dry-run-tokens", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--max-runtime-minutes", type=float, default=DEFAULT_MAX_RUNTIME_MINUTES
    )
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    tmp_dir = _preparse_tmp_dir(argv)
    configure_environment(tmp_dir)
    args = parse_args(argv)

    if args.worker_id < 0 or args.num_workers < 1 or args.worker_id >= args.num_workers:
        raise ValueError("--worker-id must satisfy 0 <= worker_id < num_workers")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    cfg = load_config(args.config)
    stage = str(cfg.get("stage", "stream_stage1"))
    seed = int(cfg.get("seed", 42))
    global_reverse = bool(cfg.get("reverse_files", False))
    sources = list(cfg.get("sources") or [])
    if not sources:
        raise ValueError("Config requires non-empty sources")

    tok_cfg = dict(cfg.get("tokenizer") or {})
    tokenizer_id = str(tok_cfg.get("id", DEFAULT_TOKENIZER_ID))
    vocab_size = int(tok_cfg.get("vocab_size", DEFAULT_VOCAB_SIZE))
    max_doc_tokens = int(tok_cfg.get("max_doc_tokens", DEFAULT_MAX_DOC_TOKENS))
    hard_doc_limit = int(tok_cfg.get("hard_doc_limit", DEFAULT_HARD_DOC_LIMIT))
    expected_eos_id = (
        int(tok_cfg["eos_token_id"])
        if tok_cfg.get("eos_token_id") is not None
        else None
    )

    out_cfg = dict(cfg.get("output") or {})
    output_repo = str(out_cfg.get("repo_id", ""))
    raw_prefix = (
        args.path_prefix
        if args.path_prefix is not None
        else out_cfg.get("path_prefix", "")
    )
    path_prefix = str(raw_prefix or "").strip("/")
    shard_tokens = int(
        args.shard_tokens
        or out_cfg.get("shard_size_tokens")
        or out_cfg.get("shard_tokens", DEFAULT_SHARD_TOKENS)
    )
    if shard_tokens <= 0:
        raise ValueError("--shard-tokens must be positive")

    target_tokens = args.dry_run_tokens or args.target_tokens or cfg.get("target_tokens")
    if target_tokens is not None:
        target_tokens = int(target_tokens)
        # Partition global target_tokens across workers so sum(worker budgets) == target_tokens
        if args.dry_run_tokens is None and args.target_tokens is None and args.num_workers > 1:
            target_tokens = compute_worker_target_tokens(
                target_tokens, args.worker_id, args.num_workers
            )

    token = optional_hf_token()
    if not args.no_upload and not token:
        raise RuntimeError("HF_TOKEN is required unless --no-upload is set")

    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi(token=token)
    hfs = HfFileSystem(token=token)
    if not args.no_upload:
        if not output_repo:
            raise ValueError("output.repo_id is required when uploading")
        api.create_repo(
            repo_id=output_repo,
            repo_type="dataset",
            private=bool(out_cfg.get("private", True)),
            exist_ok=True,
            token=token,
        )

    state_rel_path = f"_state/{stage}_w{args.worker_id:02d}.json"
    local_state_path = tmp_dir / "state" / f"{stage}_w{args.worker_id:02d}.json"
    shards_root = tmp_dir / "shards"
    shards_dir = shards_root / path_prefix if path_prefix else shards_root
    shards_dir.mkdir(parents=True, exist_ok=True)
    prefetch_dir = tmp_dir / "prefetch"

    state = load_resume_state(
        local_state_path=local_state_path,
        remote_repo=output_repo or None,
        remote_state_path=state_rel_path,
        api=api,
        token=token,
        no_upload=args.no_upload,
        resume=args.resume,
        sources=sources,
    )

    domain_files: dict[str, list[RemoteFileEntry]] = {}
    for src in sources:
        name = str(src["name"])
        _, files = discover_source_files(
            api,
            src,
            global_reverse=global_reverse,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            token=token,
        )
        domain_files[name] = files
        print(
            f"Source {name}: {len(files)} assigned files for worker {args.worker_id}/{args.num_workers}",
            flush=True,
        )

    prefetcher = SingleFilePrefetcher(prefetch_dir, token)
    readers: dict[str, DomainBatchReader] = {
        str(src["name"]): DomainBatchReader(
            src,
            domain_files[str(src["name"])],
            state["cursors"][str(src["name"])],
            args.batch_size,
            prefetcher,
            hfs,
            token,
        )
        for src in sources
    }

    tokens_by_domain: dict[str, int] = {
        name: int(c.get("tokens_emitted", 0)) for name, c in state["cursors"].items()
    }
    active_cursors: dict[str, dict[str, Any]] = {
        name: dict(c) for name, c in state["cursors"].items()
    }
    exhausted: set[str] = {
        name for name, r in readers.items() if r.exhausted
    }

    started_wall = time.monotonic()
    deadline_wall = started_wall + float(args.max_runtime_minutes) * 60.0
    shard_idx = int(state["next_shard_idx"])
    total_tokens_committed = int(state["total_tokens"])
    shards_written_this_run = 0

    shard_chunks: list[np.ndarray] = []
    shard_doc_lengths: list[np.ndarray] = []
    shard_tokens_acc = 0

    # Per-domain ordered queue of in-flight tokenized batches: (Future, end_cursor)
    domain_pending: dict[str, deque[tuple[Future[Any], dict[str, Any]]]] = {
        str(src["name"]): deque() for src in sources
    }

    uploader: AsyncShardUploader | None = None

    def total_inflight() -> int:
        return sum(len(q) for q in domain_pending.values())

    def submit_for_domain(executor: ProcessPoolExecutor, src: dict[str, Any]) -> bool:
        name = str(src["name"])
        reader = readers[name]
        if reader.exhausted:
            return False
        nxt = reader.next_batch()
        if nxt is None:
            return False
        raw_rows, end_cursor = nxt
        fut = executor.submit(_worker_process_batch, (raw_rows, src))
        domain_pending[name].append((fut, end_cursor))
        return True

    def fill_pipeline(executor: ProcessPoolExecutor, max_inflight: int) -> None:
        if time.monotonic() >= deadline_wall:
            return
        # Prioritize domains with fewest pending items among those needing work
        while total_inflight() < max_inflight:
            progressed = False
            # Order by deficit + current pending queue length so workers stay fed
            active_srcs = [
                src
                for src in sources
                if not readers[str(src["name"])].exhausted
                and len(domain_pending[str(src["name"])]) < 2
            ]
            if not active_srcs:
                break
            total_w = sum(float(s["weight"]) for s in active_srcs)
            active_srcs.sort(
                key=lambda s: (
                    (tokens_by_domain[str(s["name"])] + len(domain_pending[str(s["name"])]) * args.batch_size * 500)
                    / (float(s["weight"]) / total_w),
                    _tie_rank(seed, str(s["name"])),
                )
            )
            for src in active_srcs:
                if total_inflight() >= max_inflight:
                    break
                if submit_for_domain(executor, src):
                    progressed = True
            if not progressed:
                break

    def flush_current_shard() -> None:
        nonlocal shard_idx, total_tokens_committed, shard_tokens_acc, shards_written_this_run
        if shard_tokens_acc <= 0 or not shard_chunks:
            return
        assert uploader is not None
        shard_id = f"{stage}_w{args.worker_id:02d}_shard_{shard_idx:05d}"
        bin_path = shards_dir / f"{shard_id}.bin"
        manifest_path = shards_dir / f"{shard_id}.manifest.json"
        remote_data_path = f"{path_prefix}/{shard_id}.bin" if path_prefix else f"{shard_id}.bin"
        remote_manifest_path = (
            f"{path_prefix}/{shard_id}.manifest.json"
            if path_prefix
            else f"{shard_id}.manifest.json"
        )
        t_cnt, t_min, t_max, f_size, sha = write_shuffled_shard(
            chunks=shard_chunks,
            doc_lengths=shard_doc_lengths,
            shard_path=bin_path,
            shard_idx=shard_idx,
            worker_id=args.worker_id,
        )
        manifest = build_explicit_v1_manifest(
            shard_id=shard_id,
            data_path=remote_data_path,
            token_count=t_cnt,
            token_id_min=t_min,
            token_id_max=t_max,
            vocab_size=vocab_size,
            file_size_bytes=f_size,
            sha256=sha,
        )
        total_tokens_committed += t_cnt
        shard_idx += 1
        shards_written_this_run += 1

        snapshot_cursors = {
            name: {
                **c,
                "tokens_emitted": int(tokens_by_domain[name]),
                "exhausted": bool(name in exhausted and not domain_pending[name]),
            }
            for name, c in active_cursors.items()
        }
        state_snapshot = {
            "version": 1,
            "stage": stage,
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "next_shard_idx": shard_idx,
            "total_tokens": total_tokens_committed,
            "cursors": snapshot_cursors,
            "updated_at": time.time(),
        }
        uploader.submit(
            UploadJob(
                shard_id=shard_id,
                bin_path=bin_path,
                manifest_path=manifest_path,
                remote_manifest_path=remote_manifest_path,
                manifest_dict=manifest,
                state_path=local_state_path,
                remote_state_path=state_rel_path,
                state_snapshot=state_snapshot,
            )
        )
        shard_chunks.clear()
        shard_doc_lengths.clear()
        shard_tokens_acc = 0

    max_inflight = max(2, args.workers * 2, len(sources))
    stop_reason = "completed"

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(
            tokenizer_id,
            vocab_size,
            max_doc_tokens,
            hard_doc_limit,
            expected_eos_id,
        ),
    ) as executor:
        # Warm up workers before starting background uploader thread to avoid multi-threaded fork warnings.
        executor.submit(int, 0).result()
        uploader = AsyncShardUploader(
            api=api,
            repo_id=output_repo,
            token=token,
            tmp_dir=tmp_dir,
            no_upload=args.no_upload,
        )
        try:
            while True:
                if args.max_shards is not None and shards_written_this_run >= args.max_shards:
                    stop_reason = "max_shards"
                    break
                if target_tokens is not None and (total_tokens_committed + shard_tokens_acc) >= target_tokens:
                    stop_reason = "target_tokens"
                    break
                if time.monotonic() >= deadline_wall:
                    stop_reason = "max_runtime"
                    break

                fill_pipeline(executor, max_inflight)

                # Update exhausted status only when both reader and pending queue are empty
                available_sources = [
                    s
                    for s in sources
                    if not (readers[str(s["name"])].exhausted and not domain_pending[str(s["name"])])
                ]
                if not available_sources:
                    stop_reason = "sources_exhausted"
                    break

                exhausted_now = {
                    str(s["name"])
                    for s in sources
                    if readers[str(s["name"])].exhausted and not domain_pending[str(s["name"])]
                }
                exhausted.update(exhausted_now)

                chosen = choose_deficit_domain(sources, tokens_by_domain, exhausted, seed)
                d_name = str(chosen["name"])
                if not domain_pending[d_name]:
                    if not submit_for_domain(executor, chosen):
                        exhausted.add(d_name)
                        continue

                fut, end_cursor = domain_pending[d_name].popleft()
                tokens_u16, doc_lens, _stats = fut.result()
                if tokens_u16.size > 0:
                    shard_chunks.append(tokens_u16)
                    shard_doc_lengths.append(doc_lens)
                    shard_tokens_acc += int(tokens_u16.size)
                    tokens_by_domain[d_name] += int(tokens_u16.size)
                active_cursors[d_name] = end_cursor
                if end_cursor.get("exhausted") and not domain_pending[d_name]:
                    exhausted.add(d_name)

                if shard_tokens_acc >= shard_tokens:
                    flush_current_shard()

            # On graceful timeout, drain all already-submitted in-flight worker batches cleanly
            if stop_reason == "max_runtime":
                for d_name, q in domain_pending.items():
                    while q:
                        fut, end_cursor = q.popleft()
                        tokens_u16, doc_lens, _ = fut.result()
                        if tokens_u16.size > 0:
                            shard_chunks.append(tokens_u16)
                            shard_doc_lengths.append(doc_lens)
                            shard_tokens_acc += int(tokens_u16.size)
                            tokens_by_domain[d_name] += int(tokens_u16.size)
                        active_cursors[d_name] = end_cursor
            else:
                # Cancel unconsumed prefetched futures without advancing active_cursors
                for q in domain_pending.values():
                    while q:
                        fut, _ = q.popleft()
                        fut.cancel()

            flush_current_shard()
        finally:
            for r in readers.values():
                r.close()
            prefetcher.close()

    uploader.finish()
    elapsed = max(time.monotonic() - started_wall, 1e-9)
    tokens_this_run = total_tokens_committed - int(state["total_tokens"])
    print(
        f"Done ({stop_reason}) | shards_this_run={shards_written_this_run} | "
        f"tokens_this_run={tokens_this_run:,} | total_tokens={total_tokens_committed:,} | "
        f"elapsed={elapsed:.1f}s | speed={tokens_this_run / elapsed:,.0f} tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
