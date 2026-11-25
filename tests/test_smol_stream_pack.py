from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
import zstandard as zstd

import smol_stream_pack as sp


class DummyFastTokenizer:
    is_fast = True
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        # Encode each character's unique doc marker deterministically into token IDs in [1, 49000)
        out = []
        for text in texts:
            # Parse doc number if formatted like "doc-00012 ..."
            tokens = [1 + (ord(ch) % 200) for ch in text[:64]]
            out.append(tokens)
        return {"input_ids": out}


def _write_multi_rg_parquet(path: Path, docs: list[str], rows_per_rg: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    schema = pa.schema([("text", pa.string()), ("id", pa.string())])
    try:
        for i in range(0, len(docs), rows_per_rg):
            chunk = docs[i : i + rows_per_rg]
            table = pa.table(
                {
                    "text": chunk,
                    "id": [f"id-{i + j:04d}" for j in range(len(chunk))],
                },
                schema=schema,
            )
            if writer is None:
                writer = pq.ParquetWriter(path, schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _write_jsonl_zst(path: Path, docs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor()
    payload = "".join(json.dumps({"text": d, "id": f"z-{i}"}) + "\n" for i, d in enumerate(docs))
    path.write_bytes(cctx.compress(payload.encode("utf-8")))


def _read_u16_docs(bin_path: Path, eos_id: int = 0) -> list[tuple[int, ...]]:
    arr = np.fromfile(bin_path, dtype="<u2")
    docs: list[tuple[int, ...]] = []
    cur: list[int] = []
    for val in arr.tolist():
        cur.append(val)
        if val == eos_id:
            docs.append(tuple(cur))
            cur = []
    assert not cur, "Trailing tokens without EOS"
    return docs


def test_configure_environment_routes_caches_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    sp.configure_environment(tmp_path / "scratch")
    assert sp.os.environ["HF_HOME"] == str(tmp_path / "scratch" / "hf_cache")
    assert sp.os.environ["HF_XET_CACHE"] == str(tmp_path / "scratch" / "hf_cache" / "xet")
    captured = capsys.readouterr()
    assert "HF_XET_HIGH_PERFORMANCE=1" in captured.err


def test_discover_source_files_reverse_and_worker_partition(tmp_path: Path) -> None:
    files = [tmp_path / f"part-{i:02d}.parquet" for i in range(6)]
    for f in files:
        f.write_bytes(b"x")
    src_cfg = {"name": "demo", "local_files": [str(f) for f in files]}

    _, w0 = sp.discover_source_files(None, src_cfg, global_reverse=True, worker_id=0, num_workers=2, token=None)
    _, w1 = sp.discover_source_files(None, src_cfg, global_reverse=True, worker_id=1, num_workers=2, token=None)

    p0 = [Path(e.path).name for e in w0]
    p1 = [Path(e.path).name for e in w1]
    assert p0 == ["part-05.parquet", "part-03.parquet", "part-01.parquet"]
    assert p1 == ["part-04.parquet", "part-02.parquet", "part-00.parquet"]
    assert set(p0).isdisjoint(set(p1))


def test_midfile_parquet_and_jsonl_zst_o1_resume_zero_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create unique verifiable documents across Parquet (multiple row groups) and .jsonl.zst
    pq_docs = [f"PQDOC_{i:04d}_" + ("alpha " * 8) for i in range(24)]
    zst_docs = [f"ZSTDOC_{i:04d}_" + ("beta " * 8) for i in range(24)]

    pq_file0 = tmp_path / "data" / "pq_00.parquet"
    pq_file1 = tmp_path / "data" / "pq_01.parquet"
    _write_multi_rg_parquet(pq_file0, pq_docs[:12], rows_per_rg=4)
    _write_multi_rg_parquet(pq_file1, pq_docs[12:], rows_per_rg=4)

    zst_file = tmp_path / "zst" / "zst_00.jsonl.zst"
    _write_jsonl_zst(zst_file, zst_docs)

    # Patch worker initializer / tokenizer to avoid network download in unit test
    class UniqueDocTokenizer:
        is_fast = True
        eos_token_id = 0

        def __call__(self, texts, **kwargs):
            # Encode doc index uniquely in token ids [100..40000] so every doc is uniquely identifiable
            ids_list = []
            for t in texts:
                tag, idx_str, _ = t.split("_", 2)
                base = 1000 if tag == "PQDOC" else 20000
                doc_id = base + int(idx_str)
                # 15 tokens per doc (+1 EOS = 16 tokens)
                ids_list.append([doc_id] + [10 + (k % 50) for k in range(14)])
            return {"input_ids": ids_list}

    def fake_worker_init(*args):
        sp._WORKER_TOKENIZER = UniqueDocTokenizer()
        sp._WORKER_EOS_ID = 0
        sp._WORKER_VOCAB_SIZE = 49152
        sp._WORKER_MAX_DOC_TOKENS = 8192
        sp._WORKER_HARD_DOC_LIMIT = 65536

    monkeypatch.setattr(sp, "_worker_init", fake_worker_init)

    cfg = {
        "stage": "unit_stage",
        "seed": 7,
        "tokenizer": {"id": "dummy", "vocab_size": 49152, "max_doc_tokens": 8192, "hard_doc_limit": 65536},
        "output": {"repo_id": "owner/repo", "shard_tokens": 200},
        "sources": [
            {
                "name": "pq_src",
                "local_files": [str(pq_file0), str(pq_file1)],
                "weight": 0.5,
                "columns": {"text": "text", "id": "id"},
                "required_columns": ["text"],
                "filters": {"min_words": 3},
            },
            {
                "name": "zst_src",
                "local_files": [str(zst_file)],
                "weight": 0.5,
                "columns": {"text": "text", "id": "id"},
                "required_columns": ["text"],
                "filters": {"min_words": 3},
            },
        ],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    work_dir = tmp_path / "work"

    # Run 1: stop after shard 0 (batch_size=3 splits mid-row-group in Parquet AND mid-file in .jsonl.zst!)
    sp.main([
        "--config", str(cfg_path),
        "--tmp-dir", str(work_dir),
        "--workers", "1",
        "--batch-size", "3",
        "--shard-tokens", "200",
        "--max-shards", "1",
        "--no-upload",
    ])

    state_file = work_dir / "state" / "unit_stage_w00.json"
    assert state_file.is_file()
    state1 = json.loads(state_file.read_text(encoding="utf-8"))
    assert state1["next_shard_idx"] == 1

    # Run 2: resume from local state to produce shard 1
    sp.main([
        "--config", str(cfg_path),
        "--tmp-dir", str(work_dir),
        "--workers", "1",
        "--batch-size", "3",
        "--shard-tokens", "200",
        "--max-shards", "1",
        "--resume",
        "--no-upload",
    ])

    shard0_bin = work_dir / "shards" / "unit_stage_w00_shard_00000.bin"
    shard1_bin = work_dir / "shards" / "unit_stage_w00_shard_00001.bin"
    shard0_man = work_dir / "shards" / "unit_stage_w00_shard_00000.manifest.json"
    shard1_man = work_dir / "shards" / "unit_stage_w00_shard_00001.manifest.json"
    assert shard0_bin.is_file() and shard1_bin.is_file()
    assert shard0_man.is_file() and shard1_man.is_file()

    # Check zero overlapping documents between shard 0 and shard 1
    docs0 = _read_u16_docs(shard0_bin, eos_id=0)
    docs1 = _read_u16_docs(shard1_bin, eos_id=0)
    ids0 = {d[0] for d in docs0}
    ids1 = {d[0] for d in docs1}
    assert len(ids0) == len(docs0)
    assert len(ids1) == len(docs1)
    assert ids0.isdisjoint(ids1), f"Overlap found: {ids0 & ids1}"

    # Verify explicit_v1 manifest structure
    m0 = json.loads(shard0_man.read_text(encoding="utf-8"))
    assert m0["schema_version"] == 1
    assert m0["manifest_type"] == "packed_binary_shard"
    assert m0["compatibility_profile"] == "explicit_v1"
    assert m0["header_bytes"] == 0
    assert m0["token_dtype"] == "uint16"
    assert m0["byte_order"] == "little"
    assert m0["vocab_size"] == 49152
    assert m0["data_path"] == "unit_stage_w00_shard_00000.bin"
    assert m0["documents"] == {"storage": "unavailable"}


def test_graceful_max_runtime_flushes_partial_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = [f"DOC_{i:04d}_" + ("hello " * 10) for i in range(40)]
    pq_file = tmp_path / "data" / "p0.parquet"
    _write_multi_rg_parquet(pq_file, docs, rows_per_rg=10)

    class SlowDummyTokenizer(DummyFastTokenizer):
        def __call__(self, texts, **kwargs):
            sp.time.sleep(0.05)
            return super().__call__(texts, **kwargs)

    def fake_worker_init(*args):
        sp._WORKER_TOKENIZER = SlowDummyTokenizer()
        sp._WORKER_EOS_ID = 0
        sp._WORKER_VOCAB_SIZE = 49152
        sp._WORKER_MAX_DOC_TOKENS = 8192
        sp._WORKER_HARD_DOC_LIMIT = 65536

    monkeypatch.setattr(sp, "_worker_init", fake_worker_init)

    cfg = {
        "stage": "timeout_stage",
        "seed": 42,
        "tokenizer": {"id": "dummy", "vocab_size": 49152},
        "output": {"repo_id": "owner/repo", "shard_tokens": 10_000_000},
        "sources": [
            {
                "name": "pq_src",
                "local_files": [str(pq_file)],
                "weight": 1.0,
                "columns": {"text": "text", "id": "id"},
                "required_columns": ["text"],
                "filters": {"min_words": 3},
            }
        ],
    }
    cfg_path = tmp_path / "timeout_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    work_dir = tmp_path / "work"

    sp.main([
        "--config", str(cfg_path),
        "--tmp-dir", str(work_dir),
        "--workers", "1",
        "--batch-size", "4",
        "--max-runtime-minutes", "0.002",
        "--no-upload",
    ])

    shard_bin = work_dir / "shards" / "timeout_stage_w00_shard_00000.bin"
    shard_man = work_dir / "shards" / "timeout_stage_w00_shard_00000.manifest.json"
    assert shard_bin.is_file()
    assert shard_man.is_file()
    man = json.loads(shard_man.read_text(encoding="utf-8"))
    assert 0 < man["token_count"] < 10_000_000
    assert man["compatibility_profile"] == "explicit_v1"


def test_commit_with_backoff_retries_412_429_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp.time, "sleep", lambda _: None)
    attempts = {"n": 0}

    class FakeResp:
        def __init__(self, code: int):
            self.status_code = code

    class FakeExc(RuntimeError):
        def __init__(self, code: int):
            super().__init__(f"HTTP {code}")
            self.response = FakeResp(code)

    class FakeApi:
        def create_commit(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise FakeExc(412)
            if attempts["n"] == 2:
                raise FakeExc(429)
            if attempts["n"] == 3:
                raise FakeExc(503)
            return "ok"

    sp.commit_with_backoff(
        FakeApi(),
        repo_id="owner/repo",
        operations=[],
        commit_message="test",
        token="hf_test",
        max_attempts=5,
    )
    assert attempts["n"] == 4
