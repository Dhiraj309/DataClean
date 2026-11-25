#!/usr/bin/env python3
"""Standalone tmux checkpoint watcher: uploads completed checkpoints to HF Hub asynchronously."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

from huggingface_hub import HfApi


_STOP_REQUESTED = False


def _handle_signal(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _snapshot_sizes(directory: Path) -> dict[str, int] | None:
    sizes: dict[str, int] = {}
    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        if p.name.endswith(".tmp") or p.name.endswith(".partial"):
            return None
        try:
            sizes[str(p.relative_to(directory))] = p.stat().st_size
        except OSError:
            return None
    return sizes


def is_checkpoint_ready(directory: Path, settle_seconds: float = 5.0) -> bool:
    if not directory.is_dir():
        return False
    has_marker = (
        (directory / "trainer_state.pt").is_file()
        or (directory / "manifest.json").is_file()
        or (directory / "model.safetensors").is_file()
        or (directory / "pytorch_model.bin").is_file()
        or (directory / "config.json").is_file()
    )
    if not has_marker:
        return False
    first = _snapshot_sizes(directory)
    if not first:
        return False
    time.sleep(settle_seconds)
    second = _snapshot_sizes(directory)
    return second is not None and first == second


def upload_folder_with_backoff(
    api: HfApi,
    *,
    folder_path: Path,
    repo_id: str,
    path_in_repo: str,
    token: str,
    max_attempts: int = 8,
) -> None:
    for attempt in range(max_attempts):
        try:
            api.upload_folder(
                folder_path=str(folder_path),
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=path_in_repo,
                commit_message=f"Upload {path_in_repo}",
                token=token,
            )
            return
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status in (412, 429) or (isinstance(status, int) and 500 <= status < 600)
            if not retryable:
                msg = str(exc)
                retryable = any(c in msg for c in ("412", "429", "500", "502", "503", "504", "timed out"))
            if attempt + 1 >= max_attempts or not retryable:
                raise
            delay = min(180.0, float(2 ** attempt)) * random.uniform(0.7, 1.3)
            print(
                f"Retry {attempt + 1}/{max_attempts} uploading {path_in_repo}: {exc} ({delay:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def load_uploaded_log(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def save_uploaded_log(path: Path, uploaded: set[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(uploaded), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default="Dhiraj309/laughlm-135m-deep-narrow-v2")
    parser.add_argument("--remote-prefix", default="checkpoints")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    output_dir = Path(args.output_dir).expanduser().resolve()
    state_file = output_dir / ".uploaded_checkpoints.json"
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    uploaded = load_uploaded_log(state_file)

    print(f"Watching {output_dir} -> hf://{args.repo_id}/{args.remote_prefix}", flush=True)
    while not _STOP_REQUESTED:
        if output_dir.is_dir():
            candidates = sorted(
                [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
                key=lambda p: int(p.name.removeprefix("checkpoint-")) if p.name.removeprefix("checkpoint-").isdecimal() else 0,
            )
            final_dir = output_dir / "final"
            if final_dir.is_dir():
                candidates.append(final_dir)
            for ckpt in candidates:
                if ckpt.name in uploaded:
                    continue
                if not is_checkpoint_ready(ckpt, settle_seconds=args.settle_seconds):
                    continue
                remote_path = f"{args.remote_prefix.strip('/')}/{ckpt.name}" if args.remote_prefix.strip("/") else ckpt.name
                print(f"Uploading {ckpt} -> {args.repo_id}/{remote_path} ...", flush=True)
                upload_folder_with_backoff(
                    api,
                    folder_path=ckpt,
                    repo_id=args.repo_id,
                    path_in_repo=remote_path,
                    token=token,
                )
                uploaded.add(ckpt.name)
                save_uploaded_log(state_file, uploaded)
                print(f"Uploaded {ckpt.name}", flush=True)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
