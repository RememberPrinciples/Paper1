#!/usr/bin/env python3
"""Download Qwen2.5-0.5B-Instruct with automatic fallback on slow links."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ROOT = Path("/root/autodl-tmp/Model")
LOCAL_DIR = MODEL_ROOT / "Qwen2.5-0.5B-Instruct"
MIN_MBPS = 1.0
REPORT_INTERVAL = 20
LOW_SPEED_LIMIT = 3

STRATEGIES = [
    ("hf_mirror_w8", "https://hf-mirror.com", 8),
    ("hf_mirror_w4", "https://hf-mirror.com", 4),
    ("hf_mirror_w1", "https://hf-mirror.com", 1),
    ("hf_official_w4", "https://huggingface.co", 4),
    ("hf_official_w1", "https://huggingface.co", 1),
]

IGNORE_PATTERNS = [
    "*.bin",
    "*.msgpack",
    "*.h5",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
    "trainer_state.json",
    "rng_state.pth",
]


def fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for unit in units:
        if abs(v) < 1024 or unit == units[-1]:
            return f"{v:.2f}{unit}"
        v /= 1024
    return f"{v:.2f}TB"


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def has_core_files(path: Path) -> bool:
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    return all((path / name).exists() for name in required) and any(path.glob("*.safetensors"))


def repo_total_size(repo_id: str, endpoint: str) -> int:
    api = HfApi(endpoint=endpoint)
    info = api.model_info(repo_id, files_metadata=True)
    total = 0
    for sibling in info.siblings:
        name = sibling.rfilename
        if any(Path(name).match(pattern) for pattern in IGNORE_PATTERNS):
            continue
        total += sibling.size or 0
    return total


def start_download(repo_id: str, local_dir: Path, endpoint: str, workers: int) -> subprocess.Popen[str]:
    code = """
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
workers = int(sys.argv[3])
ignore_patterns = sys.argv[4].split("\\n") if sys.argv[4] else None

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    ignore_patterns=ignore_patterns,
    resume_download=True,
    max_workers=workers,
)
"""
    env = os.environ.copy()
    env["HF_ENDPOINT"] = endpoint
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return subprocess.Popen(
        [sys.executable, "-u", "-c", code, repo_id, str(local_dir), str(workers), "\n".join(IGNORE_PATTERNS)],
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def drain_output(proc: subprocess.Popen[str]) -> list[str]:
    lines: list[str] = []
    if proc.stdout is None:
        return lines
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
    return lines


def terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def monitor_attempt(total: int, strategy: tuple[str, str, int]) -> bool:
    name, endpoint, workers = strategy
    print(f"ATTEMPT_START strategy={name} endpoint={endpoint} workers={workers}", flush=True)
    before = tree_size(LOCAL_DIR)
    proc = start_download(REPO_ID, LOCAL_DIR, endpoint, workers)
    start = time.time()
    last_t = start
    last_size = before
    low_speed_count = 0

    while True:
        time.sleep(REPORT_INTERVAL)
        for line in drain_output(proc):
            print(f"CHILD {line}", flush=True)

        now = time.time()
        cur_size = tree_size(LOCAL_DIR)
        interval_speed = (cur_size - last_size) / max(now - last_t, 1)
        avg_speed = (cur_size - before) / max(now - start, 1)
        remaining = max(total - cur_size, 0)
        eta = remaining / avg_speed if avg_speed > 0 else float("inf")
        print(
            "PROGRESS "
            f"strategy={name} downloaded={fmt_bytes(cur_size)} total={fmt_bytes(total)} "
            f"interval_mbps={interval_speed / 1024 / 1024:.2f} "
            f"avg_mbps={avg_speed / 1024 / 1024:.2f} eta_min={eta / 60:.1f}",
            flush=True,
        )

        if proc.poll() is not None:
            for line in drain_output(proc):
                print(f"CHILD {line}", flush=True)
            if proc.returncode == 0 and has_core_files(LOCAL_DIR):
                print(f"ATTEMPT_DONE strategy={name} size={fmt_bytes(tree_size(LOCAL_DIR))}", flush=True)
                return True
            print(f"ATTEMPT_FAIL strategy={name} returncode={proc.returncode}", flush=True)
            return False

        speed_mbps = interval_speed / 1024 / 1024
        if speed_mbps < MIN_MBPS:
            low_speed_count += 1
        else:
            low_speed_count = 0
        if low_speed_count >= LOW_SPEED_LIMIT:
            print(
                f"LOW_SPEED_ABORT strategy={name} low_speed_count={low_speed_count} "
                f"threshold_mbps={MIN_MBPS}",
                flush=True,
            )
            terminate(proc)
            return False

        last_t = now
        last_size = cur_size


def main() -> int:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if has_core_files(LOCAL_DIR):
        print(f"ALREADY_PRESENT local_dir={LOCAL_DIR} size={fmt_bytes(tree_size(LOCAL_DIR))}", flush=True)
        return 0

    total = None
    for _, endpoint, _ in STRATEGIES:
        try:
            total = repo_total_size(REPO_ID, endpoint)
            print(f"MODEL_SIZE endpoint={endpoint} total={fmt_bytes(total)}", flush=True)
            break
        except Exception as exc:
            print(f"SIZE_QUERY_FAIL endpoint={endpoint} error={type(exc).__name__}:{exc}", flush=True)
    if total is None:
        print("MODEL_FAIL reason=size_query_failed", flush=True)
        return 2

    for strategy in STRATEGIES:
        if monitor_attempt(total, strategy):
            print(f"MODEL_DONE repo={REPO_ID} local_dir={LOCAL_DIR}", flush=True)
            return 0

    print("MODEL_FAIL reason=all_strategies_failed", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
