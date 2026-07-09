#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path("/root/autodl-tmp/Model")
MIN_MBPS = 2.0
REPORT_INTERVAL = 60
LOW_SPEED_LIMIT = 3

MODELS = [
    ("TinyLlama/TinyLlama_v1.1", MODEL_ROOT / "TinyLlama-1.1B-Draft"),
    ("princeton-nlp/Sheared-LLaMA-1.3B", MODEL_ROOT / "Sheared-LLaMA-1.3B-Draft"),
    ("princeton-nlp/Sheared-LLaMA-2.7B", MODEL_ROOT / "Sheared-LLaMA-2.7B-Draft"),
]

STRATEGIES = [
    ("mirror_w4", "https://hf-mirror.com", 4),
    ("mirror_w8", "https://hf-mirror.com", 8),
    ("mirror_w1", "https://hf-mirror.com", 1),
    ("official_w4", "https://huggingface.co", 4),
]

IGNORE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
    "trainer_state.json",
    "rng_state.pth",
]


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if abs(v) < 1024 or u == units[-1]:
            return f"{v:.2f}{u}"
        v /= 1024
    return f"{v:.2f}TB"


def repo_total_size(repo_id: str, endpoint: str) -> int:
    api = HfApi(endpoint=endpoint)
    info = api.model_info(repo_id, files_metadata=True)
    total = 0
    for sibling in info.siblings:
        name = sibling.rfilename
        if name in IGNORE_PATTERNS:
            continue
        total += sibling.size or 0
    return total


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def start_download(repo_id: str, local_dir: Path, endpoint: str, workers: int) -> subprocess.Popen:
    code = r"""
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
workers = int(sys.argv[3])
ignore_patterns = sys.argv[4].split("\n") if sys.argv[4] else None

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
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def drain_output(proc: subprocess.Popen) -> list[str]:
    lines: list[str] = []
    if proc.stdout is None:
        return lines
    while True:
        # TextIOWrapper has no reliable nonblocking readline here; use select on fd.
        import select

        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
    return lines


def monitor_attempt(repo_id: str, local_dir: Path, total: int, strategy: tuple[str, str, int]) -> bool:
    name, endpoint, workers = strategy
    print(f"ATTEMPT_START repo={repo_id} strategy={name} endpoint={endpoint} workers={workers}", flush=True)
    proc = start_download(repo_id, local_dir, endpoint, workers)
    start = time.time()
    last_t = start
    last_size = tree_size(local_dir)
    low_speed_count = 0

    while True:
        time.sleep(REPORT_INTERVAL)
        for line in drain_output(proc):
            print(f"CHILD {line}", flush=True)

        now = time.time()
        cur_size = tree_size(local_dir)
        interval_speed = (cur_size - last_size) / max(now - last_t, 1)
        avg_speed = (cur_size - tree_size(local_dir / "__nonexistent__")) / max(now - start, 1)
        remaining = max(total - cur_size, 0)
        eta = remaining / avg_speed if avg_speed > 0 else float("inf")

        print(
            "PROGRESS "
            f"repo={repo_id} strategy={name} "
            f"downloaded={fmt_bytes(cur_size)} total={fmt_bytes(total)} "
            f"interval_mbps={interval_speed / 1024 / 1024:.2f} "
            f"avg_mbps={avg_speed / 1024 / 1024:.2f} "
            f"eta_min={(eta / 60):.1f}",
            flush=True,
        )

        if proc.poll() is not None:
            for line in drain_output(proc):
                print(f"CHILD {line}", flush=True)
            if proc.returncode == 0:
                final_size = tree_size(local_dir)
                print(f"ATTEMPT_DONE repo={repo_id} strategy={name} size={fmt_bytes(final_size)}", flush=True)
                return True
            print(f"ATTEMPT_FAIL repo={repo_id} strategy={name} returncode={proc.returncode}", flush=True)
            return False

        if interval_speed / 1024 / 1024 < MIN_MBPS:
            low_speed_count += 1
        else:
            low_speed_count = 0

        elapsed = now - start
        if elapsed >= REPORT_INTERVAL * LOW_SPEED_LIMIT and low_speed_count >= LOW_SPEED_LIMIT:
            print(
                f"LOW_SPEED_ABORT repo={repo_id} strategy={name} "
                f"low_speed_count={low_speed_count} threshold_mbps={MIN_MBPS}",
                flush=True,
            )
            terminate(proc)
            return False

        last_t = now
        last_size = cur_size


def main() -> int:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for repo_id, local_dir in MODELS:
        print(f"MODEL_START repo={repo_id} local_dir={local_dir}", flush=True)
        total = None
        for _, endpoint, _ in STRATEGIES:
            try:
                total = repo_total_size(repo_id, endpoint)
                break
            except Exception as exc:
                print(f"SIZE_QUERY_FAIL repo={repo_id} endpoint={endpoint} error={type(exc).__name__}:{exc}", flush=True)
        if total is None:
            print(f"MODEL_FAIL repo={repo_id} reason=size_query_failed", flush=True)
            return 2
        print(f"MODEL_SIZE repo={repo_id} total={fmt_bytes(total)}", flush=True)

        ok = False
        for strategy in STRATEGIES:
            ok = monitor_attempt(repo_id, local_dir, total, strategy)
            if ok:
                break
        if not ok:
            print(f"MODEL_FAIL repo={repo_id} reason=all_strategies_failed", flush=True)
            return 3
        print(f"MODEL_DONE repo={repo_id} local_dir={local_dir}", flush=True)

    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
