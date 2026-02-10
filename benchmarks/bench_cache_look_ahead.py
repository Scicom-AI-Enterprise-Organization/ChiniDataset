"""Benchmark: Look-ahead cache speedup on Wikipedia EN shard 0.

Downloads Wikipedia EN shard 0, builds a word-level vocabulary,
tokenises every article to uint32 arrays, writes to ChiniDataset
Parquet shards, then benchmarks iteration with different ``look_ahead``
values (0, 1, 2, 4).

Usage:
    uv run python benchmarks/bench_cache_look_ahead.py
    python  benchmarks/bench_cache_look_ahead.py          # if already in venv
"""

import gc
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from chinidataset import ParquetWriter, StreamingDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent / "output" / "look_ahead"
DATA_DIR = BASE / "data"
PARQUET_URL = "hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet"

LOOK_AHEAD_VALUES = [0, 1, 2, 4, 10]
REPEATS = 4  # first run is warmup, rest are averaged

COLUMNS = {"input_ids": "uint32[]", "labels": "uint32[]"}
VOCAB_SIZE = 50_000  # top-k words to keep

VOCAB: dict[str, int] = {}
UNK_ID = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> np.ndarray:
    """Whitespace-split tokenizer backed by a word vocabulary."""
    return np.array([VOCAB.get(w, UNK_ID) for w in text.split()], dtype=np.uint32)


def human_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# Data preparation — Wikipedia shard 0 + word vocab tokenizer
# ---------------------------------------------------------------------------


def prepare_wikipedia():
    """Load Wikipedia EN shard 0, build word vocab, return HF dataset."""
    global VOCAB
    from datasets import load_dataset

    print("Loading Wikipedia EN shard 0 ...")
    ds = load_dataset("parquet", data_files=PARQUET_URL, split="train")
    N = len(ds)
    print(f"  {N:,} articles")

    print("Building word vocabulary ...")
    counter: Counter[str] = Counter()
    for row in tqdm(ds, desc="  counting words"):
        counter.update(row["text"].split())

    VOCAB = {"<unk>": UNK_ID}
    for i, (w, _) in enumerate(counter.most_common(VOCAB_SIZE - 1), 1):
        VOCAB[w] = i
    print(f"  {len(VOCAB):,} words in vocab")
    return ds


def write_chinidataset(ds) -> Path:
    """Tokenise Wikipedia and write to ChiniDataset Parquet shards."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Tokenising & writing ChiniDataset shards ...")
    with ParquetWriter(
        out=str(DATA_DIR),
        columns=COLUMNS,
        exist_ok=True,
    ) as w:
        for row in tqdm(ds, desc="  writing", leave=False):
            ids = tokenize(row["text"])
            w.write({"input_ids": ids, "labels": ids})

    n_shards = len(list(DATA_DIR.glob("shard.*")))
    sz = human_bytes(dir_size(DATA_DIR))
    print(f"  Written: {n_shards} shards, {sz}")
    return DATA_DIR


def prepare_data() -> Path:
    """Download, tokenise, and write — or reuse existing data."""
    if DATA_DIR.exists() and (DATA_DIR / "index.json").exists():
        with open(DATA_DIR / "index.json") as f:
            index = json.load(f)
        n_existing = sum(s["samples"] for s in index["shards"])
        n_shards = len(index["shards"])
        print(f"Reusing existing data  ({n_existing:,} samples, "
              f"{n_shards} shards in {DATA_DIR})")
        return DATA_DIR

    ds = prepare_wikipedia()
    data_dir = write_chinidataset(ds)
    del ds
    gc.collect()
    return data_dir


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_look_ahead(data_dir: Path, look_ahead: int) -> dict:
    """Iterate the full dataset and return timing statistics."""
    times: list[float] = []
    count = 0

    for run in range(REPEATS):
        ds = StreamingDataset(
            local=str(data_dir),
            look_ahead=look_ahead,
            max_open_shards=max(8, look_ahead + 2),
            shuffle=False,
        )

        t0 = time.perf_counter()
        count = 0
        for sample in ds:
            _ = sample["input_ids"]
            count += 1
        elapsed = time.perf_counter() - t0

        label = "warmup" if run == 0 else f"run {run}"
        print(f"      {label}: {count:,} samples in {elapsed:.3f}s  "
              f"({count / elapsed:,.0f} samples/s)")
        times.append(elapsed)

        # Release readers between runs
        del ds
        gc.collect()

    # Skip warmup (first run)
    measured = times[1:]
    return {
        "look_ahead": look_ahead,
        "samples": count,
        "warmup": times[0],
        "times": measured,
        "avg": float(np.mean(measured)),
        "best": float(min(measured)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    data_dir = prepare_data()

    # Read back metadata for display
    with open(data_dir / "index.json") as f:
        index = json.load(f)
    n_shards = len(index["shards"])
    n_samples = sum(s["samples"] for s in index["shards"])
    sz = human_bytes(dir_size(data_dir))

    print(f"\n{'=' * 70}")
    print(f"  Look-Ahead Cache Benchmark")
    print(f"  Data: Wikipedia EN shard 0, word-vocab tokenised to uint32[]")
    print(f"  {n_samples:,} samples across {n_shards} shards ({sz})")
    print(f"  Repeats: {REPEATS} ({REPEATS - 1} measured + 1 warmup)")
    print(f"  look_ahead values: {LOOK_AHEAD_VALUES}")
    print(f"{'=' * 70}\n")

    results: list[dict] = []

    for la in LOOK_AHEAD_VALUES:
        print(f"  look_ahead={la}:")
        r = bench_look_ahead(data_dir, look_ahead=la)
        results.append(r)
        print()

    # -- Summary table --
    baseline = results[0]  # look_ahead=0
    n = baseline["samples"]

    print(f"{'=' * 70}")
    print(f"  Summary  ({n:,} samples, {n_shards} shards)")
    print(f"{'=' * 70}")
    print(f"  {'look_ahead':>12s}  {'Avg time':>10s}  {'Best time':>10s}  "
          f"{'Avg samp/s':>12s}  {'Speedup':>8s}")
    print(f"  {'-' * 62}")

    for r in results:
        speedup = baseline["avg"] / r["avg"] if r["avg"] else 0
        print(
            f"  {r['look_ahead']:>12d}  "
            f"{r['avg']:>9.3f}s  "
            f"{r['best']:>9.3f}s  "
            f"{n / r['avg']:>11,.0f}/s  "
            f"{speedup:>7.2f}x"
        )
    print()

    # Save results JSON
    results_file = BASE / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_file}")


if __name__ == "__main__":
    main()
