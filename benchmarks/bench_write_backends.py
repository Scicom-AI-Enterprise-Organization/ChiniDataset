"""Write backend benchmark: ChiniDataset (PyArrow direct) vs Pandas (PyArrow) vs Pandas (fastparquet) vs Polars (Rust).

Usage:
    uv run python benchmarks/bench_write_backends.py
"""

import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent / "output" / "write_backends"
PARQUET_URL = "hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet"
VOCAB: dict[str, int] = {}
UNK_ID = 0
COLUMNS = {"input_ids": "uint32[]", "labels": "uint32[]"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> np.ndarray:
    return np.array([VOCAB.get(w, UNK_ID) for w in text.split()], dtype=np.uint32)


def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# 1. ChiniDataset ParquetWriter (PyArrow direct)
# ---------------------------------------------------------------------------

def bench_chinidataset(ds):
    from chinidataset import ParquetWriter

    p = BASE / "chinidataset"
    clean_dir(p)

    t0 = time.perf_counter()
    with ParquetWriter(out=str(p), columns=COLUMNS, exist_ok=True) as w:
        for row in tqdm(ds, desc="  ChiniDataset", leave=False):
            ids = tokenize(row["text"])
            w.write({"input_ids": ids, "labels": ids})
    elapsed = time.perf_counter() - t0

    return {"time": elapsed, "size": dir_size(p)}


# ---------------------------------------------------------------------------
# 2. Pandas + PyArrow engine
# ---------------------------------------------------------------------------

def bench_pandas_pyarrow(ds):
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = BASE / "pandas_pyarrow"
    clean_dir(p)

    t0 = time.perf_counter()

    # Buffer rows, flush every ~64MB
    buf = []
    shard_idx = 0
    buf_bytes = 0

    for row in tqdm(ds, desc="  Pandas+PyArrow", leave=False):
        ids = tokenize(row["text"])
        buf.append({"input_ids": ids, "labels": ids})
        buf_bytes += ids.nbytes * 2
        if buf_bytes >= (1 << 26):  # 64MB
            pdf = pd.DataFrame(buf)
            pdf.to_parquet(p / f"shard.{shard_idx:05}.parquet", engine="pyarrow", index=False)
            shard_idx += 1
            buf.clear()
            buf_bytes = 0

    if buf:
        pdf = pd.DataFrame(buf)
        pdf.to_parquet(p / f"shard.{shard_idx:05}.parquet", engine="pyarrow", index=False)

    elapsed = time.perf_counter() - t0

    return {"time": elapsed, "size": dir_size(p)}


# ---------------------------------------------------------------------------
# 3. Pandas + fastparquet engine
# ---------------------------------------------------------------------------

def bench_pandas_fastparquet(ds):
    p = BASE / "pandas_fastparquet"
    clean_dir(p)

    t0 = time.perf_counter()

    buf = []
    shard_idx = 0
    buf_bytes = 0

    for row in tqdm(ds, desc="  Pandas+fastparquet", leave=False):
        ids = tokenize(row["text"])
        # fastparquet needs plain lists, not numpy arrays
        ids_list = ids.tolist()
        buf.append({"input_ids": ids_list, "labels": ids_list})
        buf_bytes += len(ids_list) * 4 * 2
        if buf_bytes >= (1 << 26):
            pdf = pd.DataFrame(buf)
            pdf.to_parquet(p / f"shard.{shard_idx:05}.parquet", engine="fastparquet", index=False)
            shard_idx += 1
            buf.clear()
            buf_bytes = 0

    if buf:
        pdf = pd.DataFrame(buf)
        pdf.to_parquet(p / f"shard.{shard_idx:05}.parquet", engine="fastparquet", index=False)

    elapsed = time.perf_counter() - t0

    return {"time": elapsed, "size": dir_size(p)}


# ---------------------------------------------------------------------------
# 4. Polars (Rust engine)
# ---------------------------------------------------------------------------

def bench_polars(ds):
    import polars as pl

    p = BASE / "polars"
    clean_dir(p)

    t0 = time.perf_counter()

    buf_ids = []
    buf_labels = []
    shard_idx = 0
    buf_bytes = 0

    for row in tqdm(ds, desc="  Polars", leave=False):
        ids = tokenize(row["text"])
        ids_list = ids.tolist()
        buf_ids.append(ids_list)
        buf_labels.append(ids_list)
        buf_bytes += len(ids_list) * 4 * 2
        if buf_bytes >= (1 << 26):
            pldf = pl.DataFrame({"input_ids": buf_ids, "labels": buf_labels})
            pldf.write_parquet(p / f"shard.{shard_idx:05}.parquet")
            shard_idx += 1
            buf_ids.clear()
            buf_labels.clear()
            buf_bytes = 0

    if buf_ids:
        pldf = pl.DataFrame({"input_ids": buf_ids, "labels": buf_labels})
        pldf.write_parquet(p / f"shard.{shard_idx:05}.parquet")

    elapsed = time.perf_counter() - t0

    return {"time": elapsed, "size": dir_size(p)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global VOCAB

    from datasets import load_dataset

    BASE.mkdir(parents=True, exist_ok=True)

    print("Loading wikipedia shard 0 ...")
    ds = load_dataset("parquet", data_files=PARQUET_URL, split="train")
    N = len(ds)
    print(f"  {N:,} articles")

    print("Building vocab ...")
    counter: Counter[str] = Counter()
    for row in tqdm(ds, desc="  words"):
        counter.update(row["text"].split())
    VOCAB = {"<unk>": 0}
    for i, (w, _) in enumerate(counter.most_common(49_999), 1):
        VOCAB[w] = i
    print(f"  {len(VOCAB):,} words\n")

    # --- Run benchmarks ---
    results = {}

    print("1/4  ChiniDataset ParquetWriter (PyArrow direct)")
    results["chinidataset"] = bench_chinidataset(ds)

    print("2/4  Pandas to_parquet (PyArrow engine)")
    results["pandas_pyarrow"] = bench_pandas_pyarrow(ds)

    print("3/4  Pandas to_parquet (fastparquet engine)")
    results["pandas_fastparquet"] = bench_pandas_fastparquet(ds)

    print("4/4  Polars write_parquet (Rust engine)")
    results["polars"] = bench_polars(ds)

    # --- Print results ---
    base_time = results["chinidataset"]["time"]

    print(f"\n{'=' * 80}")
    print(f" Write Backend Benchmark — {N:,} articles, uint32[] arrays")
    print(f"{'=' * 80}")
    print(f"  {'Writer':<35s} | {'Time':>7s} | {'Samples/s':>12s} | {'Size':>10s} | {'vs Chini':>8s}")
    print(f"  {'-' * 35}-+-{'-' * 7}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}")

    labels = {
        "chinidataset":       "ChiniDataset (PyArrow direct)",
        "pandas_pyarrow":     "Pandas + PyArrow",
        "pandas_fastparquet": "Pandas + fastparquet",
        "polars":             "Polars (Rust)",
    }

    for key, label in labels.items():
        r = results[key]
        t = r["time"]
        throughput = N / t
        ratio = base_time / t
        print(f"  {label:<35s} | {t:6.1f}s | {throughput:>10,.0f}/s | {human_bytes(r['size']):>10s} | {ratio:>7.2f}x")

    print(f"  {'-' * 35}-+-{'-' * 7}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}")
    print()


if __name__ == "__main__":
    main()
