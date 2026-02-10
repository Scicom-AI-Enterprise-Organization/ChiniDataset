"""uv run python benchmarks/bench_write_mp.py"""

import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

NUM_WORKERS = 4
BASE = Path(__file__).resolve().parent / "output" / "write_mp"
PARQUET_URL = "hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet"

VOCAB: dict[str, int] = {}
UNK_ID = 0


def tokenize(text: str) -> np.ndarray:
    return np.array([VOCAB.get(w, UNK_ID) for w in text.split()], dtype=np.uint32)


def tokenize_for_chini(row):
    return {"input_ids": tokenize(row["text"])}


def _mds_partition(args):
    sub_dir, dataset, start, end = args
    from streaming import MDSWriter
    with MDSWriter(out=sub_dir, columns={"input_ids": "ndarray:uint32"}) as w:
        for i in range(start, end):
            w.write({"input_ids": tokenize(dataset[i]["text"])})


def human_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def bench(label, fn, path):
    path.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = fn(path)
    elapsed = time.perf_counter() - t0
    size = human_bytes(dir_size(path))
    print(f"  {label:45s} | {elapsed:7.2f}s | {n / elapsed:>10,.0f} samples/s | {size:>10s}")
    return {"time": elapsed, "throughput": n / elapsed}


def main():
    global VOCAB

    from datasets import load_dataset

    BASE.mkdir(parents=True, exist_ok=True)

    print("Loading wikipedia shard 0 …")
    ds = load_dataset("parquet", data_files=PARQUET_URL, split="train")
    N = len(ds)
    print(f"  {N:,} articles")

    print("Building vocab …")
    counter: Counter[str] = Counter()
    for row in tqdm(ds, desc="  words"):
        counter.update(row["text"].split())
    VOCAB = {"<unk>": 0}
    for i, (w, _) in enumerate(counter.most_common(49_999), 1):
        VOCAB[w] = i
    print(f"  {len(VOCAB):,} words")

    print(f"\n{'=' * 95}")
    print(f" write_mp benchmark — {N:,} articles, {NUM_WORKERS} workers, vocab {len(VOCAB):,}")
    print(f"{'=' * 95}")
    print(f"  {'Method':45s} | {'Time':>7s} | {'Throughput':>14s} | {'Size':>10s}")
    print(f"  {'-' * 45}-+-{'-' * 7}-+-{'-' * 14}-+-{'-' * 10}")

    results = {}

    results["mds_seq"] = bench("MosaicML  MDSWriter (sequential)", lambda p: (
        _mds_seq(ds, p), N
    )[-1], BASE / "mds_seq")

    results["mds_par"] = bench(f"MosaicML  MDSWriter (parallel {NUM_WORKERS}w)", lambda p: (
        _mds_par(ds, p, N), N
    )[-1], BASE / "mds_par")

    results["chini_seq"] = bench("ChiniDataset  ParquetWriter (sequential)", lambda p: (
        _chini_seq(ds, p), N
    )[-1], BASE / "chini_seq")

    results["chini_mp"] = bench(f"ChiniDataset  write_mp (parallel {NUM_WORKERS}w)", lambda p: (
        _chini_mp(ds, p), N
    )[-1], BASE / "chini_mp")

    print(f"  {'-' * 45}-+-{'-' * 7}-+-{'-' * 14}-+-{'-' * 10}")

    baseline = results["mds_seq"]["time"]
    print(f"\n  Speedup vs MosaicML sequential:")
    for k, l in [("mds_par", "MosaicML parallel"), ("chini_seq", "ChiniDataset sequential"), ("chini_mp", "ChiniDataset write_mp")]:
        print(f"    {l:45s}  {baseline / results[k]['time']:.2f}x")

    with open(BASE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {BASE / 'results.json'}")


def _mds_seq(ds, path):
    from streaming import MDSWriter
    with MDSWriter(out=str(path), columns={"input_ids": "ndarray:uint32"}) as w:
        for row in tqdm(ds, desc="  MDS seq", leave=False):
            w.write({"input_ids": tokenize(row["text"])})


def _mds_par(ds, path, N):
    from streaming.base.util import merge_index
    chunk = (N + NUM_WORKERS - 1) // NUM_WORKERS
    args = [(str(path / f"{i:05d}"), ds, i * chunk, min((i + 1) * chunk, N)) for i in range(NUM_WORKERS)]
    with Pool(NUM_WORKERS) as pool:
        pool.map(_mds_partition, args)
    merge_index(str(path))


def _chini_seq(ds, path):
    from chinidataset import ParquetWriter
    with ParquetWriter(out=str(path), columns={"input_ids": "uint32[]"}, exist_ok=True) as w:
        for row in tqdm(ds, desc="  Chini seq", leave=False):
            w.write({"input_ids": tokenize(row["text"])})


def _chini_mp(ds, path):
    from chinidataset import ParquetWriter
    with ParquetWriter(out=str(path), columns={"input_ids": "uint32[]"}, exist_ok=True) as w:
        w.write_mp(ds, num_workers=NUM_WORKERS, transform=tokenize_for_chini)


if __name__ == "__main__":
    main()
