"""uv run python benchmarks/bench_uint32.py"""

import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE = Path(__file__).resolve().parent / "output" / "uint32"
PARQUET_URL = "hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet"

VOCAB: dict[str, int] = {}
UNK_ID = 0


def tokenize(text: str) -> np.ndarray:
    return np.array([VOCAB.get(w, UNK_ID) for w in text.split()], dtype=np.uint32)


def human_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


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

    print(f"\n{'=' * 85}")
    print(f" uint32 benchmark — {N:,} articles, vocab {len(VOCAB):,}")
    print(f"{'=' * 85}")
    print(f"  {'Method':40s} | {'Time':>7s} | {'Throughput':>14s} | {'Size':>10s}")
    print(f"  {'-' * 40}-+-{'-' * 7}-+-{'-' * 14}-+-{'-' * 10}")

    # -- ChiniDataset tokenize + write --
    from chinidataset import ParquetWriter

    p = BASE / "chini"
    p.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ParquetWriter(out=str(p), columns={"input_ids": "uint32[]", "labels": "uint32[]"}, exist_ok=True) as w:
        for row in tqdm(ds, desc="  Chini write", leave=False):
            ids = tokenize(row["text"])
            w.write({"input_ids": ids, "labels": ids})
    chini_write = time.perf_counter() - t0
    chini_size = human_bytes(dir_size(p))
    print(f"  {'ChiniDataset tokenize + write':40s} | {chini_write:7.2f}s | {N / chini_write:>10,.0f} rows/s | {chini_size:>10s}")

    # -- MosaicML tokenize + write --
    from streaming import MDSWriter
    from streaming.base.format.mds.encodings import Encoding, _encodings

    class UInt32(Encoding):
        def encode(self, obj):
            return obj.tobytes()
        def decode(self, data):
            return np.frombuffer(data, np.uint32)

    _encodings["uint32"] = UInt32

    p = BASE / "mds"
    p.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with MDSWriter(out=str(p), columns={"input_ids": "uint32", "labels": "uint32"}) as w:
        for row in tqdm(ds, desc="  MDS write", leave=False):
            ids = tokenize(row["text"])
            w.write({"input_ids": ids, "labels": ids})
    mds_write = time.perf_counter() - t0
    mds_size = human_bytes(dir_size(p))
    print(f"  {'MosaicML tokenize + write':40s} | {mds_write:7.2f}s | {N / mds_write:>10,.0f} rows/s | {mds_size:>10s}")

    print(f"  {'-' * 40}-+-{'-' * 7}-+-{'-' * 14}-+-{'-' * 10}")

    # -- ChiniDataset read --
    from chinidataset import StreamingDataset

    d = StreamingDataset(local=str(BASE / "chini"))
    t0 = time.perf_counter()
    count = 0
    for sample in tqdm(d, desc="  Chini read", leave=False, total=N):
        _ = sample["input_ids"]
        count += 1
    chini_read = time.perf_counter() - t0
    print(f"  {'ChiniDataset read':40s} | {chini_read:7.2f}s | {count / chini_read:>10,.0f} rows/s |")

    # -- MosaicML read --
    from streaming import StreamingDataset as MosaicDS

    d = MosaicDS(local=str(BASE / "mds"), shuffle=False, batch_size=1)
    t0 = time.perf_counter()
    count = 0
    for sample in tqdm(d, desc="  MDS read", leave=False, total=N):
        _ = sample["input_ids"]
        count += 1
    mds_read = time.perf_counter() - t0
    print(f"  {'MosaicML read':40s} | {mds_read:7.2f}s | {count / mds_read:>10,.0f} rows/s |")

    print(f"  {'-' * 40}-+-{'-' * 7}-+-{'-' * 14}-+-{'-' * 10}")

    print(f"\n  Summary:")
    print(f"    {'Metric':30s} {'MosaicML':>12s} {'ChiniDataset':>14s} {'Speedup':>10s}")
    print(f"    {'-' * 68}")
    print(f"    {'Tokenize + Write (rows/s)':30s} {N / mds_write:>10,.0f}/s {N / chini_write:>12,.0f}/s {mds_write / chini_write:>9.2f}x")
    print(f"    {'Read (rows/s)':30s} {count / mds_read:>10,.0f}/s {count / chini_read:>12,.0f}/s {mds_read / chini_read:>9.2f}x")


if __name__ == "__main__":
    main()
