"""Benchmark: ChiniDataset ParquetReader vs MosaicML StreamingDataset read.

Writes Wikipedia EN shard 0 to both formats, then benchmarks sequential
read throughput.

Uses subprocess for MosaicML to avoid huggingface-hub version conflicts.

Usage:
    uv run python benchmarks/bench_read_comparison.py
"""

import json
import shutil
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE = Path(__file__).resolve().parent / "output" / "read_comparison"
PARQUET_URL = "hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet"

VOCAB: dict[str, int] = {}
UNK_ID = 0
REPEATS = 3


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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_wikipedia():
    """Load Wikipedia, build vocab, return HF dataset."""
    global VOCAB
    from datasets import load_dataset

    print("Loading Wikipedia EN shard 0 ...")
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
    print(f"  {len(VOCAB):,} words")
    return ds


def write_chinidataset(ds):
    """Write tokenized Wikipedia to ChiniDataset (Parquet) format."""
    from chinidataset import ParquetWriter

    chini_dir = BASE / "chinidataset"
    if chini_dir.exists():
        shutil.rmtree(chini_dir)
    chini_dir.mkdir(parents=True, exist_ok=True)

    print("Writing ChiniDataset shards ...")
    with ParquetWriter(
        out=str(chini_dir),
        columns={"input_ids": "uint32[]", "labels": "uint32[]"},
        exist_ok=True,
    ) as w:
        for row in tqdm(ds, desc="  chini write", leave=False):
            ids = tokenize(row["text"])
            w.write({"input_ids": ids, "labels": ids})
    sz = human_bytes(dir_size(chini_dir))
    n_shards = len(list(chini_dir.glob("shard.*")))
    print(f"  ChiniDataset: {sz}, {n_shards} shards")


def write_mosaicml(ds):
    """Write tokenized Wikipedia to MosaicML MDS format via subprocess."""
    mds_dir = BASE / "mosaicml"
    if mds_dir.exists():
        shutil.rmtree(mds_dir)
    mds_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenized data to a temp file for the subprocess
    print("Tokenizing for MDS ...")
    temp_file = BASE / "_temp_tokens.npz"
    all_ids = []
    for row in tqdm(ds, desc="  tokenize", leave=False):
        all_ids.append(tokenize(row["text"]))
    np.savez(str(temp_file), *all_ids)
    print(f"  {len(all_ids):,} samples tokenized")

    # Write MDS in subprocess (avoids huggingface-hub conflict)
    print("Writing MosaicML MDS shards (subprocess) ...")
    script = textwrap.dedent(f"""\
        import sys
        import numpy as np
        # Force huggingface-hub compat
        import importlib
        from streaming.base.format.mds.encodings import Encoding, _encodings
        from streaming import MDSWriter

        class UInt32(Encoding):
            def encode(self, obj):
                return obj.tobytes()
            def decode(self, data):
                return np.frombuffer(data, np.uint32)
        _encodings["uint32"] = UInt32

        data = np.load("{temp_file}", allow_pickle=True)
        n = len(data.files)

        with MDSWriter(out="{mds_dir}", columns={{"input_ids": "uint32", "labels": "uint32"}}) as w:
            for i in range(n):
                ids = data[f"arr_{{i}}"]
                w.write({{"input_ids": ids, "labels": ids}})
        print(f"  MDS wrote {{n:,}} samples")
    """)

    # Run with pinned huggingface-hub
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
    )
    if result.returncode != 0:
        print(f"  MDS write FAILED:\n{result.stderr}")
        return False
    print(result.stdout.strip())
    temp_file.unlink(missing_ok=True)

    sz = human_bytes(dir_size(mds_dir))
    print(f"  MosaicML MDS: {sz}")
    return True


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_chinidataset_read():
    """Benchmark ChiniDataset ParquetReader (pandas backend)."""
    from chinidataset import StreamingDataset

    chini_dir = BASE / "chinidataset"
    times = []
    count = 0

    for run in range(REPEATS):
        ds = StreamingDataset(local=str(chini_dir))
        t0 = time.perf_counter()
        count = 0
        for sample in tqdm(ds, desc=f"    run {run + 1}", leave=False):
            _ = sample["input_ids"]
            count += 1
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    run {run + 1}: {count:,} samples in {elapsed:.3f}s  ({count / elapsed:,.0f} samples/s)")

    return {"samples": count, "times": times, "avg": np.mean(times), "best": min(times)}


def bench_mosaicml_read():
    """Benchmark MosaicML StreamingDataset (MDS) via subprocess."""
    mds_dir = BASE / "mosaicml"
    results_file = BASE / "_mds_bench.json"

    script = textwrap.dedent(f"""\
        import json
        import time
        import numpy as np
        from tqdm import tqdm
        from streaming import StreamingDataset as MosaicDS

        REPEATS = {REPEATS}
        mds_dir = "{mds_dir}"
        times = []
        count = 0

        for run in range(REPEATS):
            ds = MosaicDS(local=mds_dir, shuffle=False, batch_size=1)
            t0 = time.perf_counter()
            count = 0
            for sample in tqdm(ds, desc=f"    run {{run + 1}}", leave=False):
                _ = sample["input_ids"]
                count += 1
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"    run {{run + 1}}: {{count:,}} samples in {{elapsed:.3f}}s  ({{count / elapsed:,.0f}} samples/s)")

        result = {{
            "samples": count,
            "times": times,
            "avg": float(np.mean(times)),
            "best": float(min(times)),
        }}
        with open("{results_file}", "w") as f:
            json.dump(result, f)
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "HF_HUB_DISABLE_TELEMETRY": "1"},
    )
    if result.returncode != 0:
        print(f"  MosaicML read FAILED:\n{result.stderr}")
        return None
    print(result.stdout.strip())

    with open(results_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    BASE.mkdir(parents=True, exist_ok=True)

    chini_dir = BASE / "chinidataset"
    mds_dir = BASE / "mosaicml"

    # Check if data exists
    chini_ready = chini_dir.exists() and list(chini_dir.glob("shard.*"))
    mds_ready = mds_dir.exists() and list(mds_dir.glob("shard.*"))

    if not chini_ready or not mds_ready:
        ds = prepare_wikipedia()
        if not chini_ready:
            write_chinidataset(ds)
        if not mds_ready:
            ok = write_mosaicml(ds)
            if not ok:
                print("\nERROR: Could not write MDS data. Skipping MosaicML benchmark.")
                print("You may need: uv pip install 'huggingface-hub<1.0' 'mosaicml-streaming'")
                mds_ready = False
            else:
                mds_ready = True
        del ds
    else:
        print(f"Using existing data in {BASE}")

    print(f"\n{'=' * 70}")
    print(f" Read Benchmark: ChiniDataset (Parquet) vs MosaicML (MDS)")
    print(f" Data: Wikipedia EN shard 0, uint32[] columns")
    print(f" ChiniDataset size: {human_bytes(dir_size(chini_dir))}")
    if mds_ready:
        print(f" MosaicML size:     {human_bytes(dir_size(mds_dir))}")
    print(f" Repeats: {REPEATS}")
    print(f"{'=' * 70}\n")

    print("  ChiniDataset (ParquetReader, pandas backend):")
    r_chini = bench_chinidataset_read()

    r_mosaic = None
    if mds_ready:
        print()
        print("  MosaicML (StreamingDataset, MDS):")
        r_mosaic = bench_mosaicml_read()

    # -- Summary --
    n = r_chini["samples"]
    print(f"\n{'=' * 70}")
    print(f" Summary  ({n:,} samples)")
    print(f"{'=' * 70}")

    if r_mosaic:
        speedup_avg = r_mosaic["avg"] / r_chini["avg"] if r_chini["avg"] else 0
        speedup_best = r_mosaic["best"] / r_chini["best"] if r_chini["best"] else 0
        print(f"  {'':25s} {'MosaicML (MDS)':>16s} {'ChiniDataset (PQ)':>18s} {'Speedup':>10s}")
        print(f"  {'-' * 71}")
        print(f"  {'Avg time':25s} {r_mosaic['avg']:>15.3f}s {r_chini['avg']:>17.3f}s {speedup_avg:>9.2f}x")
        print(f"  {'Best time':25s} {r_mosaic['best']:>15.3f}s {r_chini['best']:>17.3f}s {speedup_best:>9.2f}x")
        print(f"  {'Avg throughput':25s} {n / r_mosaic['avg']:>12,.0f}/s {n / r_chini['avg']:>14,.0f}/s")
        print(f"  {'Best throughput':25s} {n / r_mosaic['best']:>12,.0f}/s {n / r_chini['best']:>14,.0f}/s")
    else:
        print(f"  ChiniDataset avg:  {r_chini['avg']:.3f}s  ({n / r_chini['avg']:,.0f} samples/s)")
        print(f"  ChiniDataset best: {r_chini['best']:.3f}s  ({n / r_chini['best']:,.0f} samples/s)")
        print(f"\n  (MosaicML benchmark skipped due to dependency conflict)")
    print()


if __name__ == "__main__":
    main()
