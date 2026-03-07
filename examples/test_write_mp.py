"""Verify write_mp: tqdm per-worker bars + fork-based dataset sharing (no pickle overhead).

Usage:
    python test_write_mp.py           # run fixed version
    python test_write_mp.py --broken  # simulate original pickling problem
"""

import argparse
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chinidataset import ParquetWriter
import chinidataset.writer.parquet as _parquet_module


# ---------------------------------------------------------------------------
# Fake large dataset — simulates a real NLP corpus
# Each row has ~2 KB of text to make memory pressure visible
# ---------------------------------------------------------------------------
LONG_TEXT = "word " * 400  # ~2 KB per sample

class LargeDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "text": f"[{idx}] " + LONG_TEXT,
            "speaker": f"spk{idx % 100}",
            "processed_text": f"proc_{idx} " + LONG_TEXT[:100],
            "token_filename": f"/data/tokens/{idx:08d}.pt",
        }


# ---------------------------------------------------------------------------
# Broken worker (old behaviour): dataset pickled inside every args tuple
# ---------------------------------------------------------------------------
def _broken_worker(args):
    sub_dir, dataset, start, end, columns, writer_kwargs = args
    with ParquetWriter(out=sub_dir, columns=columns, **writer_kwargs) as w:
        for i in range(start, end):
            w.write(dataset[i])


def run_broken(dataset, out_dir, num_workers):
    """Simulate the original bug: full dataset pickled per worker."""
    columns = {"text": "str", "speaker": "str", "processed_text": "str", "token_filename": "str"}
    N = len(dataset)
    chunk_size = (N + num_workers - 1) // num_workers
    writer_kwargs = {"exist_ok": True}

    partition_args = []
    for part_id in range(num_workers):
        start = part_id * chunk_size
        end = min(start + chunk_size, N)
        if start >= N:
            break
        sub_dir = str(Path(out_dir) / f"{part_id:05d}")
        partition_args.append((sub_dir, dataset, start, end, columns, writer_kwargs))

    print(f"[BROKEN] Pickling {len(partition_args)} args tuples each containing "
          f"the full dataset ({N} rows × ~2KB ≈ {N * 2 // 1024} MB)...")
    t0 = time.time()
    with Pool(processes=len(partition_args)) as pool:
        pool.map(_broken_worker, partition_args)
    print(f"[BROKEN] Done in {time.time() - t0:.1f}s")


def run_fixed(dataset, out_dir, num_workers):
    """Fixed version: dataset shared via fork, tqdm per worker."""
    columns = {"text": "str", "speaker": "str", "processed_text": "str", "token_filename": "str"}
    print(f"[FIXED] Dataset set as global (fork copy-on-write), no pickling...")
    t0 = time.time()
    with ParquetWriter(out=out_dir, columns=columns, exist_ok=True) as writer:
        writer.write_mp(dataset, num_workers=num_workers, use_tqdm=True)
    print(f"[FIXED] Done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broken", action="store_true",
                        help="Simulate the original pickling problem")
    parser.add_argument("-n", type=int, default=200_000,
                        help="Number of rows (default 200k ≈ ~400MB)")
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    dataset = LargeDataset(args.n)
    approx_mb = args.n * 2 // 1024
    print(f"Dataset: {args.n:,} rows ≈ {approx_mb} MB  |  workers: {args.workers}\n")

    out_dir = tempfile.mkdtemp(prefix="test_write_mp_")
    try:
        if args.broken:
            run_broken(dataset, out_dir, args.workers)
        else:
            run_fixed(dataset, out_dir, args.workers)
    finally:
        shutil.rmtree(out_dir)


if __name__ == "__main__":
    main()
