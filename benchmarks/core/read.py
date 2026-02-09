"""Read benchmark: ChiniDataset vs MosaicML."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import BASE


def bench_chinidataset_read():
    from chinidataset import StreamingDataset

    results = {}
    pq_path = BASE / "chinidataset_data"
    pq_parallel_path = BASE / "chinidataset_parallel"

    ds = StreamingDataset(local=str(pq_path))
    t0 = time.perf_counter()
    count = sum(1 for _ in ds)
    elapsed = time.perf_counter() - t0
    print(f"  ChiniDataset sequential: {count:,} samples | {count/elapsed:,.0f} samples/s")
    results["chinidataset_read"] = {"time": elapsed, "throughput": count / elapsed}

    ds_shuf = StreamingDataset(local=str(pq_path), shuffle=True)
    t0 = time.perf_counter()
    count = sum(1 for _ in ds_shuf)
    elapsed = time.perf_counter() - t0
    print(f"  ChiniDataset shuffled: {count:,} samples | {count/elapsed:,.0f} samples/s")
    results["chinidataset_shuffle"] = {"time": elapsed, "throughput": count / elapsed}

    ds_merged = StreamingDataset(local=str(pq_parallel_path))
    t0 = time.perf_counter()
    count = sum(1 for _ in ds_merged)
    elapsed = time.perf_counter() - t0
    print(f"  ChiniDataset merged: {count:,} samples | {count/elapsed:,.0f} samples/s")
    results["chinidataset_merge_read"] = {"time": elapsed, "throughput": count / elapsed}

    return results


def bench_mosaicml_read():
    from streaming import StreamingDataset as MosaicDS

    results = {}
    mds_path = BASE / "mosaicml_data"
    mds_parallel_path = BASE / "mosaicml_parallel"

    ds = MosaicDS(local=str(mds_path), shuffle=False, batch_size=32)
    t0 = time.perf_counter()
    count = sum(1 for _ in ds)
    elapsed = time.perf_counter() - t0
    print(f"  MosaicML sequential: {count:,} samples | {count/elapsed:,.0f} samples/s")
    results["mosaicml_read"] = {"time": elapsed, "throughput": count / elapsed}

    ds_shuf = MosaicDS(local=str(mds_path), shuffle=True, batch_size=32, num_canonical_nodes=1)
    t0 = time.perf_counter()
    count = sum(1 for _ in ds_shuf)
    elapsed = time.perf_counter() - t0
    print(f"  MosaicML shuffled: {count:,} samples | {count/elapsed:,.0f} samples/s")
    results["mosaicml_shuffle"] = {"time": elapsed, "throughput": count / elapsed}

    ds_merged = MosaicDS(local=str(mds_parallel_path), shuffle=False, batch_size=32)
    t0 = time.perf_counter()
    count = sum(1 for _ in ds_merged)
    elapsed = time.perf_counter() - t0
    print(f"  MosaicML merged: {count:,} samples | {count/elapsed:,.0f} samples/s")
    results["mosaicml_merge_read"] = {"time": elapsed, "throughput": count / elapsed}

    return results


def main():
    print("\n== Read Benchmark ==")
    results = bench_chinidataset_read()
    results.update(bench_mosaicml_read())

    import json
    with open(BASE / "read_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {BASE / 'read_results.json'}")


if __name__ == "__main__":
    main()
