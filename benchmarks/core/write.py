"""Write benchmark: ChiniDataset vs MosaicML."""

import sys
import time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import BASE, NUM_PARTITIONS, human_bytes, dir_size, shard_count, clean_dir, load_imdb


def _chinidataset_write_partition(args):
    sub_dir, samples, columns = args
    from chinidataset import ParquetWriter
    with ParquetWriter(out=sub_dir, columns=columns) as w:
        for sample in samples:
            w.write(sample)


def _mosaicml_write_partition(args):
    sub_dir, samples, columns = args
    from streaming import MDSWriter
    with MDSWriter(out=sub_dir, columns=columns) as w:
        for sample in samples:
            w.write({"text": sample["text"], "label": sample["label"]})


def bench_chinidataset_write(hf_ds):
    from chinidataset import ParquetWriter
    from chinidataset.util import merge_index

    N = len(hf_ds)
    columns = {"text": "str", "label": "int32"}
    results = {}

    pq_path = BASE / "chinidataset_data"
    clean_dir(pq_path)

    t0 = time.perf_counter()
    with ParquetWriter(out=str(pq_path), columns=columns) as w:
        for row in hf_ds:
            w.write(row)
    elapsed = time.perf_counter() - t0
    size = dir_size(pq_path)
    shards = shard_count(pq_path)
    print(f"  ChiniDataset write: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)} | {shards} shards")
    results["chinidataset_write"] = {"time": elapsed, "throughput": N / elapsed, "size": size, "shards": shards}

    pq_parallel_path = BASE / "chinidataset_parallel"
    clean_dir(pq_parallel_path)
    chunk_size = (N + NUM_PARTITIONS - 1) // NUM_PARTITIONS

    partition_args = []
    for part_id in range(NUM_PARTITIONS):
        start = part_id * chunk_size
        end = min(start + chunk_size, N)
        sub_dir = str(pq_parallel_path / f"{part_id:05d}")
        samples = [hf_ds[i] for i in range(start, end)]
        partition_args.append((sub_dir, samples, columns))

    t0 = time.perf_counter()
    with Pool(processes=NUM_PARTITIONS) as pool:
        pool.map(_chinidataset_write_partition, partition_args)
    merge_index(str(pq_parallel_path))
    elapsed = time.perf_counter() - t0
    size = dir_size(pq_parallel_path)
    print(f"  ChiniDataset parallel (4 procs) + merge: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)}")
    results["chinidataset_parallel"] = {"time": elapsed, "throughput": N / elapsed, "size": size}

    return results


def bench_mosaicml_write(hf_ds):
    from streaming import MDSWriter
    from streaming.base.util import merge_index

    N = len(hf_ds)
    columns = {"text": "str", "label": "int"}
    results = {}

    mds_path = BASE / "mosaicml_data"
    clean_dir(mds_path)

    t0 = time.perf_counter()
    with MDSWriter(out=str(mds_path), columns=columns) as w:
        for row in hf_ds:
            w.write({"text": row["text"], "label": row["label"]})
    elapsed = time.perf_counter() - t0
    size = dir_size(mds_path)
    shards = shard_count(mds_path)
    print(f"  MosaicML write: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)} | {shards} shards")
    results["mosaicml_write"] = {"time": elapsed, "throughput": N / elapsed, "size": size, "shards": shards}

    mds_parallel_path = BASE / "mosaicml_parallel"
    clean_dir(mds_parallel_path)
    chunk_size = (N + NUM_PARTITIONS - 1) // NUM_PARTITIONS

    partition_args = []
    for part_id in range(NUM_PARTITIONS):
        start = part_id * chunk_size
        end = min(start + chunk_size, N)
        sub_dir = str(mds_parallel_path / f"{part_id:05d}")
        samples = [hf_ds[i] for i in range(start, end)]
        partition_args.append((sub_dir, samples, columns))

    t0 = time.perf_counter()
    with Pool(processes=NUM_PARTITIONS) as pool:
        pool.map(_mosaicml_write_partition, partition_args)
    merge_index(str(mds_parallel_path))
    elapsed = time.perf_counter() - t0
    size = dir_size(mds_parallel_path)
    print(f"  MosaicML parallel (4 procs) + merge: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)}")
    results["mosaicml_parallel"] = {"time": elapsed, "throughput": N / elapsed, "size": size}

    return results


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    hf_ds = load_imdb()

    print("\n== Write Benchmark ==")
    results = bench_chinidataset_write(hf_ds)
    results.update(bench_mosaicml_write(hf_ds))

    import json
    with open(BASE / "write_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {BASE / 'write_results.json'}")


if __name__ == "__main__":
    main()
