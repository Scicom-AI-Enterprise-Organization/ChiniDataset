import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import BASE, NUM_PARTITIONS, human_bytes, dir_size, shard_count, clean_dir, tokenize


def _register_mds_uint32():
    from streaming.base.format.mds.encodings import Encoding, _encodings
    class UInt32(Encoding):
        def encode(self, obj):
            return obj.tobytes()
        def decode(self, data):
            return np.frombuffer(data, np.uint32)
    _encodings["uint32"] = UInt32


def _chinidataset_write_partition(args):
    sub_dir, dataset, start, end, columns = args
    from chinidataset import ParquetWriter
    with ParquetWriter(out=sub_dir, columns=columns, exist_ok=True) as w:
        for i in range(start, end):
            w.write({"input_ids": tokenize(dataset[i]["text"])})


def _mosaicml_write_partition(args):
    sub_dir, dataset, start, end, columns = args
    _register_mds_uint32()
    from streaming import MDSWriter
    with MDSWriter(out=sub_dir, columns=columns) as w:
        for i in range(start, end):
            w.write({"input_ids": tokenize(dataset[i]["text"])})


def bench_chinidataset_write(hf_ds, parallel=True):
    from chinidataset import ParquetWriter
    from chinidataset.util import merge_index

    N = len(hf_ds)
    columns = {"input_ids": "uint32[]"}
    results = {}

    p = BASE / "chinidataset_data"
    clean_dir(p)
    t0 = time.perf_counter()
    with ParquetWriter(out=str(p), columns=columns, exist_ok=True) as w:
        for row in hf_ds:
            w.write({"input_ids": tokenize(row["text"])})
    elapsed = time.perf_counter() - t0
    size = dir_size(p)
    shards = shard_count(p)
    print(f"  ChiniDataset write: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)} | {shards} shards")
    results["chinidataset_write"] = {"time": elapsed, "throughput": N / elapsed, "size": size, "shards": shards}

    if parallel:
        p = BASE / "chinidataset_parallel"
        clean_dir(p)
        chunk = (N + NUM_PARTITIONS - 1) // NUM_PARTITIONS
        args = [(str(p / f"{i:05d}"), hf_ds, i * chunk, min((i + 1) * chunk, N), columns) for i in range(NUM_PARTITIONS)]
        t0 = time.perf_counter()
        with Pool(NUM_PARTITIONS) as pool:
            pool.map(_chinidataset_write_partition, args)
        merge_index(str(p))
        elapsed = time.perf_counter() - t0
        size = dir_size(p)
        print(f"  ChiniDataset parallel ({NUM_PARTITIONS} procs) + merge: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)}")
        results["chinidataset_parallel"] = {"time": elapsed, "throughput": N / elapsed, "size": size}

    return results


def bench_mosaicml_write(hf_ds, parallel=True):
    _register_mds_uint32()
    from streaming import MDSWriter
    from streaming.base.util import merge_index

    N = len(hf_ds)
    columns = {"input_ids": "uint32"}
    results = {}

    p = BASE / "mosaicml_data"
    clean_dir(p)
    t0 = time.perf_counter()
    with MDSWriter(out=str(p), columns=columns) as w:
        for row in hf_ds:
            w.write({"input_ids": tokenize(row["text"])})
    elapsed = time.perf_counter() - t0
    size = dir_size(p)
    shards = shard_count(p)
    print(f"  MosaicML write: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)} | {shards} shards")
    results["mosaicml_write"] = {"time": elapsed, "throughput": N / elapsed, "size": size, "shards": shards}

    if parallel:
        p = BASE / "mosaicml_parallel"
        clean_dir(p)
        chunk = (N + NUM_PARTITIONS - 1) // NUM_PARTITIONS
        args = [(str(p / f"{i:05d}"), hf_ds, i * chunk, min((i + 1) * chunk, N), columns) for i in range(NUM_PARTITIONS)]
        t0 = time.perf_counter()
        with Pool(NUM_PARTITIONS) as pool:
            pool.map(_mosaicml_write_partition, args)
        merge_index(str(p))
        elapsed = time.perf_counter() - t0
        size = dir_size(p)
        print(f"  MosaicML parallel ({NUM_PARTITIONS} procs) + merge: {elapsed:.3f}s | {N/elapsed:,.0f} samples/s | {human_bytes(size)}")
        results["mosaicml_parallel"] = {"time": elapsed, "throughput": N / elapsed, "size": size}

    return results
