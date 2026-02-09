# Benchmark Results

## IMDB (25,000 samples, text + label)

Dataset: [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) test split

All numbers are **samples per second** (higher is better), measured as `num_samples / wall_clock_seconds` using `time.perf_counter()`.

| Metric | MosaicML (MDS) | ChiniDataset (Parquet) | Speedup |
|---|---|---|:---:|
| Write | 126,888/s | 139,425/s | 1.1x |
| Parallel write + merge | 51,562/s | 108,138/s | **2.1x** |
| Read (sequential) | 13,779/s | 439,258/s | **32x** |
| Read (shuffled) | 12,298/s | 459,032/s | **37x** |
| DataLoader (w=0) | 13,238/s | 363,728/s | **28x** |
| DataLoader (w=2) | 1,801/s | 2,218/s | 1.2x |
| DataLoader (w=4) | 1,067/s | 1,164/s | 1.1x |
| Read (merged) | 13,662/s | 463,930/s | **34x** |

### What each metric measures

| Metric | Description |
|---|---|
| **Write** | Write all rows one-by-one via `writer.write(row)` into a single shard |
| **Parallel write + merge** | Write in 4 partitions concurrently, then merge index files into one |
| **Read (sequential)** | Iterate all rows in order via `for sample in dataset` |
| **Read (shuffled)** | Same iteration but with `shuffle=True` (random shard access pattern) |
| **DataLoader (w=0)** | Full PyTorch DataLoader loop, single process (no workers) |
| **DataLoader (w=2/4)** | PyTorch DataLoader with 2 or 4 worker subprocesses |
| **Read (merged)** | Sequential read from a dataset created by parallel write + merge |

### Why ChiniDataset is faster on reads

MosaicML's MDS format stores samples as individual binary blobs and reads them one at a time. ChiniDataset stores data as Parquet columnar batches read via PyArrow, which loads entire row groups in a single I/O call and decodes columns in bulk using native C++/SIMD. This is why reads are 30-37x faster.

### Why multi-worker DataLoader is slow for both

> Multi-worker DataLoader (w=2, w=4) is bottlenecked by process spawn + IPC overhead on this small dataset. PyTorch forks subprocesses and pickles every batch across process boundaries. On a 25k-sample dataset, the fork/pickle cost dominates the actual read time. Both libraries hit the same wall. Multi-worker only helps on large datasets where per-sample I/O cost exceeds the process overhead.

### Reproduce

```bash
python benchmarks/run.py
```

### Environment

- Apple Silicon (macOS)
- Python 3.11
- PyTorch 2.x
- chinidataset 0.1.0
- mosaicml-streaming 0.13.0
