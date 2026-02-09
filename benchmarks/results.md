# Benchmark Results

## IMDB (25,000 samples, text + label)

Dataset: [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) test split

All numbers are **samples per second** (higher is better), measured as `num_samples / wall_clock_seconds` using `time.perf_counter()`.

| Metric | MosaicML (samples/s) | ChiniDataset (samples/s) | Speedup |
|---|---|---|:---:|
| Write | 126,306 | 128,645 | 1.0x |
| Parallel write + merge | 11,598 | 20,010 | **1.7x** |
| Read (sequential) | 13,011 | 280,706 | **22x** |
| Read (shuffled) | 13,606 | 434,398 | **32x** |
| DataLoader (w=0) | 11,239 | 197,322 | **18x** |
| DataLoader (w=2) | 1,803 | 2,148 | 1.2x |
| DataLoader (w=4) | 1,056 | 1,153 | 1.1x |
| Read (merged) | 12,044 | 471,112 | **39x** |

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
