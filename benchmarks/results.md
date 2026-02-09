# Benchmark Results

## IMDB (25,000 samples, text + label)

Dataset: [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) test split

All numbers are **samples per second** (higher is better), measured as `num_samples / wall_clock_seconds` using `time.perf_counter()`.

| Metric | MosaicML (samples/s) | ChiniDataset (samples/s) | Speedup |
|---|---|---|:---:|
| Write | 126,888 | 139,425 | 1.1x |
| Parallel write + merge | 51,562 | 108,138 | **2.1x** |
| Read (sequential) | 13,779 | 439,258 | **32x** |
| Read (shuffled) | 12,298 | 459,032 | **37x** |
| DataLoader (w=0) | 13,238 | 363,728 | **28x** |
| DataLoader (w=2) | 1,801 | 2,218 | 1.2x |
| DataLoader (w=4) | 1,067 | 1,164 | 1.1x |
| Read (merged) | 13,662 | 463,930 | **34x** |

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
