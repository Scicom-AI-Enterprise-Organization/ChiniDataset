# ChiniML

Parquet-native streaming dataset library for ML training. Drop-in replacement for MosaicML's `LocalDataset`.

Write sharded Parquet datasets with `ParquetWriter`, read them with `StreamingDataset`. Every shard is a standard `.parquet` file -- inspectable by pandas, Spark, DuckDB, anyone.

## Install

```bash
uv pip install git+https://github.com/Scicom-AI-Enterprise-Organization/StreamingDataset.git
uv pip install "chiniml[hf] @ git+https://github.com/Scicom-AI-Enterprise-Organization/StreamingDataset.git"  # + HuggingFace Hub streaming
```

## Write

```python
from datasets import load_dataset
from chiniml import ParquetWriter

hf_ds = load_dataset("ag_news", split="test")
col = {"text": "str", "label": "int32"}

with ParquetWriter(out="./data", columns=col) as w:
    for row in hf_ds:
        w.write(row)
```

## Read

```python
from chiniml import StreamingDataset
from torch.utils.data import DataLoader

ds = StreamingDataset(local="./data")
loader = DataLoader(ds, batch_size=32)

for batch in loader:
    texts = batch["text"]
    labels = batch["label"]
```

Also supports HuggingFace Hub streaming and map-style access:

```python
ds = StreamingDataset(local="/tmp/cache", remote="hf://user/dataset")
sample = ds[42]
```

## Parallel Write + Merge

```python
from chiniml import ParquetWriter
from chiniml.util import merge_index

for part_id, chunk in enumerate(chunks):
    with ParquetWriter(out=f"./output/{part_id:05d}", columns=columns) as w:
        for sample in chunk:
            w.write(sample)

merge_index("./output")
```

## Examples

[Examples](/examples/) through here.

## Benchmarks

[AG News](https://huggingface.co/datasets/ag_news) test set (7,600 samples, text + label):

| Metric | MosaicML (MDS) | ChiniML (Parquet) | Speedup |
|---|---|---|:---:|
| Write | 139,419/s | 132,685/s | 1.0x |
| Parallel write + merge | 50,003/s | 87,673/s | **1.8x** |
| Read (sequential) | 13,371/s | 297,698/s | **22x** |
| Read (shuffled) | 14,010/s | 451,237/s | **32x** |
| DataLoader (w=0) | 13,989/s | 266,998/s | **19x** |
| DataLoader (w=2) | 577/s | 677/s | 1.2x |
| DataLoader (w=4) | 320/s | 354/s | 1.1x |
| Read (merged) | 14,067/s | 154,868/s | **11x** |

> **Note:** Multi-worker DataLoader (w=2, w=4) is bottlenecked by process spawn + IPC overhead on this small dataset. Both libraries hit the same wall. Multi-worker only helps on large datasets where per-sample read cost exceeds the fork/pickle overhead.

Reproduce with:

```bash
python benchmarks/bench_agnews.py
```

## Package

```
chiniml/
├── writer/parquet.py        ParquetWriter + write_batch()
├── dataset/streaming.py     StreamingDataset (IterableDataset + map-style)
├── dataset/reader.py        ParquetReader (numpy-backed, LRU cached)
├── dataset/cache.py         CacheManager (S3/HF download + LRU eviction)
├── dataset/shuffle.py       Deterministic block-based shuffling
├── dataset/partition.py     Partition across workers
├── dataset/world.py         Distributed topology detection
├── hashing.py               File integrity (xxh64, md5, sha256)
└── util.py                  merge_index, bytes_to_int
```

## License

Apache-2.0. Adapted from [mosaicml/streaming](https://github.com/mosaicml/streaming).
