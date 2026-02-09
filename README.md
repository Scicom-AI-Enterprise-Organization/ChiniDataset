# ChiniDataset

Parquet-native streaming dataset library for ML training. Drop-in replacement for MosaicML's `LocalDataset`.

Write sharded Parquet datasets with `ParquetWriter`, read them with `StreamingDataset`. Every shard is a standard `.parquet` file -- inspectable by pandas, Spark, DuckDB, anyone.

## Install

```bash
uv pip install git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git
uv pip install "chinidataset[hf] @ git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git"  # + HuggingFace Hub streaming
```

## Write

```python
from datasets import load_dataset
from chinidataset import ParquetWriter

hf_ds = load_dataset("stanfordnlp/imdb", split="test")

col = {"text": "str", "label": "int32"}

with ParquetWriter(out="./data", columns=col) as w:
    for row in hf_ds:
        w.write(row)
```

## Read

```python
from chinidataset import StreamingDataset
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
from chinidataset import ParquetWriter
from chinidataset.util import merge_index

for part_id, chunk in enumerate(chunks):
    with ParquetWriter(out=f"./output/{part_id:05d}", columns=columns) as w:
        for sample in chunk:
            w.write(sample)

merge_index("./output")
```

## Examples

[Examples](/examples/) through here.

## Benchmarks

[IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) test set (25,000 samples, text + label):

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

Reproduce with the scripts in [benchmarks/](/benchmarks/):

```bash
python benchmarks/run.py
```

See [benchmarks/results.md](/benchmarks/results.md) for full details.

## Package

```
chinidataset/
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

## Links

- [HuggingFace Hub](https://huggingface.co/)
- [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [mosaicml/streaming](https://github.com/mosaicml/streaming)

