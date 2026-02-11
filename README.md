# ChiniDataset

Resumable streamable Parquet-native streaming dataset library for large scale training. Why Chini? Idk probably Chini Lake.

## Install

```bash
uv pip install git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git
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

#### Write using built-in multiprocessing (`write_mp`)

```python
from chinidataset import ParquetWriter

columns = {"input_ids": "uint32[]"}

def tokenize_row(row):
    return {"input_ids": tokenize(row["text"])}

with ParquetWriter(out="./output", columns=columns) as writer:
    writer.write_mp(hf_ds, num_workers=4, transform=tokenize_row)
```

## Read

```python
from chinidataset import StreamingDataset

ds = StreamingDataset(local="./data", remote="hf://user/dataset")
```

#### Also supports hf streaming from [hf datasets](https://huggingface.co/datasets/nazhan/wikipedia-shard-0-chini):

```python
from datasets import load_dataset
from chinidataset import ParquetWriter, StreamingDataset
from huggingface_hub import HfApi

# load the dataset
ds = load_dataset(
    "parquet",
    data_files="hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet",
    split="train",
)

def transform(row):
    return {"title": row["title"], "text": row["text"]}

# write partitions using mp for speed, it supports index merging too
columns = {"title": "str", "text": "str"}

with ParquetWriter(out="./wikipedia-shard-0", columns=columns) as w:
    w.write_mp(ds, num_workers=4, transform=transform)

# upload partitions to hf 
api = HfApi(token="TOKEN")
api.upload_folder(
    folder_path="./wikipedia-shard-0",
    repo_id="nazhan/wikipedia-shard-0-chini",
    repo_type="dataset",
)

# stream the data remotely from hf
ds = StreamingDataset(
    local="/tmp/wiki_cache",
    remote="hf://nazhan/wikipedia-shard-0-chini",
)
```

#### Using look_ahead param to optimize reading speed

```python
from chinidataset import StreamingDataset

ds = StreamingDataset(
    local="./data",
    max_open_shards=8,   # keep at most 8 shard readers in memory (LRU eviction)
    look_ahead=2,        # pre-load next 2 shards in background threads
)
```


## Examples

- [1_example_chinidataset.ipynb](/examples/1_example_chinidataset.ipynb)
- [2_example_uint32_numpy_array_write.ipynb](/examples/2_example_uint32_numpy_array_write.ipynb)
- [3_example_write_mp.ipynb](/examples/3_example_write_mp.ipynb)
- [4_simulation_lrucache_look_ahead.ipynb](/examples/4_simulation_lrucache_look_ahead.ipynb)

## Benchmarks

#### 1. General benchmark (write, read, read shuffled)

[Wikipedia EN](https://huggingface.co/datasets/wikimedia/wikipedia) shard 0 (156,289 articles, word tokenizer, `input_ids` uint32[]):

| Metric | MosaicML (MDS) | ChiniDataset (PQ) | Speedup |
|---|---|---|:---:|
| Write | 9,790/s | 11,192/s | **1.1x** |
| Read | 12,339/s | 122,317/s | **9.9x** |
| Read (shuffled) | 16,490/s | 222,433/s | **13.5x** |

Run: [benchmarks/run.py](/benchmarks/run.py)

```bash
uv run python benchmarks/run.py
```

#### 2. Uint32 numpy array tokens write & read benchmark

[Wikipedia EN](https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/20231101.en/train-00000-of-00041.parquet) shard (156,289 articles, word tokenizer O(1) dict lookup, `uint32[]` arrays):

| Metric | MosaicML | ChiniDataset | Speedup |
|---|---|---|:---:|
| Tokenize + Write | 13.96s (11,198 rows/s) | 16.27s (9,605 rows/s) | 0.9x |
| Read | 166.95s (936 rows/s) | 1.30s (120,680 rows/s) | **128.9x** |

Run: [benchmarks/bench_uint32.py](/benchmarks/bench_uint32.py)

```bash
uv run python benchmarks/bench_uint32.py
```

#### 3. `write_mp` parallel write benchmark

[Wikipedia EN](https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/20231101.en/train-00000-of-00041.parquet) shard (156,289 articles, word-level tokenizer, `uint32[]` arrays):

| Method | Time (s) | Samples/s | Speedup |
|---|---|---|:---:|
| MosaicML MDSWriter (sequential) | 16.03s | 9,752 | 1.0x |
| MosaicML MDSWriter (parallel 4w) | 7.79s | 20,068 | 2.06x |
| ChiniDataset ParquetWriter (sequential) | 12.82s | 12,188 | 1.25x |
| ChiniDataset write_mp (parallel 4w) | **6.24s** | **25,050** | **2.57x** |

Run: [benchmarks/bench_write_mp.py](/benchmarks/bench_write_mp.py)

```bash
uv run python benchmarks/bench_write_mp.py
```

#### 4. Write backend comparison (PyArrow vs Pandas vs Polars)

[Wikipedia EN](https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/20231101.en/train-00000-of-00041.parquet) shard (156,289 articles, word tokenizer, `uint32[]` arrays).
| Writer | Backend | Time | Samples/s | vs ChiniDataset |
|---|---|---|---|:---:|
| **ChiniDataset `ParquetWriter`** | PyArrow (direct) | **13.7s** | **11,446/s** | **1.00x** |
| Pandas `to_parquet` | PyArrow | 14.8s | 10,578/s | 0.92x |
| Pandas `to_parquet` | fastparquet | 24.3s | 6,434/s | 0.56x |
| Polars `write_parquet` | Rust | 23.2s | 6,739/s | 0.59x |

Run: [benchmarks/bench_write_backends.py](/benchmarks/bench_write_backends.py)

```bash
uv run python benchmarks/bench_write_backends.py
```

## Package

```
chinidataset/
├── writer/parquet.py        ParquetWriter + write_mp() + write_batch()
├── dataset/streaming.py     StreamingDataset (IterableDataset + map-style)
├── dataset/reader.py        ParquetReader (pandas backend, LRU cached)
├── dataset/cache.py         CacheManager (S3/HF download + LRU eviction)
├── dataset/shuffle.py       Deterministic block-based shuffling
├── dataset/partition.py     Partition across workers
├── dataset/world.py         Distributed topology detection
├── hashing.py               File integrity (xxh64, md5, sha256)
└── util.py                  merge_index, bytes_to_int
```

## Links

- [mosaicml/streaming](https://github.com/mosaicml/streaming)

