# ChiniDataset

Resumable streamable Parquet-native streaming dataset library for large scale training.

Write sharded Parquet datasets with `ParquetWriter`, read them with `StreamingDataset`. Every shard is a standard `.parquet` file -- inspectable by pandas, Spark, DuckDB, anyone.

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

## Parallel Write (`write_mp`)

`write_mp` parallelises the **entire pipeline**: it partitions the dataset across N workers, each worker iterates + transforms + writes its own chunk, then merges the index files automatically.

```python
from chinidataset import ParquetWriter

columns = {"input_ids": "uint32[]"}

def tokenize_row(row):
    return {"input_ids": tokenize(row["text"])}

with ParquetWriter(out="./output", columns=columns) as writer:
    writer.write_mp(hf_ds, num_workers=4, transform=tokenize_row)
```

`transform` is optional — skip it if your data is already in the right shape:

```python
with ParquetWriter(out="./output", columns=columns) as writer:
    writer.write_mp(hf_ds, num_workers=4)
```

Under the hood: partitions dataset → spawns N processes → each writes to `output/{part_id}/` → `merge_index` combines into one `index.json`.

## LRU Cache + Look-Ahead

```python
from chinidataset import StreamingDataset

ds = StreamingDataset(
    local="./data",
    max_open_shards=8,   # keep at most 8 shard readers in memory (LRU eviction)
    look_ahead=2,        # pre-load next 2 shards in background threads
)
```

## Examples

- [1_example_chinidataset.ipynb](/examples/1_example_chinidataset.ipynb) — Write + Read + DataLoader
- [2_example_uint32_numpy_array_write.ipynb](/examples/2_example_uint32_numpy_array_write.ipynb) — Tokenized uint32 arrays
- [3_example_write_mp.ipynb](/examples/3_example_write_mp.ipynb) — Parallel write with `write_mp`
- [4_simulation_lrucache_look_ahead.ipynb](/examples/4_simulation_lrucache_look_ahead.ipynb) — LRU cache + look-ahead benchmark

## Benchmarks

### 1. General benchmark (write, read, read shuffled)

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

### 2. Uint32 numpy array tokens write & read benchmark

[Wikipedia EN](https://huggingface.co/datasets/wikimedia/wikipedia/blob/main/20231101.en/train-00000-of-00041.parquet) shard (156,289 articles, word tokenizer O(1) dict lookup, `uint32[]` arrays):

| Metric | MosaicML | ChiniDataset | Speedup |
|---|---|---|:---:|
| Tokenize + Write | 13.96s (11,198 rows/s) | 16.27s (9,605 rows/s) | 0.9x |
| Read | 166.95s (936 rows/s) | 1.30s (120,680 rows/s) | **128.9x** |

Run: [benchmarks/bench_uint32.py](/benchmarks/bench_uint32.py)

```bash
uv run python benchmarks/bench_uint32.py
```

### 3. `write_mp` parallel write benchmark

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

