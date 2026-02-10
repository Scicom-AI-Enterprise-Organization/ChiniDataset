import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE = Path(__file__).resolve().parent / "output" / "wikipedia"
NUM_PARTITIONS = 4
PARQUET_URL = "hf://datasets/wikimedia/wikipedia/20231101.en/train-00000-of-00041.parquet"

VOCAB: dict[str, int] = {}
UNK_ID = 0


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def shard_count(path: Path) -> int:
    return len(list(path.rglob("shard.*")))


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_wikipedia():
    from datasets import load_dataset
    print("Loading wikipedia shard 0 …")
    ds = load_dataset("parquet", data_files=PARQUET_URL, split="train")
    print(f"  {len(ds):,} articles")
    return ds


def build_vocab(ds, max_vocab=50_000):
    global VOCAB
    print("Building vocab …")
    counter: Counter[str] = Counter()
    for row in tqdm(ds, desc="  words"):
        counter.update(row["text"].split())
    VOCAB = {"<unk>": 0}
    for i, (w, _) in enumerate(counter.most_common(max_vocab - 1), 1):
        VOCAB[w] = i
    print(f"  {len(VOCAB):,} words")
    return VOCAB


def tokenize(text: str) -> np.ndarray:
    return np.array([VOCAB.get(w, UNK_ID) for w in text.split()], dtype=np.uint32)
