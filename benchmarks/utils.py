import shutil
import time
from pathlib import Path


BASE = Path("./benchmark_output/imdb")
NUM_PARTITIONS = 4


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def shard_count(path: Path) -> int:
    return len(list(path.glob("shard.*")))


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)


def load_imdb():
    from datasets import load_dataset
    hf_ds = load_dataset("stanfordnlp/imdb", split="test")
    print(f"Loaded {len(hf_ds):,} samples")
    return hf_ds
