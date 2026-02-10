"""General benchmark: write, read, read (shuffled) — ChiniDataset vs MosaicML. Wikipedia shard 0, word-level tokenizer, uint32[]."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import BASE, human_bytes, load_wikipedia, build_vocab
from core.write import bench_chinidataset_write, bench_mosaicml_write
from core.read import bench_chinidataset_read, bench_mosaicml_read


def print_summary(results, N):
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"\nDataset: Wikipedia EN shard 0 ({N:,} articles, word tokenizer, input_ids uint32[])\n")
    print(f"{'Metric':<25} {'MosaicML (MDS)':>18} {'ChiniDataset (PQ)':>18} {'Speedup':>10}")
    print("-" * 75)

    def row(label, mds_key, pq_key):
        mds = results.get(mds_key)
        pq = results.get(pq_key)
        if not mds or not pq:
            return
        speedup = pq["throughput"] / mds["throughput"] if mds["throughput"] else 0
        print(f"{label:<25} {mds['throughput']:>14,.0f}/s {pq['throughput']:>14,.0f}/s {speedup:>9.1f}x")

    row("Write", "mosaicml_write", "chinidataset_write")
    if results.get("mosaicml_write") and results.get("chinidataset_write"):
        print(f"{'File size':<25} {human_bytes(results['mosaicml_write']['size']):>18} {human_bytes(results['chinidataset_write']['size']):>18}")
        print(f"{'Shards':<25} {results['mosaicml_write']['shards']:>18} {results['chinidataset_write']['shards']:>18}")
    row("Read", "mosaicml_read", "chinidataset_read")
    row("Read (shuffled)", "mosaicml_shuffle", "chinidataset_shuffle")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    hf_ds = load_wikipedia()
    build_vocab(hf_ds)
    N = len(hf_ds)
    results = {}

    print("\n== Write ==")
    results.update(bench_chinidataset_write(hf_ds, parallel=False))
    results.update(bench_mosaicml_write(hf_ds, parallel=False))

    print("\n== Read ==")
    results.update(bench_chinidataset_read())
    results.update(bench_mosaicml_read())

    print_summary(results, N)

    with open(BASE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {BASE / 'results.json'}")


if __name__ == "__main__":
    main()
