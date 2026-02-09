"""Full IMDB benchmark: write, read, dataloader — ChiniDataset vs MosaicML."""

import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import BASE, human_bytes, load_imdb
from core.write import bench_chinidataset_write, bench_mosaicml_write
from core.read import bench_chinidataset_read, bench_mosaicml_read
from core.dataloader import bench_chinidataset_dataloader, bench_mosaicml_dataloader


def print_summary(results, N):
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"\nDataset: imdb test ({N:,} samples, text + label)\n")
    print(f"{'Metric':<25} {'MosaicML (MDS)':>18} {'ChiniDataset (PQ)':>18} {'Speedup':>10}")
    print("-" * 75)

    def row(label, mds_key, pq_key, fmt="throughput"):
        mds = results.get(mds_key)
        pq = results.get(pq_key)
        if not mds or not pq:
            return
        if fmt == "throughput":
            speedup = pq["throughput"] / mds["throughput"] if mds["throughput"] else 0
            print(f"{label:<25} {mds['throughput']:>14,.0f}/s {pq['throughput']:>14,.0f}/s {speedup:>9.1f}x")
        elif fmt == "time":
            speedup = mds["time"] / pq["time"] if pq["time"] else 0
            print(f"{label:<25} {mds['time']:>15.3f}s {pq['time']:>15.3f}s {speedup:>9.1f}x")

    row("Write", "mosaicml_write", "chinidataset_write")
    if results.get("mosaicml_write") and results.get("chinidataset_write"):
        print(f"{'File size':<25} {human_bytes(results['mosaicml_write']['size']):>18} {human_bytes(results['chinidataset_write']['size']):>18}")
        print(f"{'Shards':<25} {results['mosaicml_write']['shards']:>18} {results['chinidataset_write']['shards']:>18}")
    row("Parallel write+merge", "mosaicml_parallel", "chinidataset_parallel")
    row("Read (sequential)", "mosaicml_read", "chinidataset_read")
    row("Read (shuffled)", "mosaicml_shuffle", "chinidataset_shuffle")
    row("DataLoader (w=0)", "mosaicml_dl_w0", "chinidataset_dl_w0")
    row("DataLoader (w=2)", "mosaicml_dl_w2", "chinidataset_dl_w2")
    row("DataLoader (w=4)", "mosaicml_dl_w4", "chinidataset_dl_w4")
    row("Read (merged)", "mosaicml_merge_read", "chinidataset_merge_read")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    hf_ds = load_imdb()
    N = len(hf_ds)
    results = {}

    print("\n== Write ==")
    results.update(bench_chinidataset_write(hf_ds))
    results.update(bench_mosaicml_write(hf_ds))

    print("\n== Read ==")
    results.update(bench_chinidataset_read())
    results.update(bench_mosaicml_read())

    print("\n== DataLoader ==")
    results.update(bench_chinidataset_dataloader())
    results.update(bench_mosaicml_dataloader())

    print_summary(results, N)

    with open(BASE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {BASE / 'results.json'}")

    shutil.rmtree(BASE)
    print("Cleaned up.")


if __name__ == "__main__":
    main()
