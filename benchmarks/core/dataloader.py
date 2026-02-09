"""DataLoader benchmark: ChiniDataset vs MosaicML."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader
from utils import BASE


def bench_chinidataset_dataloader():
    from chinidataset import StreamingDataset

    results = {}
    pq_path = BASE / "chinidataset_data"

    for nw in [0, 2, 4]:
        ds = StreamingDataset(local=str(pq_path))
        loader = DataLoader(ds, batch_size=32, num_workers=nw)
        t0 = time.perf_counter()
        count = sum(len(b["label"]) for b in loader)
        elapsed = time.perf_counter() - t0
        print(f"  ChiniDataset DataLoader (w={nw}): {count:,} samples | {count/elapsed:,.0f} samples/s")
        results[f"chinidataset_dl_w{nw}"] = {"time": elapsed, "throughput": count / elapsed}

    return results


def bench_mosaicml_dataloader():
    from streaming import StreamingDataset as MosaicDS

    results = {}
    mds_path = BASE / "mosaicml_data"

    for nw in [0, 2, 4]:
        ds = MosaicDS(local=str(mds_path), shuffle=False, batch_size=32)
        loader = DataLoader(ds, batch_size=32, num_workers=nw)
        t0 = time.perf_counter()
        count = sum(len(b["label"]) for b in loader)
        elapsed = time.perf_counter() - t0
        print(f"  MosaicML DataLoader (w={nw}): {count:,} samples | {count/elapsed:,.0f} samples/s")
        results[f"mosaicml_dl_w{nw}"] = {"time": elapsed, "throughput": count / elapsed}

    return results


def main():
    print("\n== DataLoader Benchmark ==")
    results = bench_chinidataset_dataloader()
    results.update(bench_mosaicml_dataloader())

    import json
    with open(BASE / "dataloader_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {BASE / 'dataloader_results.json'}")


if __name__ == "__main__":
    main()
