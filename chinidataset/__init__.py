"""ChiniDataset: Parquet streaming dataset library for ML training.

Write Parquet shards with ParquetWriter, read them with StreamingDataset.

Example:
    >>> from chinidataset import ParquetWriter, StreamingDataset
    >>>
    >>> # Write
    >>> with ParquetWriter(out="./data", columns={"x": "float32[]", "y": "int32"}) as w:
    ...     w.write({"x": [1.0, 2.0], "y": 0})
    >>>
    >>> # Read
    >>> dataset = StreamingDataset(local="./data", shuffle=True, batch_size=32)
    >>> for sample in dataset:
    ...     print(sample)
"""

from chinidataset.dataset import StreamingDataset
from chinidataset.util import merge_index
from chinidataset.writer import ParquetWriter

__all__ = ['ParquetWriter', 'StreamingDataset', 'merge_index']
__version__ = '0.1.0'
