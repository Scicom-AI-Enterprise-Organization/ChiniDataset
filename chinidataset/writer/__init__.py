"""Writer classes for creating streaming dataset shards."""

from chinidataset.writer.base import Writer
from chinidataset.writer.parquet import ParquetWriter

__all__ = ['Writer', 'ParquetWriter']
