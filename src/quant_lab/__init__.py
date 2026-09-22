"""quant_lab: a standalone personal crypto quantitative research lab.

The lab is intentionally small and explicit. It moves data through a single
vertical slice::

    historical data -> Parquet -> DuckDB -> Polars -> features
        -> strategy -> VectorBT -> performance report

Nothing here places live orders or talks to production trading systems.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
