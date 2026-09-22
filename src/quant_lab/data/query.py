"""DuckDB + Polars analytics layer.

DuckDB queries Parquet directly — no ingestion step, no server. Polars is the
preferred transformation engine; pandas is only used at library boundaries
(VectorBT, QuantStats).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from .storage import DatasetKey, dataset_dir, list_partitions


def connect(database: str | Path = ":memory:", *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(database), read_only=read_only)


def relation_sql(root: Path, key: DatasetKey) -> str:
    """Return a DuckDB ``read_parquet(...)`` expression for a dataset."""
    glob = dataset_dir(root, key) / "*" / "*.parquet"
    return f"read_parquet('{glob}')"


def query(
    root: Path,
    key: DatasetKey,
    *,
    select: str = "*",
    where: str | None = None,
    group_by: str | None = None,
    order_by: str | None = None,
    limit: int | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Run a small structured query over a normalized dataset via DuckDB.

    This is sugar over :func:`query_sql` for the common shape. For anything
    complex, write the SQL directly.
    """
    sql = f"SELECT {select} FROM {relation_sql(root, key)}"
    if where:
        sql += f" WHERE {where}"
    if group_by:
        sql += f" GROUP BY {group_by}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return query_sql(sql, con=con)


def query_sql(sql: str, *, con: duckdb.DuckDBPyConnection | None = None) -> pl.DataFrame:
    """Execute SQL with DuckDB and return a Polars frame.

    If any referenced Parquet file is missing, returns an empty frame.
    """
    own = con is None
    con = con or connect()
    try:
        result = con.execute(sql)
        return result.pl()
    except duckdb.IOException as exc:  # missing files / bad glob
        raise FileNotFoundError(str(exc)) from exc
    finally:
        if own:
            con.close()


def sql_from_parquet(glob: str, *, columns: list[str] | None = None) -> str:
    cols = ", ".join(columns) if columns else "*"
    return f"SELECT {cols} FROM read_parquet('{glob}')"


def dataset_exists(root: Path, key: DatasetKey) -> bool:
    return len(list_partitions(root, key)) > 0


def count_rows(root: Path, key: DatasetKey) -> int:
    if not dataset_exists(root, key):
        return 0
    df = query(root, key, select="count(*) AS n")
    return int(df["n"][0])


def sql_query_sql(sql: str, *, con: duckdb.DuckDBPyConnection | None = None) -> pl.DataFrame:
    """Alias kept for readability in callers that think in SQL."""
    return query_sql(sql, con=con)


__all__ = [
    "connect",
    "count_rows",
    "dataset_exists",
    "query",
    "query_sql",
    "relation_sql",
    "sql_from_parquet",
]
