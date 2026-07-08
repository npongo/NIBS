"""
nibs_helper.py — lightweight loader/query helper for the NIBS India cropland GHG
parquet dataset (https://github.com/npongo/NIBS).

This is a trimmed-down cousin of the repo's own Shared/NIBSData2.py: it keeps the
download-cache-then-query pattern but drops the hard geopandas/plotnine/scipy
dependencies so it works for plain tabular analysis. Reach for the real
NIBSData2.py (bundled in the repo, not here) only when you need the choropleth
map-drawing helpers.

Typical usage:

    from nibs_helper import NIBS

    nibs = NIBS(data_dir="./nibs_data")
    nibs.search("residue_burning")               # find candidate dataset names
    df = nibs.query('''
        select district, apy_crop, kg_co2e_ha, total_Gg_co2e_map
        from read_parquet('vwM_district_residue_burning_co2e')
        where gwp_time_period = 100
    ''')

`query()` scans the SQL for read_parquet('<dataset_name>') references (with or
without a .parquet suffix or a full path), resolves each one against the
bundled URL catalog, downloads it to `data_dir` on first use, and rewrites the
SQL to point at the local cached file before executing it with duckdb.
"""
from __future__ import annotations

import json
import re
from os import path, makedirs
from typing import Optional

import duckdb
import pandas as pd

_CATALOG_PATH = path.join(path.dirname(__file__), "..", "references", "nib_urls.json")


class NIBS:
    def __init__(self, data_dir: str = "./nibs_data", catalog_path: str = _CATALOG_PATH):
        self.data_dir = data_dir
        if not path.exists(self.data_dir):
            makedirs(self.data_dir)
        with open(catalog_path, encoding="utf-8") as f:
            self._urls: dict[str, str] = json.load(f)
        self._ci_keys = {k.lower(): k for k in self._urls}

    # ------------------------------------------------------------------
    # catalog lookup
    # ------------------------------------------------------------------
    def resolve_key(self, name: str) -> str:
        """Match a dataset name (case-insensitive, .parquet suffix or full path
        tolerated) to its canonical catalog key. Raises ValueError if not found."""
        base = path.basename(name)
        if base.lower().endswith(".parquet"):
            base = base[: -len(".parquet")]
        if base.lower() in self._ci_keys:
            return self._ci_keys[base.lower()]
        raise ValueError(
            f"Dataset '{name}' not found in the NIBS catalog. "
            f"Use NIBS.search('{base}') to find close matches."
        )

    def get_url(self, name: str) -> str:
        return self._urls[self.resolve_key(name)]

    def search(self, keyword: str) -> list[str]:
        """Case-insensitive substring search over the ~300 dataset names.
        Use this whenever you're not 100% sure of the exact dataset name —
        it's much cheaper than guessing and re-running failed queries."""
        kw = keyword.lower()
        return sorted(k for k in self._urls if kw in k.lower())

    def all_datasets(self) -> list[str]:
        return sorted(self._urls.keys())

    # ------------------------------------------------------------------
    # download + cache
    # ------------------------------------------------------------------
    def ensure_local(self, name: str) -> str:
        """Download the dataset to data_dir if not already cached, return the
        local path. Raises a clear error (rather than a raw connection
        traceback) if the environment can't reach Dropbox — sandboxes often
        block it — so the caller can fall back to asking the user for a
        manually-downloaded copy."""
        key = self.resolve_key(name)
        local_path = path.join(self.data_dir, f"{key}.parquet")
        if path.exists(local_path):
            return local_path
        url = self._urls[key]
        try:
            import requests

            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            raise RuntimeError(
                f"Could not download '{key}' from Dropbox ({e}). This environment "
                "may block outbound access to dropbox.com. Ask the user to download "
                f"the file from {url} and either place it at {local_path} or tell you "
                "the local path so you can point read_parquet() at it directly."
            ) from e
        return local_path

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    _READ_PARQUET_RE = re.compile(r"read_parquet\(\s*'([^']+)'\s*\)", re.IGNORECASE)

    def _localize_sql(self, sql: str) -> str:
        def repl(m: re.Match) -> str:
            token = m.group(1)
            # already a local path that exists (or looks like one) - leave alone
            if path.exists(token):
                return m.group(0)
            local_path = self.ensure_local(token)
            return f"read_parquet('{local_path}')"

        return self._READ_PARQUET_RE.sub(repl, sql)

    def query(self, sql: str, geom_column: Optional[str] = "geog") -> pd.DataFrame:
        """Run a duckdb SQL query against one or more NIBS datasets, referenced
        by short name inside read_parquet(...). Downloads/caches as needed.

        If the result has a `geog` column (spatial geometry, WKT-ish binary),
        it's cast to a plain string automatically so plain pandas / duckdb can
        handle it — build a GeoDataFrame yourself afterwards if you need real
        spatial ops (`gpd.GeoSeries.from_wkt(df[geom_column])`).
        """
        local_sql = self._localize_sql(sql)
        con = duckdb.connect(database=":memory:")
        con.execute("SET default_collation = 'nocase';")
        df = con.execute(local_sql).fetch_df()
        if geom_column and geom_column in df.columns and df[geom_column].dtype == object:
            # already came back as bytes/geometry from duckdb spatial types in some
            # exports; leave as-is if it's already a plain string
            pass
        con.close()
        return df

    def describe(self, name: str) -> pd.DataFrame:
        """Return the column schema (name + type) for a dataset without loading
        the whole thing into memory beyond what duckdb needs to read the footer."""
        local_path = self.ensure_local(name)
        con = duckdb.connect(database=":memory:")
        df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{local_path}')").fetch_df()
        con.close()
        return df

    def load(self, name: str) -> pd.DataFrame:
        """Shortcut for `select * from <dataset>`."""
        return self.query(f"select * from read_parquet('{name}')")
