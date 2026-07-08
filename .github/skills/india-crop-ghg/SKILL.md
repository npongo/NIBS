---
name: india-crop-ghg
description: Analyze the NIBS India cropland greenhouse-gas dataset (github.com/npongo/NIBS) — ~300 parquet files covering CH4 (rice paddy methane), N2O (fertilizer-induced, rice/upland crops), nitrogen balance, and residue-burning CO2e emissions by Indian district, crop, and farm size, built on Agriculture Census/APY data. Use whenever the user mentions NIBS, India crop/cropland GHG or emissions data, Indian district- or state-level agricultural emissions, rice paddy methane in India, N2O from Indian fertilizer use, residue/stubble burning emissions in India, or a .parquet file from this dataset (vwM_district_*, vwG_national_*, ch4_bhatia_2013, n2o_ipcc_2019, apy, etc.) — even without saying "NIBS" explicitly. Also use when picking, downloading, or querying these files, or choosing between multiple published emission models.
license: CC-BY-SA-4.0
---

# NIBS: India Cropland GHG Emissions Data

This dataset is the output of a specific research project (Clark et al. 2026,
*Environ. Res. Lett.* 21 064027, "Potential for reducing greenhouse gas
emissions from cropland in India: where, which (crop), and who (farmers)").
It estimates GHG emissions from Indian cropland at the district level, running
**several independently published emission-factor models per gas in parallel**
rather than trusting a single one. That's the single most important thing to
keep in mind: almost every question ("how much methane does rice produce in
Punjab") has more than one defensible answer depending on which model backs
it, and part of doing this analysis well is surfacing that rather than picking
one number silently.

This skill lives inside the NIBS repo itself (CC BY-SA 4.0), alongside
`Shared/NIBSData2.py` and `Shared/NIBSUrls.py`, which are the canonical
source this skill's bundled `references/nib_urls.json` was derived from. If a
task needs the choropleth map-drawing helpers (geopandas + plotnine), use
`Shared/NIBSData2.py` directly rather than this skill's lighter-weight
`scripts/nibs_helper.py`, which only covers plain tabular query/download.

## Workflow

1. **Clarify scope before writing any query.** You need four things, and it's
   fine to ask directly rather than guess:
   - **Gas / topic**: methane (rice), N2O (rice or upland crops), nitrogen
     balance, or residue burning?
   - **Geographic level**: national total, state, or district (map-level)?
   - **Breakdown**: by crop, by farm size, by irrigation status, or none?
   - **Which model(s)**: a specific published model, or a comparison across
     all of them? If the user doesn't have a preference, default to the
     `_all_models` summary view where one exists so you're not silently
     picking a model for them.

   Read `references/dataset_catalog.md` for the full naming grammar
   (`<prefix><admin level><gas><aggregation variable><crop><name><model>`,
   per this repo's own README) before picking file names. The short version:
   **always prefer a dataset starting with `vwA_`, `vwG_`, or `vwM_` over a
   raw/base table** — `vwM_` is for map/district-level work (has a `geog`
   column), `vwG_` is for chart-ready tabular summaries, `vwA_` is the
   general-purpose analysis-ready tier. Raw tables are inputs and
   intermediate model results, not what a normal "how much GHG" question
   needs.

2. **Find the right dataset name(s).** Use `scripts/nibs_helper.py`'s
   `NIBS().search(keyword)` to search the ~300 dataset names rather than
   guessing full names from memory — they're long and the differences between
   similar ones (which model, which crop subset, farm-size vs irrigation
   breakdown) matter. `NIBS().all_datasets()` lists everything.

3. **Check the schema before querying.** Column names are close-but-not-
   identical across files. Run `NIBS().describe(dataset_name)` and read the
   real column list rather than assuming a per-hectare column is called
   `kg_co2e_ha` vs `total_emissions_kg_co2e_ha` etc.

4. **Query with duckdb.** `NIBS().query(sql)` accepts SQL with
   `read_parquet('dataset_name')` (short name, no path or extension needed) —
   it resolves each reference against the URL catalog, downloads and caches
   the file locally on first use, and rewrites the SQL to point at the local
   copy. Almost every gas view has a `gwp_time_period` column (100 or 20-year
   GWP horizon) — filter on it explicitly and say in your answer which
   horizon you used, since CH4 results especially shift a lot between them.

   ```python
   from nibs_helper import NIBS
   nibs = NIBS(data_dir="./nibs_data")
   df = nibs.query("""
       select district, apy_crop, kg_co2e_ha, total_Gg_co2e_map
       from read_parquet('vwM_district_residue_burning_co2e')
       where gwp_time_period = 100
   """)
   ```

5. **Handle the spatial column if present.** `vwM_*` views carry a `geog`
   column (geometry). For tabular analysis just leave it or drop it
   (`select * exclude geog`). For an actual choropleth map, cast it to WKT and
   build a GeoDataFrame:
   ```python
   import geopandas as gpd
   gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geog"]), crs="EPSG:4326")
   ```
   `national_boundaries` and `vwM_india_states` are the background/outline
   layers used for context in maps.

6. **If a download fails**, the environment running this may not have
   outbound access to dropbox.com (some sandboxed or restricted-network setups
   block it). `nibs_helper.py` raises a clear error for this rather than a raw
   connection traceback; when it happens, tell the user which URL you need and
   ask them to either download it manually and share the local path, or run
   the analysis somewhere with open network access.

7. **Present results appropriately for what was asked**: a quick number or
   small table can just go in chat; anything the user will want to keep,
   re-sort, or hand to someone else should become a real file.

## Bundled resources

- `references/nib_urls.json` — dataset name → Dropbox parquet URL, all ~300
  entries. Source of truth for what data exists and where to get it.
- `references/dataset_catalog.md` — naming conventions, gas/model taxonomy,
  common columns, and a worked example. Read this before picking dataset
  names for anything beyond a trivial single-file query.
- `scripts/nibs_helper.py` — `NIBS` class: search/resolve dataset names,
  download+cache parquet files, run duckdb queries against short dataset
  names, describe schemas. Requires `duckdb`, `pandas`, `requests`. Does
  **not** require geopandas/plotnine — pull those in yourself (or use
  `Shared/NIBSData2.py` instead) only if the task actually needs a map.

## Things to get right

- Don't silently pick one model when several exist for a gas — say which one
  you used, or show the comparison.
- Don't skip the `gwp_time_period` filter — an unfiltered query on a table
  that has both 20- and 100-year rows will double-count / mix incompatible
  numbers.
- Crop names and farm-size categories in the raw data are the original census
  labels (e.g. `'MARGINAL (BELOW 1.0)'`, `'Moong(Green Gram)'`) — clean them up
  for presentation, but don't assume the raw label always matches what you'd
  expect (check `references/dataset_catalog.md`'s categorical section).
- This is peer-reviewed research data with a specific paper behind it — when
  summarizing findings, it's worth a one-line citation back to Clark et al.
  2026 rather than presenting numbers as if they came from nowhere.
