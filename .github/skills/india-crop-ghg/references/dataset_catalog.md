# NIBS dataset catalog — how the ~300 parquet files are organized

Source repo: https://github.com/npongo/NIBS.
Paper: Clark, 2026, *Environmental Research Letters* 21 064027 — "Potential for
reducing greenhouse gas emissions from cropland in India: where, which (crop),
and who (farmers)". Data licensed CC BY-SA 4.0.

The full database (~178 GiB, DuckDB format) lives on Dropbox. For practical use
every table and view has also been exported as an individual parquet file, also
on Dropbox. `nib_urls.json` in this folder maps every dataset name to its direct
Dropbox download URL (304 entries) — it's a cleaned-up copy of this repo's own
`Shared/NIBSUrls.py`. Don't try to re-derive URLs by pattern-matching; always
look them up in that file (or via `NIBS.search()` / `NIBS.resolve_key()` in
`scripts/nibs_helper.py`).

## Naming grammar (per this repo's own README)

Names follow this token order (not every dataset uses every token):

```
<view prefix> <admin level> <GHG gas> <aggregation variable> <crop> <name> <model>
```

**Rule from the repo author: always prefer datasets starting with `vwA_`,
`vwG_`, or `vwM_` over raw/base tables.** Raw tables are inputs, coefficients,
and intermediate model results — only worth opening if you're checking a
methodology or input assumption, not for an "how much GHG" question. `vwR_` is
niche (one report-prepackaging view) and generally not a first stop either.

| Prefix | Meaning (per repo README) | Typical use |
|---|---|---|
| `vwA_` | **A**nalytical view — packages raw table data in an analysis-ready form. The general-purpose "cleaned up" tier; some are also cross-model comparison tables, but not all. | Default choice for tabular analysis that isn't specifically a map or a chart |
| `vwM_` | **M**ap view — has a `geog` geometry column; admin level (district or national) is aggregated so there's one record per mapping unit once you filter GWP/model/etc. | District-level or choropleth-map questions |
| `vwG_` | **G**raph view — tabular data meant to feed non-map graphs (bar/line charts). Can be national or other aggregation levels, not exclusively national. | Chart-ready summaries, comparisons across a categorical variable (farm size, irrigation, model) |
| `vwR_` | **R**eport view — prepackaged for a specific report output (currently just one: fertilizer nutrients by farm size/crop/irrigation) | Niche |

The second token after the prefix is often the **admin level** — `district` or
`national` — telling you the spatial aggregation directly from the name.

When a user asks a spatial or district-level question, look in `vwM_*` first.
When they want chart-ready comparisons (e.g. across farm size or models), look
in `vwG_*`. `vwA_*` is the safe default for general tabular analysis.

## GHG gas tokens (per repo README)

- `n2o` — nitrous oxide from nutrient management and synthetic/organic fertilizer additions.
- `ch4` — methane from rice production.
- `residue` — N2O and CH4 from crop residue burning.
- `n2o_n` — N2O expressed as N2O-N (nitrogen-mass units, not CO2e).
- `n2o_co2e` — N2O converted to CO2e using **AR6 GWP values**; filter `gwp_time_period` to 100 or 20.
- `ch4_co2e` — CH4 converted to CO2e using **AR6 GWP values**; filter `gwp_time_period` to 100 or 20.

Cutting across all of these: **`gwp_time_period`** is a recurring filter
column (100 or 20 — the AR6 global-warming-potential horizon in years) on any
`_co2e` dataset. Always ask or default to 100-year GWP and say so explicitly,
since results differ a lot between horizons (CH4 especially). Datasets that
aren't CO2e-converted (`n2o_n`, raw `ch4`) don't need this filter.

## The models behind each gas

The project runs **several published emission-factor models per gas** in
parallel (an ensemble), rather than picking one. Always check with the user
which model they want, or default to the `_all_models` view (may need
filtering by a `model` column) and show the spread. Model suffixes, per the
README (actual file names are lowercase, e.g. `ipcc_2019` not `IPCC_2019`):

- `bhatia_2013` — Bhatia et al. 2013 emission-factor model, used for both N2O and CH4.
- `ipcc_2019` — IPCC's updated 2019 methodology, used for both N2O and CH4.
- `eagle_2020` — Eagle et al. 2020 model, used for N2O **and** for the separate nitrogen-balance views — don't confuse the two uses of this suffix.
- `shcherbak_2014` — Shcherbak et al. 2014 non-linear N2O model. (See the catalog quirk below re: a duplicate `shcherback_2014` spelling.)
- `hiroko_akiyama_2005` — Akiyama et al. 2005 water-regime emission-factor model (rice N2O).
- `karan_2021` — residue-burning emissions using Karan et al. 2021's RPR (residue-to-product ratio) functions, limited to 28 crops. **If a residue-burning dataset's name does NOT end in `karan_2021`, its RPR data comes from Biomass Atlas v2.0 instead** — a `_28` suffix there means it's limited to the same 28 crops Karan et al. cover (for comparability), while `_44` (or no crop-count suffix) means the full Biomass Atlas v2.0 crop set.
- `all_models` — contains results from every model at once; will likely need filtering/pivoting on a model-identifying column.

Concrete examples: `vwG_national_ch4_summary_all_models`,
`vwM_district_ch4_summary_bhatia_2013`,
`vwG_national_n2o_co2e_rice_summary_all_models`,
`vwM_district_n_balance_results_eagle_2020`,
`vwM_district_residue_burning_co2e` (Biomass Atlas),
`vwM_district_residue_burning_karan_2021_co2e` (Karan et al.).

## Aggregation variable and crop tokens (per repo README)

Aggregation variable — which categorical dimension the data is broken out by:

- **`farm_size`** — aggregated across farm-size categories from the agriculture
  census: `MARGINAL (BELOW 1.0)`, `SMALL (1.0 - 1.99)`, `SEMI-MEDIUM (2.0 - 3.99)`,
  `MEDIUM (4.0 - 9.99)`, `LARGE (10 AND ABOVE)` (hectares).
- **`irrigated`** — aggregated across irrigation status: irrigated or rainfed.
- **`apy_crop`** — aggregated across crops reported in the APY (Area,
  Production, Yield) agricultural census. Many views repeat once per crop;
  some `_6_class_` views collapse everything outside the top crops into
  `'Other Crops'`.

Crop token — which crop subset the dataset covers (separate from `apy_crop`
above, which means "broken out by crop"; this token instead restricts scope):

- **`rice`** — rice only.
- **`upland_crop`** — crops other than rice. (The README spells this
  `upload_crop` in one place — that's a documentation typo; actual file names
  use `upland_crop`.)

**District / state / spatial geometry** — `vwM_*` views carry district
identifiers plus `geog`; `district_boundaries` / `state_boundaries` /
`vwM_india_states` / `national_boundaries` are the map-background geometry
tables.

## Worked example: "what's the district-level rice methane picture?"

1. Gas = CH4, scope = district → look under `vwM_district_` and `ch4`.
2. `NIBS().search("ch4")` turns up `vwM_district_ch4_summary_bhatia_2013`,
   `_nikolaisen_2023`, `_yan_2005` (per-model) — there's no single
   `_all_models` map view at district level for CH4, only a national one
   (`vwG_national_ch4_summary_all_models`), so if the user wants a model
   comparison at district level use the `vwA_` correlation view
   (`vwA_district_ch4_rice_model_corr_wide`) or query two `vwM_` files and
   join on district.
3. Always filter `where gwp_time_period = 100` (or ask which horizon).
4. Columns typically include a per-hectare rate (`kg_co2e_ha` or similar) and a
   district total (`total_Gg_co2e_map` or similar) — run `NIBS().describe(name)`
   to confirm exact column names before writing the full query rather than
   guessing; naming isn't perfectly consistent across views.

## Known catalog quirk

The two national CO2e summary views for the Shcherbak et al. 2014 N2O model
(`vwG_national_n2o_co2e_rice_summary_*` and
`vwG_national_n2o_co2e_upland_crop_summary_*`) exist **twice under two
spellings** in the source catalog — `shcherbak_2014` and `shcherback_2014` —
both are real, separately-hosted files. Every other Shcherbak view in the
catalog uses the `shcherbak_2014` spelling only. Treat this as a data
provenance question rather than assuming one is a typo to ignore: check both
if you land here, and flag the discrepancy to the user rather than silently
picking one.

## Don't guess column names — introspect

Column names are close-but-not-identical across the ~300 files (e.g. some use
`total_Gg_co2e_map`, others `total_emissions_gg_co2e`). Before building a query
against a dataset you haven't used yet in this conversation, run
`NIBS().describe(dataset_name)` (or `DESCRIBE SELECT * FROM read_parquet(...)`
directly in duckdb) and read the actual column list rather than assuming.
