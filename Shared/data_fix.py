# this script is used to re-export corupt parquet files. 

import os

import duckdb

out_dir = "../data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "state_boundaries.parquet")

with duckdb.connect(r"F:\npongo Dropbox\benjamin clark\CIL\Products\ShareableData\sql_to_duckdb\india_agriculture_census_ghg_results_v2.duckdb") as conn:
    conn.execute("install spatial")
    conn.execute("load spatial")
    conn.execute(f"""
        COPY state_boundaries
        TO '{out_path}' (FORMAT PARQUET)
    """)

print(f"Saved to {out_path}")


with duckdb.connect(database=":memory:") as con:
    con.execute("install spatial")
    con.execute("load spatial")
    con.execute("SET default_collation = 'nocase';")
    # Create an in-memory DuckDB connection
    # Load the India national and state boundaries
    state_df = con.execute("SELECT * EXCLUDE geog, (CAST(geog AS string)) AS geog FROM read_parquet('../data/state_boundaries.parquet')").fetch_df()
state_df.head(3)


#fix bad geometries


with duckdb.connect(database=":memory:") as con:
    con.execute("install spatial")
    con.execute("load spatial")
    con.execute("SET default_collation = 'nocase';")
    con.execute("CREATE OR REPLACE TABLE state_boundaries AS SELECT * FROM read_parquet('../data/state_boundaries.parquet')")
    con.execute("""with a as 
                (
                    select ST_CollectionExtract(cast(CAST(geog AS string) AS geometry), 3) as geog_fix 
                    FROM read_parquet('../data/state_boundaries.parquet') where state_code in ('RJ')
                )
                update state_boundaries set geog = geog_fix 
                from a where state_code in ('RJ')
                """)
    con.execute("copy state_boundaries to '../data/state_boundaries.parquet' (format 'parquet')")
    df = con.execute("select * exclude geog, cast(geog as string) as geog FROM read_parquet('../data/state_boundaries.parquet') where state_code in ('RJ')").fetch_df()
print(df.loc[0, 'geog'])