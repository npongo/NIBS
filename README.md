# Potential for reducing greenhouse gas emissions from cropland in India: where, Which(crop), and Who(farmers) 

This respository stores code files used to generate the graphs for the paper along with example notebooks for how to access and use the raw data and documentation. The raw data is stored in a duckdb database and is ~178Gib in size. Due to the size it is hosted on dropbox and can be downloaded **[here]( https://www.dropbox.com/scl/fi/smggq1ewhi07h0jzu5spq/india_agriculture_census_ghg_results_v2.duckdb?rlkey=ipch2mku8rtb0x1vo08xqdr9y&st=lawcyar0&dl=1)**. The tables and views in the database have been exported to parquet files so they can be directly accessed from the web. Links to each table/view parquet file are in the excel spreadsheet that documents all the database tables and views located in the documents file. 

## Examples Folder
The examples folder contains jupitor notebooks that showcase how to use the data to identify potenial district for GHG emission reduction. 

## Paper Folder
The paper folder contains the code files used to generate the graphs for the published paper. 

## Documentation Folder
The documentation folder contain the documentation for the database and document with detailed descriptions of the methods.

## Accessing the raw data 
The raw data is stored in a number of parquet files that are publicly available using the URLs that can be located in the **[database documentation excel](Documentation/database%20documentation.xlsx)** file located in the Documents directory of this repository. The parquet files can be opended in excel use the method presented **[here](https://medium.com/@simon.peter.mueller/excel-parquet-integration-mastering-data-analysis-with-duckdb-6a6a6b773128)**. See the examples below to use DuckDB in python or R to access the raw files for analysis. Under the **[Shared folder](Shared/)** there are helper classes to make accessing and analysing the data easier with python. Also in the Examples directory there are tutorial jupitor notebooks that demonstatred using the data to identify target districts for emissions reductions. 

Retrieve data for district nitrogen balance. 
For more examples code look in the examples fold.
### Python: Example loading a spatial parquet data file.
```python
import duckdb 

crs = "EPSG:4326"
table_path = "<local path to table_name.parquet>"  # use the excel sheet in the documentation fold to download the required .parquet file then supply its local path here.
with duckdb.connect(database=":memory:") as con:
    con.execute("install spatial")
    con.execute("load spatial")
    # load a spatial dataset
    df = con.execute(f"""
                     SELECT exclude geog *
                     ,  cast(geog as string) as geog_str  
                     FROM read_parquet({table_path})
                     """).fetch_df()
    gdf = gpd.GeoDataFrame(df,geometry= gpd.GeoSeries.from_wkt(df[geom_column]),crs=crs)

gdf.head()

```

### Python: Example using the helper class to manage local cacheing and dataset loading
```python
#clone the respository 
import duckdb 
from Shared import NIBSData2
nibs = NIBSData2.NIBSData()  # defualt initiliztion will create folders within the working directory to cache parquet files locally after first use.

try:
    sql = f"""select * 
    from read_parquet('{path.join(nibs.data_dir,'vwG_national_ch4_summary_all_models.parquet')}')"""  #allows for customization of sql with where clauses. 
    df = nibs.query_dataset(sql) # take sql statement to allow for where clauses. Use nibs.load_dataset to load a dataset by name only.
except Exception as e:
    print("Error loading national CH4 summary data:", e)

df.head()
```
    
### R: Example loading a table dataset
```r
library(duckdb)

# Create a connection to an in-memory DuckDB database
con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")


# Query the table
result <- dbGetQuery(con, "SELECT * FROM read_parquet(<path to parquet file>)")
print(result)
```


## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
