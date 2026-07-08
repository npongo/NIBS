# Potential for reducing greenhouse gas emissions from cropland in India: where, Which(crop), and Who(farmers) 

This respository stores code files used to generate the graphs for the **[paper](https://iopscience.iop.org/article/10.1088/1748-9326/ae499c)** along with example notebooks for how to access and use the raw data and documentation. The raw data is stored in a duckdb database and is ~178Gib in size. Due to the size it is hosted on dropbox and can be downloaded **[here]( https://www.dropbox.com/scl/fi/smggq1ewhi07h0jzu5spq/india_agriculture_census_ghg_results_v2.duckdb?rlkey=ipch2mku8rtb0x1vo08xqdr9y&st=lawcyar0&dl=1)**. The tables and views in the database have been exported to parquet files so they can be directly accessed from the web. Links to each table/view parquet file are in the excel spreadsheet that documents all the database tables and views located in the documents file. 

## Examples Folder
The examples folder contains jupitor notebooks that showcase how to use the data to identify potenial district for GHG emission reduction. 

## Paper Folder
The paper folder contains the code files used to generate the graphs for the published **[paper](https://iopscience.iop.org/article/10.1088/1748-9326/ae499c)**. 

## Documentation Folder
The documentation folder contain the documentation for the database and document with detailed descriptions of the methods.

### Names convension of data files
The naming convension of the data files is as follows:
<view prifix><admin level><GHG gas><aggregation variable><crop><name><model>
|**Part**|**Description**|
|-------------------------------|----------|
|**view prefix**|
|vwA_ | Analystical view, packages raw table data in a analytical ready manner. Use these views preferablity to raw table data.|
|vwM_ | Map view. These data tables can be used to draw maps of the associated variables. They are typically aggregated but may require filtering by GWP, model or other variables before there is one record per mapping unit. They will have a admin level that is either district or national that indicates what the spatial mapping unit is. |
|vwG_ | Graph view. These datasets support the creation of graphs other than maps. They contain tablular data. |
|vwR_ | Report view. These dataset have prepackaged information for a report. |
|**admin level**|
|district | District level. The data represents a district estimate. |
|national | National level. The data is aggregated at national level |
|**GHG gas**|
|n2o| Nitrous oxide from nutrient management and the additions of synthetic and organic fertilizer. |
|ch4| Methane from rice production | 
|Residue | $N_2O$ and $CH_4$ from the burning of crop residues.|
|n2o_n | Converted to $N_2O-N$ as the unit.
|n2o_co2e | $N_2O$ converted to $CO_2e$ based on AR6 GWP values. The gwp_time_period should be filtered to eithert 100 or 20. |
|ch4_co2e | $CH_4$ converted to $CO_2e$ GWP base on AR6 GWP values. The gwp_time_period should be filtered to eithert 100 or 20. |
|**aggregation variable**|
|farm_size | Data aggregated across the farm size categorical variable. |
|irrigated | Data aggregated across the irrigation status categorical variable, irrigated or rainfed. |
|apy_crop| Data aggregated across the crops reported in the APY dataset.|
|**crop**|
|rice| Data is for rice only.|
|upload_crop | Data is for crops other than rice. |
|**name**| Descriptive name.|
|**model**|
|bhatia_2013| Data estimated using the Bhatia et al. 2013 EF models for $N_2O$ and $CH_4$.
|IPCC_2019| Data estimated using the EF IPCC updated methodology in 2019 models for $N_2O$ and $CH_4$.
|Eagle_2020| Data estimated using the model from Eagle et al. 2020 for $N_2O$.
|Shcherbak_2014| Data estimated using the non-linear model from Shcherbak et al. 2014|
|hiroko_akiyama_2005| Data estimated using the water regime EF model from Akiyama et al. 2005|
|karen_2021| Residue burn emission estimated using the RPR functions from Karen et al. 2021. If a residue burning dataset does not end in Karen_2021 then the RPR data is from the Biomass Altas v2.0 the suffix 28 means the data represents the Biomass Altas v2.0 results limited to the coresponding 28 crops in the Karen et al 2021 data results.|
|all_models| Contains results from all models and may need filtering by the model variable.|

**NOTE:** All analysis should be done with datasets starting with "vw[A,G,M]". Most datasets will need filtering and/or aggretating to achive the final result.
	
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
