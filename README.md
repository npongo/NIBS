# Potential for reducing greenhouse gas emissions from cropland in India: where, Which(crop), and Who(farmers) 

This respository stores code files used to generate the graphs for the paper along with example notebooks for how to access and use the raw data and documentation. The raw data is stored in a duckdb database and is ~178Gib in size. Due to the size it is hosted on dropbox and can be downloaded by https://www.dropbox.com/scl/fi/smggq1ewhi07h0jzu5spq/india_agriculture_census_ghg_results_v2.duckdb?rlkey=ipch2mku8rtb0x1vo08xqdr9y&dl=1. The tables and views in the database have been exported to parquet files so they can be directly accessed from the web. Links to each table/view parquet file are in the excel spreadsheet that documents all the database tables and views located in the documents file. 


##Examples Folder
The examples folder contains jupitor notebooks that showcase how to use the data to identify potenial district for GHG emission reduction. 

##Paper Folder
The paper folder contains the code files used to generate the graphs for the published paper. 

##Documentation Folder
The documentation folder contain the documentation for the database and document with detailed descriptions of the methods.

##Accessing the raw data 
Retrieve data for district nitrogen balance. 
For more examples code look in the examples fold.
### Python
```python
import duckdb 

with duckdb.connect("https://www.dropbox.com/scl/fi/smggq1ewhi07h0jzu5spq/india_agriculture_census_ghg_results_v2.duckdb?rlkey=ipch2mku8rtb0x1vo08xqdr9y&raw=1",readonly=True) as conn:
    sql = ""
    df = con.execute(sql).to_df()
    con.close

df.head(5)

```
    
### R
```r
library(duckdb)

# Create a connection to an in-memory DuckDB database
con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")

# Create a table and insert some data
dbExecute(con, "CREATE TABLE items (id INTEGER, name STRING, price DOUBLE)")
dbExecute(con, "INSERT INTO items VALUES (1, 'Apple', 0.99), (2, 'Banana', 0.59), (3, 'Cherry', 2.99)")

# Query the table
result <- dbGetQuery(con, "SELECT * FROM items")
print(result)
```
## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
