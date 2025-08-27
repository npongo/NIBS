#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import sys
sys.path.extend(["..\\"])
import math as m
import pandas as pd
from  Shared.NIBSData2 import *
from  plotnine import ggplot, aes, labs
from  os import path
import warnings
import matplotlib.pyplot as plt
from IPython.display import display

# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"b:\Repos\NIBS\Data" , 
                chart_dir=r"b:\Repos\NIBS\Graphs")

crops = ['Rice','Wheat','Maize']
refs = ['shcherbak_2014', 'ipcc_2019', 'eagle_2020', 'bhatia_2013']

#%%
# low_color = "green"
# mid_color= "yellow"
# high_color = "red"
# colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
#colors = ['#81C000',  '#F2F900',  '#FFE700',  '#FF8F00',  '#FF0500']

#%%
dfs = []
for ref in refs:
    for crop in crops:
        sql = f"""select * exclude geog, cast(geog as string) as geog 
        from read_parquet('{path.join(nibs.data_dir,f'vwM_district_n2o_co2e_apy_crop_results_{ref}.parquet')}')
        where apy_crop = '{crop.lower()}'
            and gwp_time_period = 100
        """
        df = nibs.load_map_data(sql)
        dfs.append((ref, crop, df))
        df.head()
        
# %%     
graphs = list()
units = "$\\frac{Kg\ CO_2e_{100}} {Kg\ Yield}$"
for ref, crop, df in dfs:
    df.head()
    file_name = f"map_{crop}_kg_no2_co2e100_kg_yield_{ref}.svg"
    title = f"\n{crop} N$_2$O Emissions By Yield\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace('Ipcc','IPCC')})"
    g = nibs.percentile_map(df
               , 'kg_n2o_kg_yield'
               , title
               , units
               , file_name
               ,round = 3
               ,legend_position=(.7,.1)
               )
    graphs.append((ref, crop, g))
    g.show()
    
# %%
spearman_dfs = []
for crop in crops:
    spearman_sql  = f"""select * exclude geog, cast(geog as string) as geog  
    from read_parquet('{path.join(nibs.data_dir,f'vwM_district_rank_n2o_yield_summary_all_models.parquet')}')
    where apy_crop = '{crop}'"""
    df =  nibs.load_map_data(spearman_sql)
    spearman_dfs.append((crop, df))


#%%
rename_dict = {
    'kg_n2o_kg_yield_bhatia_2013': 'Bhatia\net al. 2013',
    'kg_n2o_kg_yield_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology',
    'kg_n2o_kg_yield_eagle_2020': 'Eagle et al.\n2020',
    'kg_n2o_kg_yield_shcherbak_2014': 'Shcherbak et al.\n2014'
}

grouped_spearman_corrs = []
for crop, df in spearman_dfs:
    file_name = f"n2o_yield_{crop}_spearman_correlation_matrix.svg"
    title = f"\n{crop} Spearman Correlation of District Rankings"


    n2o_ha_spearman_corr_plot = nibs.spearman_plot(df
                , file_name
                , rename_dict
                , title
                , 12
                , 'white'
                ,figure_size=(7,6)
                )

    n2o_ha_spearman_corr_plot.show()

#%%
dfs = []
for ref in refs:
    sql = f"""select  * exclude geog, cast(geog as string) as geog  
    from read_parquet('{path.join(nibs.data_dir,f'vwM_district_max_n2o_6_class_apy_crop_yield_{ref}.parquet')}')
    """
    df =  nibs.load_map_data(sql)
    df = nibs.apy_crop_6_class_to_catagorical(df,['max_apy_crop'])
    dfs.append((ref, df))

# %%
crop_colors = {
    'Rice': 'green',  
    'Small millets': 'yellowgreen',  
    'Arhar/Tur':  'yellow',  
    'Urad': 'orange',  
    'Dry chillies':'#FF8F60' , 
    'Oilseeds':'#cc5500', 
    'Soyabean': '#ff6500',
    'Arecanut': 'Red',  
    'Rapeseed & Mustard': '#cb4154',  
    'Green Gram': 'darkred',  
    'Other Crops': 'tan',  
}

for ref,  df in dfs:
    title = f'Crop with largest $N_2O$ emissions per yield\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})'
    file_name = f"ap_max_n2o_6_class_apy_crop_yield_{ref}.svg"
    g_c = nibs.manual_catagorical_map(df
                        , 'max_apy_crop'
                        ,  title
                        , 'Crop'
                        , crop_colors
                        , file_name= file_name
                        , legend_position=(.7,.02)
                        ,dpi=nibs.dpi)
    g_c.show()


# %%
cramer_v_sql = f"""select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,f'vwM_district_max_n2o_apy_crop_yield_all_models.parquet')}')"""
cramer_v_df =  nibs.load_map_data(cramer_v_sql)

#%%
rename_dict = {
    'apy_crop_bhatia_2013': 'Bhatia\net al., 2013',
    'apy_crop_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology',
    'apy_crop_eagle_2020': 'Eagle\net al.,2020',
    'apy_crop_shcherbak_2014': 'Shcherbak\net al.,2014',
}
ha_cramers_v = nibs.cramers_v_plot(cramer_v_df
                    , "apy_crop_kg_kg_yield_cramers_v_matrix.svg"
                    , rename_dict
                    , 'District crops with highest yield scaled $N_2O$ emission\n($\\frac{Kg\ N_2O}{Kg\ Yield}$ Cramér\'s V)'
                    , legend_title="Cramér\'s V"
                    ,figure_size=(7,6)
                    )
ha_cramers_v.show()
