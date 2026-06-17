  
#%%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
sys.path.extend(["..\\"])
from Shared.NIBSData2 import *
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"B:\Repos\NIBS\Data" , chart_dir=r"B:\Repos\NIBS\Graphs")
crops = ['Rice','Wheat','Maize']
ref = 'eagle_2020'

#%%
sql = f"""select *  exclude geog, cast(geog as string) as geog
        from read_parquet('{path.join(nibs.data_dir,'vwM_district_n_balance_results_eagle_2020.parquet')}') 
        where gwp_time_period = 100
        """
df =nibs.load_map_data(sql)

#%%

g_kg_ha = nibs.percentile_map(df
               , 'mean_n_balance_n_kg_ha'
               , "All Crops Per Hectare Nitrogen Balance\n(Eagle et al., 2020)"
               , "$Kg\ N\ Ha^{-1}$"
               , "map_n_balance_all_crops_kg_ha_eagle_2020.png"
               ,legend_position=(.7,.1)
               )
   
g_kg_ha.show()

g_total = nibs.percentile_map(df
               , 'total_n_balance_n_Gg_n_map'
               , "All Crops Total Nitrogen Balance\n(Eagle et al., 2020)"
               , "Gg N"
               , "map_n_balance_all_crops_total_Gg_n_eagle_2020.png"
               ,legend_position=(.7,.1)
               )
   
g_total.show()

#%%

dfs = []
for crop in crops:
    sql = f"""select * exclude geog, cast(geog as string) as geog
    from read_parquet('{path.join(nibs.data_dir,f'vwM_district_n_balance_apy_crop_results_eagle_2020.parquet')}') 
    where apy_crop = '{crop}'
        and gwp_time_period = 100
    """
    df = nibs.load_map_data(sql)
    dfs.append((ref, crop, df))

#%%
for ref, crop, df in dfs:
    file_name = f"map_{crop}_n_bal_kg_ha_eagle_2020.png"
    title = f"{crop} Per Hectare Nitrogen Balance\n(Eagle et al., 2020)"
    g_ha = nibs.percentile_map(df
               , 'mean_n_balance_n_kg_ha'
               , title
               ,  "$Kg\ N\ Ha^{-1}$"
               , file_name
               ,legend_position=(.7,.1))
    g_ha.show()
    
    file_name = f"map_{crop}_n_bal_total_Gg_n_eagle_2020.png"
    title = f"{crop} Total Nitrogen Balance\n(Eagle et al., 2020)"
    g_t = nibs.percentile_map(df
               , 'total_n_balance_n_Gg_n_map'
               , title
               ,  "$Kg\ N\ Ha^{-1}$"
               , file_name
               ,legend_position=(.7,.1)
               ,format="eps"
               )
    
    g_t.show()

# %%
crop_colors = {
    'Rice': 'green',  
    'Wheat': 'yellowgreen',  
    'Maize':  'yellow',  
    'Cotton': 'orange',  
    'Dry chillies':'#FF8F60' ,
    'Bajra': 'Red',  
    'Ragi': '#cb4154',  
    'Other Crops': 'Brown',  
}


#%%

sql = f"""select * exclude geog, cast(geog as string) as geog    
    from read_parquet('{path.join(nibs.data_dir,f'vwM_district_max_n_bal_6_class_apy_crop_summary_eagle_2020.parquet')}') 
"""
df =nibs.load_map_data(sql)
df = nibs.apy_crop_6_class_to_catagorical(df, ['mean_n_balance_n_kg_ha_apy_crop','total_n_balance_n_kg_ha_apy_crop'])

#%%
g_ha = nibs.manual_catagorical_map(df
                            , 'mean_n_balance_n_kg_ha_apy_crop'
                            , 'Crop with largest Nitrogen Balance\nper Hectare (Eagle et al., 2020)'
                            , 'Crop'
                            , crop_colors
                            , 'map_max_n_bal_6_class_apy_kg_ha_mean_eagle_2020.png'
                            , legend_position=(.7,.1))
g_ha.show()

g_t = nibs.manual_catagorical_map(df
                            , 'total_n_balance_n_kg_ha_apy_crop'
                            , 'Crop with largest Total Nitrogen Balance\n(Eagle et al., 2020)'
                            , 'Crop'
                            , crop_colors
                            , 'map_max_n_bal_6_class_apy_total_mean_eagle_2020.png'
                            , legend_position=(.7,.1))
g_t.show()

# %%
