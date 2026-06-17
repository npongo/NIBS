#%%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
sys.path.extend(["..\\"])
import math as m
import pandas as pd
from  Shared.NIBSData2 import *
from  plotnine import ggplot, aes, labs
from  os import path
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"b:\Repos\NIBS\Data" , 
                chart_dir=r"b:\Repos\NIBS\Graphs")
# %%  national results 
print("Loading national CH4 summary data...")
try:
    ch4_sql_national = f"""select * 
    from read_parquet('{path.join(nibs.data_dir,'vwG_national_ch4_summary_all_models.parquet')}')"""
    ch4_df_national = nibs.load_table_data(ch4_sql_national)
    ch4_df_national
except Exception as e:
    print("Error loading national CH4 summary data:", e)

#%%
ch4_sql_bhatia_2013 = f"""
select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_rice_ch4_co2e_bhatia_2013.parquet')}')
where gwp_time_period = 100
"""
ch4_df_bhatia_2013 = nibs.load_map_data( ch4_sql_bhatia_2013)
ch4_df_bhatia_2013.head()

#%% 
t_bhatia_2013 = nibs.percentile_map(ch4_df_bhatia_2013
               , 'total_Gg_co2e_map'
               , "Total rice $CH_4$ emissions\n(Bhatia et al., 2013)"
               , "$Gg\ CO_2e_{100}$"
               , "map_total_ch4_emissions_Gg_bhatia_2013.png"
               ,legend_position=(.7,.02)
               )
   
t_bhatia_2013.show()

ha_bhatia_2013 = nibs.percentile_map(ch4_df_bhatia_2013
               , 'kg_ch4_co2e_ha'
               , "Per hectare rice $CH_4$ emissions\n(Bhatia et al., 2013)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_kg_ha_bhatia_2013.png"
               ,legend_position=(.7,.02)
               )
   
ha_bhatia_2013.show()

min_ha_bhatia_2013 = nibs.percentile_map(ch4_df_bhatia_2013
               , 'min_kg_ch4_co2e_ha'
               , "Minimum per hectare rice $CH_4$ emissions\n(Bhatia et al., 2013)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_min_kg_ha_bhatia_2013.png"
               ,legend_position=(.7,.02)
               )
   
min_ha_bhatia_2013.show()

max_ha_bhatia_2013 = nibs.percentile_map(ch4_df_bhatia_2013
               , 'max_kg_ch4_co2e_ha'
               , "Maximum per hectare rice $CH_4$ emissions\n(Bhatia et al., 2013)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_max_kg_ha_bhatia_2013.png"
               ,legend_position=(.7,.02)
               )
   
max_ha_bhatia_2013.show()

# %%
ch4_sql_yan_2005 = f"""
select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_rice_ch4_co2e_yan_2005.parquet')}')
where gwp_time_period = 100
"""
ch4_df_yan_2005 = nibs.load_map_data( ch4_sql_yan_2005)

#%% 
t_yan_2005 = nibs.percentile_map(ch4_df_yan_2005
               , 'total_Gg_co2e_map'
               , "Total rice $CH_4$ emissions\n(Yan et al.)"
               , "$Gg\ CO_2e_{100}$"
               , "map_total_ch4_emissions_Gg_yan_2005.png"
               ,legend_position=(.7,.02)
               )
   
t_yan_2005.show()

ha_yan_2005 = nibs.percentile_map(ch4_df_yan_2005
               , 'kg_ch4_co2e_ha'
               , "Per hectare rice $CH_4$ emissions\n(Yan et al., 2005)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_kg_ha_yan_2005.png"
               ,legend_position=(.7,.02)
               )
   
ha_yan_2005.show()
 
min_ha_yan_2005 = nibs.percentile_map(ch4_df_yan_2005
               , 'min_kg_ch4_co2e_ha'
               , "Minimum per hectare rice $CH_4$ emissions\n(Yan et al., 2005)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_min_kg_ha_yan_2005.png"
               ,legend_position=(.7,.02)
               )
   
min_ha_yan_2005.show()

max_ha_yan_2005 = nibs.percentile_map(ch4_df_yan_2005
               , 'max_kg_ch4_co2e_ha'
               , "Maximum per hectare rice $CH_4$ emissions\n(Yan et al., 2005)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_max_kg_ha_yan_2005.png"
               ,legend_position=(.7,.02)
               )
   
max_ha_yan_2005.show()


# %%
ch4_sql_nikolaisen_2023 = f"""
select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_rice_ch4_co2e_nikolaisen_2023.parquet')}')
where gwp_time_period = 100
"""
ch4_df_nikolaisen_2023 = nibs.load_map_data( ch4_sql_nikolaisen_2023)

#%% 
t_nikolaisen_2023 = nibs.percentile_map(ch4_df_nikolaisen_2023
               , 'total_Gg_co2e_map'
               , "\nTotal rice $CH_4$ emissions\n(Nikolaisen et al., 2023)"
               , "$Gg\ CO_2e_{100}$"
               , "map_total_ch4_emissions_Gg_nikolaisen_2023.png"
               ,legend_position=(.7,.02)
               )
   
t_nikolaisen_2023.show()

ha_nikolaisen_2023 = nibs.percentile_map(ch4_df_nikolaisen_2023
               , 'kg_ch4_co2e_ha'
               , "Per hectare rice $CH_4$ emissions\n(Nikolaisen et al., 2023)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_kg_ha_nikolaisen_2023.png"
               ,legend_position=(.7,.02)
               )
   
ha_nikolaisen_2023.show()

min_ha_nikolaisen_2023 = nibs.percentile_map(ch4_df_nikolaisen_2023
               , 'min_kg_ch4_co2e_ha'
               , "Minimum per hectare rice $CH_4$ emissions\n(Nikolaisen et al., 2023)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_min_kg_ha_nikolaisen_2023.png"
               ,legend_position=(.7,.02)
               )
   
min_ha_nikolaisen_2023.show()

max_ha_nikolaisen_2023 = nibs.percentile_map(ch4_df_nikolaisen_2023
               , 'max_kg_ch4_co2e_ha'
               , "Maximum per hectare rice $CH_4$ emissions\n(Nikolaisen et al., 2023)"
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , "map_ch4_emissions_max_kg_ha_nikolaisen_2023.png"
               ,legend_position=(.7,.02)
               )
   
max_ha_nikolaisen_2023.show()

# %%
gA = ha_bhatia_2013 + labs(title='\n     A')  + theme( plot_title= element_text(ha='left', size=32))
gB = ha_yan_2005 + labs(title='\n     B')  + theme( plot_title= element_text(ha='left', size=32))
gC = ha_nikolaisen_2023 + labs(title='\n     C')  + theme( plot_title= element_text(ha='left', size=32))

gD = t_bhatia_2013 + labs(title='     D')  + theme( plot_title= element_text(ha='left', size=32))
gE = t_yan_2005 + labs(title='     E')  + theme( plot_title= element_text(ha='left', size=32))
gF = t_nikolaisen_2023 + labs(title='     F')  + theme( plot_title= element_text(ha='left', size=32))

g = (gA| gB| gC)/(gD| gE| gF)  + theme(figure_size=(27,18))
g.save(path.join(nibs.chart_dir,"map_ch4_emissions_plate.png"), dpi=300)
g



# %%
model_uncer_sql = f"""select 'Bhatia et al., 2013' as ref, mean_kg_ch4_ha, sd_kg_ch4_ha, mean_min_kg_ch4_ha, mean_max_kg_ch4_ha
from read_parquet('{path.join(nibs.data_dir,'vwG_national_ch4_summary_bhatia_2013.parquet')}')
union
select 'Nikolaisen et al., 2023' as ref, mean_kg_ch4_ha, sd_kg_ch4_ha, mean_min_kg_ch4_ha, mean_max_kg_ch4_ha
from read_parquet('{path.join(nibs.data_dir,'vwG_national_ch4_summary_nikolaisen_2023.parquet')}')
union
select 'Yan et al., 2005' as ref, mean_kg_ch4_ha, sd_kg_ch4_ha, mean_min_kg_ch4_ha, mean_max_kg_ch4_ha
from read_parquet('{path.join(nibs.data_dir,'vwG_national_ch4_summary_yan_2005.parquet')}')"""
model_uncer_df = nibs.load_table_data(model_uncer_sql)
model_uncer_df

#%%
dist_max_farm_size_sql_bhatia = f"""
select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_max_ch4_farm_size_summary_bhatia_2013.parquet')}')
"""
dist_max_farm_size_df_bhatia = nibs.load_map_data( dist_max_farm_size_sql_bhatia)
dist_max_farm_size_df_bhatia = nibs.farm_size_to_catagorical(dist_max_farm_size_df_bhatia, ['max_t_ch4_farm_size','max_kg_ch4_ha_farm_size'])

#%%
g_dist_max_t_farm_size_bhatia = nibs.catagorical_map(dist_max_farm_size_df_bhatia
                    , 'max_t_ch4_farm_size'
                    , '\nRice, landholding size with\nlargest total $CH_4$emissions\n(Bhatia et al., 2013)'
                    , 'Landholding\nSize\n'
                    , 'map_district_farm_size_max_t_bhatia_2013.png'
                    , legend_position=(.7,.02))
g_dist_max_t_farm_size_bhatia

#%%
g_dist_max_ha_farm_size_bhatia = nibs.catagorical_map(dist_max_farm_size_df_bhatia
                    , 'max_kg_ch4_ha_farm_size'
                    , '\nRice, landholding size\nwith largest $CH_4$ emissions perhectare\n(Bhatia et al., 2013)'
                    , 'Landholding\nSize\n'
                    , 'map_district_farm_size_max_ha_bhatia_2013.png'
                    , legend_position=(.7,.02))
g_dist_max_ha_farm_size_bhatia

#%%
dist_max_farm_size_sql_nikolaisen = f"""
select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_max_ch4_farm_size_summary_nikolaisen_2023.parquet')}')
"""
dist_max_farm_size_df_nikolaisen = nibs.load_map_data( dist_max_farm_size_sql_nikolaisen)
dist_max_farm_size_df_nikolaisen = nibs.farm_size_to_catagorical(dist_max_farm_size_df_nikolaisen, ['max_t_ch4_farm_size','max_kg_ch4_ha_farm_size'])

#%%
g_dist_max_t_farm_size_nikolaisen = nibs.catagorical_map(dist_max_farm_size_df_nikolaisen
                    , 'max_t_ch4_farm_size'
                    , '\nRice, landholding size with\nlargest total $CH_4$emissions\n(Nikolaisen et al., 2023)'
                    , 'Landholding\nSize\n'
                    , 'map_district_farm_size_max_t_nikolaisen_2023.png'
                    , legend_position=(.7,.02))
g_dist_max_t_farm_size_nikolaisen

#%%
g_dist_max_ha_farm_size_nikolaisen = nibs.catagorical_map(dist_max_farm_size_df_nikolaisen
                    , 'max_kg_ch4_ha_farm_size'
                    , '\nRice, landholding size\nwith largest $CH_4$ emissions perhectare\n(Nikolaisen et al., 2023)'
                    , 'Landholding\nSize\n'
                    , 'map_district_farm_size_max_ha_nikolaisen_2023.png'
                    , legend_position=(.7,.02))
g_dist_max_ha_farm_size_nikolaisen

#%%
dist_max_farm_size_sql_yan = f"""
select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_max_ch4_farm_size_summary_yan_2005.parquet')}')
"""
dist_max_farm_size_df_yan = nibs.load_map_data( dist_max_farm_size_sql_yan)
dist_max_farm_size_df_yan = nibs.farm_size_to_catagorical(dist_max_farm_size_df_yan, ['max_t_ch4_farm_size','max_kg_ch4_ha_farm_size'])

#%%
g_dist_max_t_farm_size_yan = nibs.catagorical_map(dist_max_farm_size_df_yan
                    , 'max_t_ch4_farm_size'
                    , '\nRice, landholding size with\nlargest total $CH_4$emissions\n(Yan et al., 2005)'
                    , 'Landholding\nSize\n'
                    , 'map_district_farm_size_max_t_yan_2005.png'
                    , legend_position=(.7,.02))
g_dist_max_t_farm_size_yan

#%%
g_dist_max_ha_farm_size_yan = nibs.catagorical_map(dist_max_farm_size_df_yan
                    , 'max_kg_ch4_ha_farm_size'
                    , '\nRice, landholding size\nwith largest $CH_4$ emissions perhectare\n(Yan et al., 2005)'
                    , 'Landholding\nSize\n'
                    , 'map_district_farm_size_max_ha_yan_2005.png'
                    , legend_position=(.7,.02))
g_dist_max_ha_farm_size_yan

# %%
gA = g_dist_max_ha_farm_size_bhatia + labs(title='\n     A')  + theme( plot_title= element_text(ha='left', size=32))
gB = g_dist_max_ha_farm_size_yan + labs(title='\n     B')  + theme( plot_title= element_text(ha='left', size=32))
gC = g_dist_max_ha_farm_size_nikolaisen + labs(title='\n     C')  + theme( plot_title= element_text(ha='left', size=32))

gD = g_dist_max_t_farm_size_bhatia + labs(title='     D')  + theme( plot_title= element_text(ha='left', size=32)) 
gE = g_dist_max_t_farm_size_yan + labs(title='     E')  + theme( plot_title= element_text(ha='left', size=32))
gF = g_dist_max_t_farm_size_nikolaisen + labs(title='     F')  + theme( plot_title= element_text(ha='left', size=32))

g = (gA| gB| gC)/(gD| gE| gF)+ theme(figure_size=(24,16))
g.save(path.join(nibs.chart_dir,"map_district_ch4_farm_size_max_plate.png"), dpi=300)
g


# %%
spearman_sql  = f"""select *  exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_rank_ch4_farm_size_summary_all_models.parquet')}')"""

spearman_df = nibs.load_map_data( spearman_sql)

# %%
ch4_ha_rename_dict = {
    'rank_kg_ch4_ha_farm_size_bhatia_2013': 'Bhatia\net al., 2013',
    'rank_kg_ch4_ha_farm_size_nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'rank_kg_ch4_ha_farm_size_yan_2005': 'Yan et al.,\n2005'
}

ch4_ha_spearman_corr_plot = nibs.avg_spearman_plot(spearman_df
              , 'ch4_farm_size_average_spearman_correlation.png'
              , ch4_ha_rename_dict
              , 'District Average $CH_4$ Comparison\n(Spearman Correlation $Kg\ CO_2e_{100}\ Ha^{-1}$)'
              , 12
              , 'white'
              ,figure_size=(7,6)
              )

ch4_ha_spearman_corr_plot.show()



# %%
cramer_v_sql = f"""select *  exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_max_ch4_farm_size_summary_all_models.parquet')}')"""
cramer_v_df = nibs.load_map_data( cramer_v_sql)
 
 #%%
rename_kg_dict = {
    'max_kg_ch4_ha_farm_size_bhatia_2013': 'Bhatia\net al., 2013',
    'max_kg_ch4_ha_farm_size_nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'max_kg_ch4_ha_farm_size_yan_2005': 'Yan et al.,\n2005'
}

ha_cramers_v = nibs.cramers_v_plot(cramer_v_df
                    , "farm_size_kg_ha_ch4_cramerfarm_size_kg_ha_ch4_cramers_s_v_matrix.png"
                    , rename_kg_dict
                    , 'District farm size with\nhighest methane emission\n($Kg\ Ch_4\ Ha^{-1}$ Cramér\'s V)'
                    , legend_title="Cramér\'s V"
                    ,figure_size=(7,6)
                    )
ha_cramers_v.show()


 #%%
rename_t_dict = {
    'max_t_ch4_farm_size_bhatia_2013': 'Bhatia\net al., 2013',
    'max_t_ch4_farm_size_nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'max_t_ch4_farm_size_yan_2005': 'Yan et al.,\n2005'
}

t_cramers_v = nibs.cramers_v_plot(cramer_v_df
                    , "farm_size_t_ch4_cramers_v_matrix.png"
                    , rename_t_dict
                    , 'District farm size with\nhighest methane emission\n ($Total\ Gg\ Ch_4$ Cramér\'s V)'
                    , legend_title="Cramér\'s V"
                    ,figure_size=(7,6)
                    )
t_cramers_v.show()

# %%
ha_plot = ha_cramers_v + labs(title='\nA')  + theme( plot_title= element_text(ha='left', size=22))
t_plot = t_cramers_v + labs(title='\nB')  + theme( plot_title= element_text(ha='left', size=22))  
cramers_v_plate = ha_plot | t_plot  
cramers_v_plate = cramers_v_plate + theme(figure_size=(14,6))
cramers_v_plate.save(path.join(nibs.chart_dir,"district_ch4_cramers_v_plate.png"), dpi=300)
cramers_v_plate


# %%
ha_corr_sql = f"""
select * from read_parquet('{path.join(nibs.data_dir,'vwA_district_ch4_rice_model_corr_wide.parquet')}')
"""

ha_corr_df = nibs.load_table_data(ha_corr_sql)

# %%
rename_mean_kg_dict = {
    'mean_kg_ch4_ha__bhatia_2013': 'Bhatia\net al., 2013',
    'mean_kg_ch4_ha__nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'mean_kg_ch4_ha__yan_2005': 'Yan et al.,\n2005'
}


ch4_ha_spearman_corr_plot = nibs.spearman_plot(ha_corr_df
              , 'district_rice_kg_ch4_ha_spearman_matrix.png'
              , rename_mean_kg_dict
              , 'District Model Comparison\n(Spearman Correlation $Kg\ CH_4\ Ha^{-1}$)'
              , 12
              , 'white'
              ,figure_size=(7,6)
              )

ch4_ha_spearman_corr_plot.show()

# %%
rename_total_t_dict = {
    'total_t_ch4__bhatia_2013': 'Bhatia\net al., 2013',
    'total_t_ch4__nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'total_t_ch4__yan_2005': 'Yan et al.,\n2005'
}


ch4_t_spearman_corr_plot = nibs.spearman_plot(ha_corr_df
              , 'district_rice_ch4_Gg_spearman_matrix.png'
              , rename_total_t_dict
              , 'District Model Comparison\n(Spearman Correlation\n$Total\ District\ CH_4$)'
              , 12
              , 'white'
              ,figure_size=(7,6)
              )

ch4_t_spearman_corr_plot.show()
# %%
ha_plot = ch4_ha_spearman_corr_plot + labs(title='\nA')  + theme( plot_title= element_text(ha='left', size=22))
t_plot = ch4_t_spearman_corr_plot + labs(title='\nB')  + theme( plot_title= element_text(ha='left', size=22))  
ch4_matrix_plate = ha_plot | t_plot
ch4_matrix_plate = ch4_matrix_plate + theme(figure_size=(14,6))
ch4_matrix_plate.save(path.join(nibs.chart_dir,"district_ch4_spearman_correlation_plate.png"), dpi=300)
ch4_matrix_plate
# %%
