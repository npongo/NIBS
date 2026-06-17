#%%
# %reload_ext autoreload
# %autoreload 2


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.extend(["..\\"])
import math as m
import pandas as pd
from Shared.NIBSData2 import *
from plotnine import ggplot, aes, labs
from os import path
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"b:\Repos\NIBS\Data" , 
                chart_dir=r"b:\Repos\NIBS\Graphs")

## Biomass Altas 2.0v 
# %%
residue_burning_sql = f"""select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_residue_burning_co2e.parquet')}')
where gwp_time_period = 100
"""
residue_burning_df = nibs.load_map_data(residue_burning_sql)

#%% 
biomass_Gg_g = nibs.percentile_map(residue_burning_df
               , 'total_Gg_co2e_map'
               , "Total residue burning\nemissions (Biomass Atlas 2.0v)"
               , "$Gg\ CO_2e_{100}$\n"
               , "map_total_residue_burning_emissions_Gg.png"
               ,legend_position=(.7,.02)
               )
   
biomass_Gg_g.show()


#%% 
biomass_ha_g = nibs.percentile_map(residue_burning_df
               , 'kg_co2e_ha'
               , "Per hectare residue burning\nemissions (Biomass Atlas 2.0v)"
               , "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
               , "map_residue_burning_emissions_kg_ha_biomass_altas_v2.png"
               ,legend_position=(.7,.02)
               )
   
biomass_ha_g.show()


## Karan 2021 
# %%
residue_burning_karan_sql = f"""select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_residue_burning_karan_2021_co2e.parquet')}')
where gwp_time_period = 100
"""
residue_burning_karan_df = nibs.load_map_data(residue_burning_karan_sql)

#%% 
karan_Gg_g = nibs.percentile_map(residue_burning_karan_df
               , 'total_Gg_co2e_map'
               , "Total residue burning\nemissions (Karan et al. 2021)"
               , "$Gg\ CO_2e_{100}$\n"
               , "map_total_residue_burning_karan_emissions_Gg.png"
               ,legend_position=(.7,.02)
               )
   
karan_Gg_g.show()

#%% 
karan_ha_g = nibs.percentile_map(residue_burning_karan_df
               , 'kg_co2e_ha'
               , "Per hectare residue burning\nemissions(Karan et al. 2021)"
               , "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
               , "map_residue_burning_karan_emissions_kg_ha.png"
               ,legend_position=(.7,.02)
               )
   
karan_ha_g.show()

# %%
gA = biomass_ha_g + labs(title='\n     A')  + theme( plot_title= element_text(ha='left', size=32))
gB = karan_ha_g + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32))

gC = biomass_Gg_g + labs(title='     C')  + theme( plot_title= element_text(ha='left', size=32))
gD = karan_Gg_g + labs(title='     D')  + theme( plot_title= element_text(ha='left', size=32))

g = (gA| gB)/(gC| gD) + theme(figure_size=(18,18))
g.save(path.join(nibs.chart_dir,"map_district_residue_burning_plate.png"), dpi=300)
g


# %%
ha_corr_sql = f"""
select * from read_parquet('{path.join(nibs.data_dir,'vwA_district_residue_burning_model_corr_wide.parquet')}')
where gwp_time_period = 100
"""

ha_corr_df = nibs.load_table_data(ha_corr_sql)

# %%
burning_ha_rename_dict = {'kg_co2e_ha__biomass_altas_v2_44': 'Biomass Altas v2.0\n(All Crops)\n'
        , 'kg_co2e_ha__karan_2021': 'Karan et al.,\n2021(28 Crops)\n'
        , 'kg_co2e_ha__biomass_altas_v2_28': 'Biomass Altas v2.0\n(28 Crops)\n'
        }

burning_ha_spearman_corr_plot = nibs.spearman_plot(ha_corr_df
              , 'district_rice_crop_kg_n2o_co2e_ha_spearman_matrix.png'
              , burning_ha_rename_dict
              , 'District Residue Burning Model Comparison\n(Spearman Correlation $Kg\ CO_2e_{100}\ Ha^{-1}$)'
              , 12
              , 'white'
              ,figure_size=(7,6)
              )

burning_ha_spearman_corr_plot.show()

# %%
burning_Gg_rename_dict = {'total_Gg_co2e__biomass_altas_v2_44': 'Biomass Altas v2.0\n(All Crops)\n'
        , 'total_Gg_co2e__karan_2021': 'Karan et al.,\n2021(28 Crops)\n'
        , 'total_Gg_co2e__biomass_altas_v2_28': 'Biomass Altas v2.0\n(28 Crops)\n'
        }

burning_Gg_spearman_corr_plot = nibs.spearman_plot(ha_corr_df
              , 'district_residue_burning_co2e_Gg_spearman_heatmap.png'
              , burning_Gg_rename_dict
              , 'District Residue Burning Model Comparison\n(Spearman Correlation $Total\ District\ CO_2e_{100}$)'
              , 12
              , 'white'
              , figure_size=(7,6)
              )

burning_Gg_spearman_corr_plot.show()

# %%
plot_a = burning_ha_spearman_corr_plot + labs(title='\nA')  + theme( plot_title= element_text(ha='left', size=22))
plot_b = burning_Gg_spearman_corr_plot + labs(title='\nB')  + theme( plot_title= element_text(ha='left', size=22))  
burning_matrix_plate = plot_a | plot_b                  
burning_matrix_plate = burning_matrix_plate + theme(figure_size=(14,6))
burning_matrix_plate.save(path.join(nibs.chart_dir,"district_residue_burning_spearman_correlation_plate.png"), dpi=300) 
burning_matrix_plate

# %%
