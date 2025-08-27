#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import sys
sys.path.extend(["..\\"])
import math as m
import pandas as pd
from Shared.NIBSData2 import *
from plotnine import ggplot, aes, labs
# import patchworklib as pw
from os import path
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"b:\Repos\NIBS\Data" , 
                chart_dir=r"b:\Repos\NIBS\Graphs")

rice_refs = {'shcherbak_2014': ('Shcherbak et al., 2014','direct_Gg_co2e_map', 'direct_kg_n2o_co2e_ha')
        , 'ipcc_2019': ('IPCC 2019 Updated Methodology','direct_Gg_co2e_map', 'fert_kg_n2o_co2e_ha')
        , 'eagle_2020': ('Eagle et al., 2020','direct_Gg_co2e_map','direct_kg_n2o_co2e_ha')
        , 'bhatia_2013': ('Bhatia et al., 2013','direct_Gg_co2e_map','direct_kg_n2o_co2e_ha')
        , 'hiroko_akiyama_2005': ('Akiyama et al., 2005','direct_Gg_co2e_map','direct_kg_n2o_co2e_ha')
        }

upland_refs = {'shcherbak_2014': ('Shcherbak et al., 2014','direct_Gg_co2e_map', 'direct_kg_n2o_co2e_ha')
        , 'ipcc_2019': ('IPCC 2019 Updated Methodology','direct_Gg_co2e_map', 'fert_kg_n2o_co2e_ha')
        , 'eagle_2020': ('Eagle et al., 2020','direct_Gg_co2e_map','direct_kg_n2o_co2e_ha')
        , 'bhatia_2013': ('Bhatia et al., 2013','direct_Gg_co2e_map','fert_kg_n2o_co2e_ha')
        }

# %%
upland_dfs = []
for ref in upland_refs.keys():
  sql = f"""
  select * exclude geog, cast(geog as string) as geog
  from read_parquet('{path.join(nibs.data_dir,f'vwM_district_n2o_co2e_upland_crop_results_{ref}.parquet')}') 
  where gwp_time_period = 100"""
  df = nibs.load_map_data(sql)
  upland_dfs.append((ref, df))

#%%
upland_graphs = {}
for ref, df in upland_dfs:
  file_name = f"map_non_rice_total_n2o_emissions_Gg_{ref}.svg"
  variable = upland_refs[ref][1]
  title = f'Total non-rice $N_2O$\nfertilizer induced emissions\n({upland_refs[ref][0]})'
  g_t =  nibs.percentile_map(df
               , variable
               , title
               , "$Gg\ CO_2e_{100}$"
               , file_name
               , legend_position = (.75,.1))
   
  file_name = f"map_non_rice_ch4_emissions_kg_ha_{ref}.svg"
  variable = upland_refs[ref][2]
  title = f"Per hectare non-rice $N_2O$\nfertilizer induced emissions\n({upland_refs[ref][0]})"
  g_ha =  nibs.percentile_map(df
               , variable
               , title
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , file_name
               , legend_position = (.8,.1))

  upland_graphs[ref] = (g_t, g_ha)
  g_t.show()
  g_ha.show()


# %%

gA = (upland_graphs['bhatia_2013'][1] + labs(title='\n     A')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gB = (upland_graphs['eagle_2020'][1] + labs(title='\n     B')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gC = (upland_graphs['ipcc_2019'][1] + labs(title='\n     C')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gD = (upland_graphs['shcherbak_2014'][1] + labs(title='\n     D') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gE = (upland_graphs['bhatia_2013'][0] + labs(title='     E')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gF = (upland_graphs['eagle_2020'][0] + labs(title='     F')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gG = (upland_graphs['ipcc_2019'][0] + labs(title='     G') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gH = (upland_graphs['shcherbak_2014'][0] + labs(title='     H') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
g = (gA| gB| gC| gD)/(gE| gF| gG| gH) + theme(figure_size=(32, 16))
g.save(path.join(nibs.chart_dir,"map_district_n2o_upland_crop_ha_Gg_plate.svg"), dpi=nibs.dpi)
g



# %%
rice_dfs = []
for ref in rice_refs.keys():
  sql = f"""
  select * exclude geog, cast(geog as string) as geog
  from read_parquet('{path.join(nibs.data_dir,f'vwM_district_n2o_co2e_rice_results_{ref}.parquet')}') 
  where gwp_time_period = 100"""
  df = nibs.load_map_data(sql)
  rice_dfs.append((ref, df))

#%%
rice_graphs = {}
for ref, df in rice_dfs:
  file_name = f"map_rice_total_n2o_emissions_Gg_{ref}.svg"
  variable = rice_refs[ref][1]
  title = f'Total rice $N_2O$\nfertilizer induced emissions\n({rice_refs[ref][0]})'
  g_t =  nibs.percentile_map(df
               , variable
               , title
               , "$Gg\ CO_2e_{100}$"
               , file_name
               , legend_position = (.75,.1))
   
  file_name = f"map_rice_ch4_emissions_kg_ha_{ref}.svg"
  variable = rice_refs[ref][2]
  title = f"Per hectare rice $N_2O$\nfertilizer induced emissions\n({rice_refs[ref][0]})"
  g_ha =  nibs.percentile_map(df
               , variable
               , title
               , "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
               , file_name
               , legend_position = (.8,.1))

  rice_graphs[ref] = (g_t, g_ha)
  g_t.show()
  g_ha.show()


# %%
gA = (rice_graphs['bhatia_2013'][0] + labs(title='\n     A') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gB = (rice_graphs['eagle_2020'][0] + labs(title='\n     B') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gC = (rice_graphs['ipcc_2019'][0] + labs(title='\n     C')
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gD = (rice_graphs['shcherbak_2014'][0] + labs(title='\n     D')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gE = (rice_graphs['hiroko_akiyama_2005'][0] + labs(title='\n     E') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gF = (rice_graphs['bhatia_2013'][1]+ labs(title='     F')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gG = (rice_graphs['eagle_2020'][1] + labs(title='     G') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gH = (rice_graphs['ipcc_2019'][1] + labs(title='     H') 
                + theme( plot_title= element_text(ha='left', size=32))
                        ) 
gI = (rice_graphs['shcherbak_2014'][1] + labs(title='     I') 
                + theme( plot_title= element_text(ha='left', size=32))
                        )
gJ = (rice_graphs['hiroko_akiyama_2005'][1] + labs(title='     J')
                + theme( plot_title= element_text(ha='left', size=32))
                        )

g = (gA| gB| gC| gD| gE)/(gF| gG| gH| gI| gJ) + theme(figure_size=(40, 16))
g.save(path.join(nibs.chart_dir,"map_district_n2o_rice_ha_Gg_plate.svg"), dpi=nibs.dpi)
g




# %%
upland_corr_sql = f"""
select * from read_parquet('{path.join(nibs.data_dir,'vwA_district_n2o_co2e_upland_crop_model_corr_wide.parquet')}') 
"""

upland_corr_df = nibs.load_table_data(upland_corr_sql)

# %%

upland_ha_rename_dict = {'kg_n2o_co2e_ha__shcherbak_2014': 'Shcherbak\net al., 2014\n'
        , 'kg_n2o_co2e_ha__ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n'
        , 'kg_n2o_co2e_ha__eagle_2020': 'Eagle et\nal., 2020\n'
        , 'kg_n2o_co2e_ha__bhatia_2013': 'Bhatia et\nal., 2013\n'
        }

upland_ha_spearman_corr_plot = nibs.spearman_plot(upland_corr_df
              , 'district_upland_crop_kg_n2o_co2e_ha_spearman_matrix.svg'
              , upland_ha_rename_dict
              , 'None-rice District Model Comparison\n(Spearman Correlation $Kg\ N_2O\ Ha^{-1}$)'
              , 12
              , 'white'
              )

upland_ha_spearman_corr_plot.show()

upland_t_rename_dict = {'total_Gg_co2e_map__shcherbak_2014': 'Shcherbak\net al., 2014\n'
        , 'total_Gg_co2e_map__ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n'
        , 'total_Gg_co2e_map__eagle_2020': 'Eagle et\nal., 2020\n'
        , 'total_Gg_co2e_map__bhatia_2013': 'Bhatia et\nal., 2013\n'
        }

upland_t_spearman_corr_plot = nibs.spearman_plot(upland_corr_df
              , 'district_upland_crop_n2o_Gg_co2e_spearman_matrix.svg'
              , upland_t_rename_dict
              , 'None-rice District Model Comparison\n(Spearman Correlation $Gg\ N_2O$)'
              , 12
              , 'white'
              )
upland_t_spearman_corr_plot.show()


# %%
rice_corr_sql = f"""
select * from read_parquet('{path.join(nibs.data_dir,'vwA_district_n2o_co2e_rice_model_corr_wide.parquet')}') 
"""
rice_corr_df = nibs.load_table_data( rice_corr_sql)

#%%

rice_ha_rename_dict = {'kg_n2o_co2e_ha__shcherbak_2014': 'Shcherbak\net al., 2014\n'
        , 'kg_n2o_co2e_ha__ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n'
        , 'kg_n2o_co2e_ha__eagle_2020': 'Eagle et\nal., 2020\n'
        , 'kg_n2o_co2e_ha__bhatia_2013': 'Bhatia et\nal., 2013\n'
        , 'kg_n2o_co2e_ha__hiroko_akiyama_2005': 'Akiyama et\nal., 2005\n'
        }


rice_ha_spearman_corr_plot = nibs.spearman_plot(rice_corr_df
              , 'district_rice_crop_kg_n2o_co2e_ha_spearman_matrix.svg'
              , rice_ha_rename_dict
              , 'Rice District Model Comparison\n(Spearman Correlation $Kg\ N_2O\ Ha^{-1}$)'
              , 12
              , 'white'
              ,figure_size=(7,6)
              )

rice_ha_spearman_corr_plot.show()


rice_t_rename_dict = {'total_Gg_co2e_map__shcherbak_2014': 'Shcherbak\net al., 2014\n'
        , 'total_Gg_co2e_map__ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n'
        , 'total_Gg_co2e_map__eagle_2020': 'Eagle et\nal., 2020\n'
        , 'total_Gg_co2e_map__bhatia_2013': 'Bhatia et\nal., 2013\n'
        , 'total_Gg_co2e_map__hiroko_akiyama_2005': 'Akiyama et\nal., 2005\n'
        }

rice_t_spearman_corr_plot = nibs.spearman_plot(rice_corr_df
              , 'district_rice_crop_n2o_Gg_co2e_spearman_matrix.svg'
              , rice_t_rename_dict
              , 'Rice District Model Comparison\n(Spearman Correlation $Gg\ N_2O$)'
              , 12
              , 'white'
              ,figure_size=(7,6)
              )
rice_t_spearman_corr_plot.show()


# %%
national_summary_sql = f"""select *
from read_parquet('{path.join(nibs.data_dir,'vwG_National_co2e_summmary_all_gases_all_models_long.parquet')}') 
where gwp_time_period = 100 
"""
national_summary_df = nibs.load_table_data( national_summary_sql)

#%%
def reverse_text(s):
    return s[::-1]

gases_dic = {"$Residue\ Burning\ CO_2e_{100}$": "Residue Burning\n$CO_2e_{100}$",
             "$N_2O\ CO_2e_{100}$, (rice)": "$N_2O\ CO_2e_{100}$\n(rice)",
             "$N_2O\ CO_2e_{100}$, (upland crops)": "$N_2O\ CO_2e_{100}$\n(none-rice crops)",
             }
national_summary_df['gas'] = national_summary_df['gas'].replace(gases_dic)
gases =  national_summary_df.sort_values(['gas'])['gas'].unique()
national_summary_df['reverse_model'] = national_summary_df['model_ref'].apply(reverse_text) 
models = national_summary_df.sort_values(['gas','model_name'])['model_ref'].unique()
models_dic = {x: x.replace(" et al.,"," et al.,\n")
              .replace("(none-rice crops)","\n(none-rice crops)")
              .replace("(28 crops)","\n(28 crops)")
              .replace("(44 crops)","\n(44 crops)")
              .replace("IPCC 2019 Updated Methodology",'IPCC 2019 Updated\nMethodology') for x in models} 
national_summary_df['model_ref'] = national_summary_df['model_ref'].replace(models_dic)

national_summary_df['gas'] = pd.Categorical(national_summary_df['gas'], categories=gases, ordered=True)
national_summary_df['model_ref'] = pd.Categorical(national_summary_df['model_ref'], categories=list(models_dic.values()), ordered=False)

model_name_dic = {v[0]:v[1] for v in national_summary_df[['model_ref','model_name']].drop_duplicates().values}

def x_lab(s):
  return [model_name_dic[x] for x in s]

#%%
national_facet_g = (ggplot(national_summary_df)
      + geom_bar(aes(x='model_ref', y='value', fill='gas'), stat='identity', width=.8)
      + geom_errorbar(aes(x='model_ref', ymin='value - sd', ymax='value + sd'), width=.5)
      + labs(title='Total Cropping Emission by Model', x="Models" ,y="Tg $CO_2e_{100}$                       Kg $CO_2e_{100}\ Ha^{-1}$")
      #+ scale_y_continuous(limits=(0,250))
      + scale_x_discrete(labels=x_lab)
      + scale_fill_discrete(name='Gas')
      + theme_minimal()
      + theme(figure_size= (10,8), 
          title=element_text(size=18, backgroundcolor='white'), 
          #rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
          axis_text_x=element_text(rotation=90, size=13, hjust='center'),
          axis_text_y=element_text(rotation=90, size=14, vjust=.5, hjust=1),
          axis_title_x=element_text(size=16),
          axis_title_y=element_text(size=16),
          strip_background=element_blank(),
          strip_align_y=0.9,
          strip_text=element_blank(),
          legend_text=element_text(size=14)
      )
      + facet_grid('units~.', scales='free_y')
   
)
national_facet_g.save(path.join(nibs.chart_dir, "national_all_models_all_emissions_facet.svg"), dpi=nibs.dpi)
national_facet_g


# %%
area_national_summary_df = national_summary_df[national_summary_df['statistic'] == 'mean']

nat_g = (ggplot(area_national_summary_df)
      + geom_bar(aes(x='model_ref', y='value', fill='gas'), stat='identity', width=.5)
      + geom_errorbar(aes(x='model_ref', ymin='value - sd', ymax='value + sd'), width=.3)
      + labs(title='Total Cropping Emission by Model', x="Models", y="Kg $CO_2e_{100}\ Ha^{-1}$")
      + scale_y_continuous()
      + scale_x_discrete()
      + scale_fill_discrete(name='Gas')
      + theme_minimal()
      + theme(figure_size= (10,4), 
            title=element_text(size=22, backgroundcolor='white'), 
        #    rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
          axis_text_x=element_text(rotation=90, size=8, hjust='center'),
          axis_text_y=element_text(rotation=0, size=8, vjust='center', hjust='center'),
      )
   
)

mean_g.save(path.join(nibs.chart_dir, "area_national_all_models_all_emissions.svg"), dpi=nibs.dpi)
mean_g


#%%
national_summary_df

# %%
national_summary_sql = f"""select *
from read_parquet('{path.join(nibs.data_dir,'vwG_National_co2e_summmary_all_gases_all_models.parquet')}') 
where gwp_time_period = 100
	and model_ref in('Nikolaisen et al., 2023','Eagle et al., 2020 (rice)','Eagle et al., 2020 (none-rice crops)','Biomass Altas (44 crops)')
"""

national_summary_df = nibs.load_table_data( national_summary_sql)

#%%
def reverse_text(s):
    return s[::-1]
gases =  national_summary_df['gas'].unique()
national_summary_df['reverse_model'] = national_summary_df['model_ref'].apply(reverse_text) 
models = national_summary_df.sort_values(['gas','reverse_model'])['model_ref'].unique()
# models_dic = {x: x.replace(" et al,.","\net al.,").replace(" et al.,","\net al.,").replace("IPCC 2019 Updated Methodology",'IPCC 2019 Updated\nMethodology') for x in models} 
models_dic = {'Nikolaisen et al., 2023': 'Nikolaisen\net al., 2023',
 'Eagle et al., 2020 (rice)': 'Eagle et al.,\n2020 (rice)',
 'Eagle et al., 2020 (none-rice crops)': 'Eagle\net al., 2020\n(none-rice crops)',
 'Biomass Altas (44 crops)': 'Biomass Altas\n(44 crops)'}
national_summary_df['model_ref'] = national_summary_df['model_ref'].replace(models_dic)

gases_dic = {x: x.replace('\ CO_2e_{100}','').replace("$Residue\ Burning$","Residue\nBurning") for x in gases}
national_summary_df['gas'] = national_summary_df['gas'].replace(gases_dic)
national_summary_df

#%%
national_summary_df['gas'] = pd.Categorical(national_summary_df['gas'], categories=list(gases_dic.values()), ordered=True)
national_summary_df['model_ref'] = pd.Categorical(national_summary_df['model_ref'], categories=list(models_dic.values()), ordered=True)

#%%
ch4_g = (ggplot(national_summary_df)
      + geom_bar(aes(x='model_ref', y='total_Tg_co2e', fill='gas'), stat='identity', width=.8)
      + geom_errorbar(aes(x='model_ref', ymin='total_Tg_co2e - sd_total_Tg_co2e', ymax='total_Tg_co2e + sd_total_Tg_co2e'), width=.5)
      + labs(title='Total Cropping Emissions by Model', x="Models", y="Tg $CO_2e_{100}$")
      + scale_y_continuous(limits=(0,150))
      + scale_x_discrete()
      + scale_fill_discrete( name='Gas')
      + theme_minimal()
      + theme(figure_size= (8,8), 
            title=element_text(size=22, backgroundcolor='white'), 
            rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
            legend_text=element_text(size=18),
          axis_text_x=element_text(rotation=90, size=18, hjust='center'),
          axis_text_y=element_text(rotation=0, size=18, vjust='center', hjust='center'),
      )
   
)
ch4_g.save(path.join(nibs.chart_dir, "national_edf_models_all_emissions.svg"), dpi=nibs.dpi)
ch4_g
# %%
