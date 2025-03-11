#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import os
sys.path.extend([r"E:\npongo Dropbox\benjamin clark\PythonProjects"])
import numpy as np
import math as m
import pandas as pd
from scipy import stats
import itertools
import geopandas as gpd 
import PostDoc.db_clients.mssql_db_client as mssql  
import PostDoc.Plotting.squarify as squarify
import copy
import docx 
from docx.shared import Cm 
from PostDoc.Plotting.PlottingFunctions import *
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, scale_color_discrete, labs
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from os import path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns


chart_dir = r"E:\npongo Dropbox\benjamin clark\CIL\Products\Paper1\Enviromental Letters"
if not os.path.exists(chart_dir):
       os.makedirs(chart_dir)

print('Python %s on %s' % (sys.version, sys.platform))
#load india national boundary as map background
india_sql = "SELECT geog.STAsBinary() as geog FROM [dbo].[national_boundaries]"
india = load_map_data(db_client, india_sql)
base_map = plot_map(None, india, None, 'base map', '')
base_map

state_sql = "select * from vwM_india_states option(maxrecursion 0)"
india_states = load_map_data(db_client, state_sql)


# Open an image from a computer 
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output


# db_conn = {'server': '.\\npongo22', 'database': 'india_cost_of_cultivation_ghg_results_v1'}
# db_client = mssql.SqlServerClient(db_conn) 

db_conn_input = {'server': '.\\npongo22', 'database': 'india_agriculture_census_ghg_results_v2'}
db_client_input = mssql.SqlServerClient(db_conn_input)

def format_name(c):

    label = (c
             .replace('_',' ')
             .replace('area wt avg kg co2e ha',"Area Wt Average Kg CO$_2$e Ha$^{-1}$")
             .replace('total emissions Gg co2e',"Total Emissions Gg CO$_2$e")
             .replace('total emissions Tg co2e',"Total Emissions Tg CO$_2$e")
             .replace('no','NO')
             .replace('n2o','N$_2$O')
             .replace('ch4','CH$_4$')
             .replace('caco3','CaCO$_3$')
             .replace('nh3','NH$_3$')
             .replace('crop','organic\nfertilizer')
             .replace('kg n production kg','N Kg Crop Kg$^{-1}$')
             .replace('kg co2e production kg','CO$_2$ Kg Crop Kg$^{-1}$')
             .replace('kg co2e ha','CO$_2$ Kg Ha$^{-1}$')
             .replace('kg n ha','N Kg Ha$^{-1}$')
             .replace('kg co2e','CO$_2$ Kg')
             .replace('kg n','N Kg')
             .replace('n kg','N Kg')
             .replace('avg wt avg','')
             .replace('co2e','CO$_2$e')
             .replace('Co2e','CO$_2$e')
             .replace('manure mgmt','\nmanure mgmt')
             .replace('enteric fermentation','enteric\nfermentation')
             )
    label = label[0].upper() + label[1:]
    return label

colors = ['green', 'blue', 'red', 'orange', 'purple', 'violet', 'aqua']
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00',  '#cab2d6','#6a3d9a','#ffff99','#b15928']


#%%
low_color = "green"
mid_color= "yellow"
high_color = "red"
colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
#colors = ['#81C000',  '#F2F900',  '#FFE700',  '#FF8F00',  '#FF0500']

## Biomass Altas 2.0v 
# %%
residue_burning_sql = """select *
from vwM_district_residue_burning_co2e
where gwp_time_period = 100
"""
residue_burning_df = load_map_data(db_client_input, residue_burning_sql)
residue_burning_df.head()

#%% 
units = "$Gg\ CO_2e_{100}$\n"
residue_burning_df['total_Gg_co2e_map_p_score'] = residue_burning_df['total_Gg_co2e_map'].rank(pct=True)
breaks =  residue_burning_df['total_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
biomass_Gg_g = (ggplot(residue_burning_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="total_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total residue burning\nemissions (Biomass Atlas 2.0v)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=35, high=high_color, name=units, limits=[0,130])
   
     + theme(
        figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
biomass_Gg_g.save(filename="map_total_residue_burning_emissions_Gg.png", path=chart_dir,  units='cm', dpi=300)
biomass_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
residue_burning_df['kg_co2e_ha_p_score'] = residue_burning_df['kg_co2e_ha'].rank(pct=True)
breaks =  residue_burning_df['kg_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
biomass_ha_g = (ggplot(residue_burning_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare residue burning\nemissions (Biomass Atlas 2.0v)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=175, high=high_color, name=units, limits=[0,500])
   
     + theme(
         figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_position=(.7,.02)
     )
)
biomass_ha_g.save(filename="map_residue_burning_emissions_kg_ha_biomass_altas_v2.png", path=chart_dir,  units='cm', dpi=300)
biomass_ha_g


## Karan 2021 
# %%
residue_burning_karan_sql = """select *
from [dbo].[vwM_district_residue_burning_karan_2021_co2e]
where gwp_time_period = 100
"""
residue_burning_karan_df = load_map_data(db_client_input, residue_burning_karan_sql)
residue_burning_karan_df.head()

#%% 
units = "$Gg\ CO_2e_{100}$\n"
residue_burning_karan_df['total_Gg_co2e_map_p_score'] = residue_burning_karan_df['total_Gg_co2e_map'].rank(pct=True)
breaks =  residue_burning_karan_df['total_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
karan_Gg_g = (ggplot(residue_burning_karan_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="total_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total residue burning\nemissions (Karan et al. 2021)')
     
    + scale_fill_gradientn(colors=colors, values= [0, 0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=35, high=high_color, name=units, limits=[0,70])
   
     + theme(
        figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
karan_Gg_g.save(filename="map_total_residue_burning_karan_emissions_Gg.png", path=chart_dir,  units='cm', dpi=300)
karan_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
residue_burning_karan_df['kg_co2e_ha_p_score'] = residue_burning_karan_df['kg_co2e_ha'].rank(pct=True)
breaks =  residue_burning_karan_df['kg_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
karan_ha_g = (ggplot(residue_burning_karan_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare residue burning\nemissions(Karan et al. 2021)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=60, high=high_color, name=units, limits=[0,150])
   
     + theme(
         figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
karan_ha_g.save(filename="map_residue_burning_karan_emissions_kg_ha.png", path=chart_dir,  units='cm', dpi=300)
karan_ha_g

# %%
gA = pw.load_ggplot(biomass_ha_g + labs(title='A')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gB = pw.load_ggplot(karan_ha_g + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))

gC = pw.load_ggplot(biomass_Gg_g + labs(title='C')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gD = pw.load_ggplot(karan_Gg_g + labs(title='D')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))

#%%
g = (gA| gB)/(gC| gD)
g.savefig(path.join(chart_dir,"map_district_residue_burning_plate.png"), dpi=300)
g










# %%
ha_corr_sql = """
select * from [vwA_district_residue_burning_model_corr_wide]
where gwp_time_period = 100
"""

ha_corr_df = load_table_data(db_client_input, ha_corr_sql)
ha_corr_df.head()

# %%
corr_kg_co2e_ha_df = ha_corr_df.loc[:, ['kg_co2e_ha__biomass_altas_v2_44','kg_co2e_ha__karan_2021','kg_co2e_ha__biomass_altas_v2_28']]
corr_kg_co2e_ha_df.head()     
                
         
# %%
def cap_first_letter(s):
    if 'biomass' in s: 
        if s.endswith('28'): 
            return 'Biomass Altas v2.0\n(28 Crops)'
        if s.endswith('44'): 
            return 'Biomass Altas v2.0\n(All Crops)' 
    return 'Karan et al.,\n2021(28 Crops)'

corr_kg_co2e_ha_matrix = corr_kg_co2e_ha_df.corr(method='spearman')
corr_kg_co2e_ha_matrix.columns = [cap_first_letter(x.replace('kg_co2e_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.columns]
corr_kg_co2e_ha_matrix.index = [cap_first_letter(x.replace('kg_co2e_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.index]
corr_kg_co2e_ha_matrix

corr_kg_co2e_long = corr_kg_co2e_ha_matrix.stack().reset_index()
corr_kg_co2e_long.columns = ['model1', 'model2', 'Spearman']

# Rename categories for better readability

# corr_Gg_co2e_long['model1'] = corr_Gg_co2e_long['model1'].replace(rename_dict)
# corr_Gg_co2e_long['model2'] = corr_Gg_co2e_long['model2'].replace(rename_dict)

# Convert to categorical with the new names
corr_kg_co2e_long['model1'] = pd.Categorical(values=corr_kg_co2e_long['model1'], categories=corr_kg_co2e_long['model1'].unique(), ordered=True)
corr_kg_co2e_long['model2'] = pd.Categorical(values=corr_kg_co2e_long['model2'], categories=corr_kg_co2e_long['model2'].unique(), ordered=True)

# Reverse the order of the categories
model1_categories = corr_kg_co2e_long['model1'].unique()[::-1]
model2_categories = corr_kg_co2e_long['model2'].unique()

# Change the data type of the 'Spearman' column to float64
corr_kg_co2e_long['Spearman'] = corr_kg_co2e_long['Spearman'].astype('float64')

corr_kg_co2e_long.head()

#%%
heatmap_plot = (
    ggplot(corr_kg_co2e_long, aes(x='model1', y='model2', fill='Spearman')) +
    geom_tile() +
    geom_text(aes(label='round(Spearman, 2)'), size=12, color='black') +
    scale_fill_gradient(low='blue', high='red', name="Spearman Corr") +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District Residue Burning Model Comparison\n(Spearman Correlation $Kg\ CO_2e_{100}\ Ha^{-1}$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_residue_burning_kg_co2e_ha_spearman_heatmap.png"), dpi=300)


# %%
corr_Gg_co2e_df = ha_corr_df.loc[:, ['total_Gg_co2e__biomass_altas_v2_44','total_Gg_co2e__karan_2021','total_Gg_co2e__biomass_altas_v2_28']]
corr_Gg_co2e_df.head() 
corr_Gg_co2e_matrix = corr_Gg_co2e_df.corr(method='spearman')
corr_Gg_co2e_matrix.columns = [cap_first_letter(x.replace('total_Gg_co2e__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.columns]
corr_Gg_co2e_matrix.index = [cap_first_letter(x.replace('total_Gg_co2e__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.index]
corr_Gg_co2e_matrix

#%%
corr_Gg_co2e_long = corr_Gg_co2e_matrix.stack().reset_index()
corr_Gg_co2e_long.columns = ['model1', 'model2', 'Spearman']

# Rename categories for better readability

# corr_Gg_co2e_long['model1'] = corr_Gg_co2e_long['model1'].replace(rename_dict)
# corr_Gg_co2e_long['model2'] = corr_Gg_co2e_long['model2'].replace(rename_dict)

# Convert to categorical with the new names
corr_Gg_co2e_long['model1'] = pd.Categorical(values=corr_Gg_co2e_long['model1'], categories=corr_Gg_co2e_long['model1'].unique(), ordered=True)
corr_Gg_co2e_long['model2'] = pd.Categorical(values=corr_Gg_co2e_long['model2'], categories=corr_Gg_co2e_long['model2'].unique(), ordered=True)

# Reverse the order of the categories
model1_categories = corr_Gg_co2e_long['model1'].unique()[::-1]
model2_categories = corr_Gg_co2e_long['model2'].unique()

# Change the data type of the 'Spearman' column to float64
corr_Gg_co2e_long['Spearman'] = corr_Gg_co2e_long['Spearman'].astype('float64')

corr_Gg_co2e_long.head()

#%%

heatmap_plot = (
    ggplot(corr_Gg_co2e_long, aes(x='model1', y='model2', fill='Spearman')) +
    geom_tile() +
    geom_text(aes(label='round(Spearman, 2)'), size=12, color='black') +
    scale_fill_gradient(low='blue', high='red', name="Spearman Corr") +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District Residue Burning Model Comparison\n(Spearman Correlation $Total\ District\ CO_2e_{100}$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_residue_burning_co2e_Gg_spearman_heatmap.png"), dpi=300)



# %%
