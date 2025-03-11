
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
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, scale_color_discrete, labs,scale_fill_manual
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from os import path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



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


def percentile_score(data):
    return [stats.percentileofscore(data, n) for n in data]

#percentile colors

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00',  '#cab2d6','#6a3d9a','#ffff99','#b15928']
colors = ['green', 'blue', 'red', 'orange', 'purple', 'violet', 'aqua']
colors = ['#81C000',  '#F2F900',  '#FFE700',  '#FF8F00',  '#FF0500']

#color ramp
low_color = "green"
mid_color= "yellow"
high_color = "red"
 
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

#%%
apy_crop_replacements = {
    'Cotton(lint)': 'Cotton',  # '<1.0 Ha',
}
crop_colors = {
    'Rice': 'green',  
    'Wheat': 'yellowgreen',  
    'Maize':  'yellow',  
    'Cotton': 'orange',  
    'Dry chillies':'#FF8F60' , 
    'Coconut':'#cc5500', 
    'Soyabean': '#ff6500',
    'Jowar': 'Red',  
    'Potato': '#cb4154',  
    'Sugarcane': 'darkred',  
    'Other Crops': 'Brown',  
}


#%%
dist_max_t_apy_crop_sql_eagle = """
select *
from vwM_district_max_n2o_6_class_apy_crop_summary_eagle_2020
"""
dist_max_t_apy_crop_df_eagle = load_map_data(db_client_input, dist_max_t_apy_crop_sql_eagle)

#%%

dist_max_t_apy_crop_df_eagle['max_fert_induced_Tg_n2o_n_apy_crop'] = dist_max_t_apy_crop_df_eagle['max_fert_induced_Tg_n2o_n_apy_crop'].replace(apy_crop_replacements)
dist_max_t_apy_crop_df_eagle['max_fert_induced_kg_n2o_n_ha_apy_crop'] = dist_max_t_apy_crop_df_eagle['max_fert_induced_kg_n2o_n_ha_apy_crop'].replace(apy_crop_replacements)

#%%
# Count the occurrences of each category
counts = dist_max_t_apy_crop_df_eagle[dist_max_t_apy_crop_df_eagle['max_fert_induced_Tg_n2o_n_apy_crop'] != 'Other Crops']['max_fert_induced_Tg_n2o_n_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_eagle['max_fert_induced_Tg_n2o_n_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_eagle['max_fert_induced_Tg_n2o_n_apy_crop'], categories=ordered_categories, ordered=True)

counts = dist_max_t_apy_crop_df_eagle[dist_max_t_apy_crop_df_eagle['max_fert_induced_kg_n2o_n_ha_apy_crop'] != 'Other Crops']['max_fert_induced_kg_n2o_n_ha_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_eagle['max_fert_induced_kg_n2o_n_ha_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_eagle['max_fert_induced_kg_n2o_n_ha_apy_crop'], categories=ordered_categories, ordered=True)
dist_max_t_apy_crop_df_eagle.head()
 


#%% 
d_max_t_eagle_2020 = (ggplot(dist_max_t_apy_crop_df_eagle)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_Tg_n2o_n_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest \ntotal $N_2O$ emissions\n(Eagle et al., 2020)')
       
    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
     #   rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
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
d_max_t_eagle_2020.save(filename="map_district_n2o_apy_crop_max_t_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_eagle_2020


#%% 
d_max_ha_eagle_2020 = (ggplot(dist_max_t_apy_crop_df_eagle)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_kg_n2o_n_ha_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest $N_2O$\nemissions per hectare\n(Eagle et al., 2020)')

    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
       #rect=element_rect(),
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
d_max_ha_eagle_2020.save(filename="map_district_n2o_apy_crop_max_ha_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_eagle_2020




#%%
dist_max_t_apy_crop_sql_bhatia = """
select *
from vwM_district_max_n2o_6_class_apy_crop_summary_bhatia_2013
"""
dist_max_t_apy_crop_df_bhatia = load_map_data(db_client_input, dist_max_t_apy_crop_sql_bhatia)

#%%

dist_max_t_apy_crop_df_bhatia['max_fert_induced_Tg_n2o_n_apy_crop'] = dist_max_t_apy_crop_df_bhatia['max_fert_induced_Tg_n2o_n_apy_crop'].replace(apy_crop_replacements)
dist_max_t_apy_crop_df_bhatia['max_fert_induced_kg_n2o_n_ha_apy_crop'] = dist_max_t_apy_crop_df_bhatia['max_fert_induced_kg_n2o_n_ha_apy_crop'].replace(apy_crop_replacements)

#%%
# Count the occurrences of each category
counts = dist_max_t_apy_crop_df_bhatia[dist_max_t_apy_crop_df_bhatia['max_fert_induced_Tg_n2o_n_apy_crop'] != 'Other Crops']['max_fert_induced_Tg_n2o_n_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_bhatia['max_fert_induced_Tg_n2o_n_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_bhatia['max_fert_induced_Tg_n2o_n_apy_crop'], categories=ordered_categories, ordered=True)

counts = dist_max_t_apy_crop_df_bhatia[dist_max_t_apy_crop_df_bhatia['max_fert_induced_kg_n2o_n_ha_apy_crop'] != 'Other Crops']['max_fert_induced_kg_n2o_n_ha_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_bhatia['max_fert_induced_kg_n2o_n_ha_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_bhatia['max_fert_induced_kg_n2o_n_ha_apy_crop'], categories=ordered_categories, ordered=True)
dist_max_t_apy_crop_df_bhatia.head()
 


#%% 
d_max_t_bhatia_2013 = (ggplot(dist_max_t_apy_crop_df_bhatia)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_Tg_n2o_n_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest \ntotal $N_2O$ emissions\n(Bhatia et al., 2013)')
       
    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
     #   rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
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
d_max_t_bhatia_2013.save(filename="map_district_n2o_apy_crop_max_t_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_bhatia_2013


#%% 
d_max_ha_bhatia_2013 = (ggplot(dist_max_t_apy_crop_df_bhatia)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_kg_n2o_n_ha_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest $N_2O$\nemissions per hectare\n(Bhatia et al., 2013)')

    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
       #rect=element_rect(),
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
d_max_ha_bhatia_2013.save(filename="map_district_n2o_apy_crop_max_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_bhatia_2013






#%%
dist_max_t_apy_crop_sql_ipcc = """
select *
from vwM_district_max_n2o_6_class_apy_crop_summary_ipcc_2019
"""
dist_max_t_apy_crop_df_ipcc = load_map_data(db_client_input, dist_max_t_apy_crop_sql_ipcc)

#%%

dist_max_t_apy_crop_df_ipcc['max_fert_induced_Tg_n2o_n_apy_crop'] = dist_max_t_apy_crop_df_ipcc['max_fert_induced_Tg_n2o_n_apy_crop'].replace(apy_crop_replacements)
dist_max_t_apy_crop_df_ipcc['max_fert_induced_kg_n2o_n_ha_apy_crop'] = dist_max_t_apy_crop_df_ipcc['max_fert_induced_kg_n2o_n_ha_apy_crop'].replace(apy_crop_replacements)

#%%
# Count the occurrences of each category
counts = dist_max_t_apy_crop_df_ipcc[dist_max_t_apy_crop_df_ipcc['max_fert_induced_Tg_n2o_n_apy_crop'] != 'Other Crops']['max_fert_induced_Tg_n2o_n_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_ipcc['max_fert_induced_Tg_n2o_n_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_ipcc['max_fert_induced_Tg_n2o_n_apy_crop'], categories=ordered_categories, ordered=True)

counts = dist_max_t_apy_crop_df_ipcc[dist_max_t_apy_crop_df_ipcc['max_fert_induced_kg_n2o_n_ha_apy_crop'] != 'Other Crops']['max_fert_induced_kg_n2o_n_ha_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_ipcc['max_fert_induced_kg_n2o_n_ha_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_ipcc['max_fert_induced_kg_n2o_n_ha_apy_crop'], categories=ordered_categories, ordered=True)
dist_max_t_apy_crop_df_ipcc.head()
 


#%% 
d_max_t_ipcc_2019 = (ggplot(dist_max_t_apy_crop_df_ipcc)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_Tg_n2o_n_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest \ntotal $N_2O$ emissions\n(IPCC 2019 Updated Methodology)')
       
    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
     #   rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
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
d_max_t_ipcc_2019.save(filename="map_district_n2o_apy_crop_max_t_ipcc_2019.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_ipcc_2019


#%% 
d_max_ha_ipcc_2019 = (ggplot(dist_max_t_apy_crop_df_ipcc)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_kg_n2o_n_ha_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest $N_2O$\nemissions per hectare\n(IPCC 2019 Updated Methodology)')

    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
       #rect=element_rect(),
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
d_max_ha_ipcc_2019.save(filename="map_district_n2o_apy_crop_max_ha_ipcc_2019.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_ipcc_2019


#%%
dist_max_t_apy_crop_sql_shcherbak = """
select *
from vwM_district_max_n2o_6_class_apy_crop_summary_shcherbak_2014
"""
dist_max_t_apy_crop_df_shcherbak = load_map_data(db_client_input, dist_max_t_apy_crop_sql_shcherbak)

#%%

dist_max_t_apy_crop_df_shcherbak['max_fert_induced_Tg_n2o_n_apy_crop'] = dist_max_t_apy_crop_df_shcherbak['max_fert_induced_Tg_n2o_n_apy_crop'].replace(apy_crop_replacements)
dist_max_t_apy_crop_df_shcherbak['max_fert_induced_kg_n2o_n_ha_apy_crop'] = dist_max_t_apy_crop_df_shcherbak['max_fert_induced_kg_n2o_n_ha_apy_crop'].replace(apy_crop_replacements)

#%%
# Count the occurrences of each category
counts = dist_max_t_apy_crop_df_shcherbak[dist_max_t_apy_crop_df_shcherbak['max_fert_induced_Tg_n2o_n_apy_crop'] != 'Other Crops']['max_fert_induced_Tg_n2o_n_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_shcherbak['max_fert_induced_Tg_n2o_n_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_shcherbak['max_fert_induced_Tg_n2o_n_apy_crop'], categories=ordered_categories, ordered=True)

counts = dist_max_t_apy_crop_df_shcherbak[dist_max_t_apy_crop_df_shcherbak['max_fert_induced_kg_n2o_n_ha_apy_crop'] != 'Other Crops']['max_fert_induced_kg_n2o_n_ha_apy_crop'].value_counts()
ordered_categories = list(counts.index) + ['Other Crops'] 
dist_max_t_apy_crop_df_shcherbak['max_fert_induced_kg_n2o_n_ha_apy_crop'] = pd.Categorical(dist_max_t_apy_crop_df_shcherbak['max_fert_induced_kg_n2o_n_ha_apy_crop'], categories=ordered_categories, ordered=True)
dist_max_t_apy_crop_df_shcherbak.head()
 


#%% 
d_max_t_shcherbak_2014 = (ggplot(dist_max_t_apy_crop_df_shcherbak)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_Tg_n2o_n_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest \ntotal $N_2O$ emissions\n(Shcherbak et al., 2014)')
       
    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
     #   rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
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
d_max_t_shcherbak_2014.save(filename="map_district_n2o_apy_crop_max_t_shcherbak_2014.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_shcherbak_2014


#%% 
d_max_ha_shcherbak_2014 = (ggplot(dist_max_t_apy_crop_df_shcherbak)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_fert_induced_kg_n2o_n_ha_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Crop with largest $N_2O$\nemissions per hectare\n(Shcherbak et al., 2014)')

    + scale_fill_manual(values=crop_colors,name='Crop')
    # + scale_fill_discrete(name='Crop')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
       #rect=element_rect(),
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
d_max_ha_shcherbak_2014.save(filename="map_district_n2o_apy_crop_max_ha_shcherbak_2014.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_shcherbak_2014


# %%

gA = pw.load_ggplot(d_max_ha_bhatia_2013 + labs(title='A')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gB = pw.load_ggplot(d_max_ha_eagle_2020 + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gC = pw.load_ggplot(d_max_ha_ipcc_2019 + labs(title='C')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gD = pw.load_ggplot(d_max_ha_shcherbak_2014 + labs(title='D')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))

gE = pw.load_ggplot(d_max_t_bhatia_2013 + labs(title='E')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gF = pw.load_ggplot(d_max_t_eagle_2020 + labs(title='F')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gG = pw.load_ggplot(d_max_t_ipcc_2019 + labs(title='G')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gH = pw.load_ggplot(d_max_t_shcherbak_2014 + labs(title='H')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
#%%
g = (gA| gB| gC| gD)/(gE| gF| gG| gH)
g.savefig(path.join(chart_dir,"map_district_n2o_apy_crop_max_plate.png"), dpi=300)
g


# %%
rename_dict = {
    'max_total_kg_n2o_n_ha_apy_crop_bhatia_2013': 'Bhatia\net al., 2013',
    'max_total_kg_n2o_n_ha_apy_crop_eagle_2020': 'Eagle\net al.,2020',
    'max_total_kg_n2o_n_ha_apy_crop_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology',
    'max_total_kg_n2o_n_ha_apy_crop_shcherbak_2014': 'Shcherbak\net al.,2014',

    'max_total_Tg_n2o_n_apy_crop_bhatia_2013': 'Bhatia\net al., 2013',
    'max_total_Tg_n2o_n_apy_crop_eagle_2020': 'Eagle\net al., 2020',
    'max_total_Tg_n2o_n_apy_crop_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology',
    'max_total_Tg_n2o_n_apy_crop_shcherbak_2014': 'Shcherbak\net al.,2014'
}


categorical_ha_columns = list(rename_dict.keys())[:4] # ['max_kg_n2o_ha_apy_crop_bhatia_2013', 'max_kg_n2o_ha_apy_crop_eagle_2020', 'max_kg_n2o_ha_apy_crop_ipcc_2019', 'max_kg_n2o_ha_apy_crop_shcherbak_2014']  
categorical_columns = list(rename_dict.keys())[4:] # ['max_kg_n2o_ha_apy_crop_bhatia_2013', 'max_kg_n2o_ha_apy_crop_eagle_2020', 'max_kg_n2o_ha_apy_crop_ipcc_2019', 'max_kg_n2o_ha_apy_crop_shcherbak_2014']  

# %%
from scipy.stats import chi2_contingency

cramer_v_sql = """select * from vwM_district_max_n2o_apy_crop_summary_all_models"""
cramer_v_df = load_map_data(db_client_input, cramer_v_sql)
cramer_v_df.head()

#%%
# Function to calculate Cramér's V for two categorical variables
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

cramers_v_matrix = pd.DataFrame(index=categorical_ha_columns, columns=categorical_ha_columns)

for col1 in categorical_ha_columns:
    for col2 in categorical_ha_columns:
        if col1 == col2:
            cramers_v_matrix.loc[col1, col2] = 1.0
        else:
            cramers_v_matrix.loc[col1, col2] = cramers_v(cramer_v_df[col1], cramer_v_df[col2])

print(cramers_v_matrix)

# %%
# Transform the Cramér's V matrix into long format for plotting
cramers_v_long = cramers_v_matrix.stack().reset_index()
cramers_v_long.columns = ['model1', 'model2', 'Cramers_V']

# Rename categories for better readability

cramers_v_long['model1'] = cramers_v_long['model1'].replace(rename_dict)
cramers_v_long['model2'] = cramers_v_long['model2'].replace(rename_dict)

# Convert to categorical with the new names
cramers_v_long['model1'] = pd.Categorical(values=cramers_v_long['model1'], categories=cramers_v_long['model1'].unique(), ordered=True)
cramers_v_long['model2'] = pd.Categorical(values=cramers_v_long['model2'], categories=cramers_v_long['model2'].unique(), ordered=True)

# Reverse the order of the categories
model1_categories = cramers_v_long['model1'].unique()[::-1]
model2_categories = cramers_v_long['model2'].unique()

# Change the data type of the 'Cramers_V' column to float64
cramers_v_long['Cramers_V'] = cramers_v_long['Cramers_V'].astype('float64')

cramers_v_long.head()


#%%
heatmap_plot = (
    ggplot(cramers_v_long, aes(x='model1', y='model2', fill='Cramers_V')) +
    geom_tile() +
    geom_text(aes(label='round(Cramers_V, 2)'), size=12, color='black') +
    scale_fill_gradient(low='blue', high='red', name="Cramér's V") +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District crops with highest $N_2O$ emission\n($Kg\ N_2O\ Ha^{-1}$ Cramér\'s V)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "apy_crop_kg_ha_cramers_v_heatmap.png"), dpi=300)


#%%
cramer_v_df.head()

#%%
cramers_v_matrix_total = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 == col2:
            cramers_v_matrix_total.loc[col1, col2] = 1.0
        else:
            cramers_v_matrix_total.loc[col1, col2] = cramers_v(cramer_v_df[col1], cramer_v_df[col2])

print(cramers_v_matrix_total)

# %%
# Transform the Cramér's V matrix into long format for plotting
cramers_v_total_long = cramers_v_matrix_total.stack().reset_index()
cramers_v_total_long.columns = ['model1', 'model2', 'Cramers_V']

# Rename categories for better readability

cramers_v_total_long['model1'] = cramers_v_total_long['model1'].replace(rename_dict)
cramers_v_total_long['model2'] = cramers_v_total_long['model2'].replace(rename_dict)

# Convert to categorical with the new names
cramers_v_total_long['model1'] = pd.Categorical(values=cramers_v_total_long['model1'], categories=cramers_v_total_long['model1'].unique(), ordered=True)
cramers_v_total_long['model2'] = pd.Categorical(values=cramers_v_total_long['model2'], categories=cramers_v_total_long['model2'].unique(), ordered=True)

# Reverse the order of the categories
model1_categories = cramers_v_total_long['model1'].unique()[::-1]
model2_categories = cramers_v_total_long['model2'].unique()

# Change the data type of the 'Cramers_V' column to float64
cramers_v_total_long['Cramers_V'] = cramers_v_total_long['Cramers_V'].astype('float64')

cramers_v_total_long.head()


#%%
heatmap_plot = (
    ggplot(cramers_v_total_long, aes(x='model1', y='model2', fill='Cramers_V')) +
    geom_tile() +
    geom_text(aes(label='round(Cramers_V, 2)'), size=12, color='black') +
    scale_fill_gradient(low='blue', high='red', name="Cramér's V") +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District crops with highest $N_2O$ emission\n ($Total\ Gg\ N_2O$ Cramér\'s V)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "apy_crop_t_n2o_cramers_v_heatmap.png"), dpi=300)



# %%
