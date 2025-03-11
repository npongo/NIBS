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
colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
#colors = ['#81C000',  '#F2F900',  '#FFE700',  '#FF8F00',  '#FF0500']

# %%
n2o_eagle_sql = """
select *
from vwM_district_n2o_co2e_upland_crop_results_eagle_2020
where gwp_time_period = 100"""
n2o_eagle_df = load_map_data(db_client_input, n2o_eagle_sql)
n2o_eagle_df.head()

# %%
units = "$Gg\ CO_2e_{100}$"

n2o_eagle_df['direct_Gg_co2e_map_p_score'] = n2o_eagle_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_eagle_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_eagle_Gg_g = (ggplot(n2o_eagle_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total non-rice $N_2O$\nfertilizer induced emissions\n(Eagle et al., 2020')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units, limits=[0,150])
   
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
n2o_upland_crop_eagle_Gg_g.save(filename="map_total_n2o_emissions_Gg_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_eagle_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"
 
n2o_eagle_df['direct_kg_n2o_co2e_ha_p_score'] = n2o_eagle_df['direct_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_eagle_df['direct_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_eagle_ha_g = (ggplot(n2o_eagle_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare non-rice $N_2O$\nfertilizer induced emissions\n(Eagle et al., 2020)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=250, high=high_color, name=units, limits=[0,600])
   
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
n2o_upland_crop_eagle_ha_g.save(filename="map_n2o_emissions_kg_ha_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_eagle_ha_g

# %%
n2o_ipcc_sql = """
select *
from vwM_district_n2o_co2e_upland_crop_results_ipcc_2019
where gwp_time_period = 100"""
n2o_ipcc_df = load_map_data(db_client_input, n2o_ipcc_sql)
n2o_ipcc_df.head()


# %%
units = "$Gg\ CO_2e_{100}$"
 
n2o_ipcc_df['direct_Gg_co2e_map_p_score'] = n2o_ipcc_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_ipcc_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_ipcc_Gg_g = (ggplot(n2o_ipcc_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total non-rice $N_2O$\nfertilizer induced emissions\n(IPCC Update 2019)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_upland_crop_ipcc_Gg_g.save(filename="map_total_n2o_emissions_Gg_ipcc_2019.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_ipcc_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_ipcc_df['fert_kg_n2o_co2e_ha_p_score'] = n2o_ipcc_df['fert_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_ipcc_df['fert_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_ipcc_ha_g = (ggplot(n2o_ipcc_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="fert_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare non-rice $N_2O$\nfertilizer induced emissions\n(IPCC Update 2019)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=400, high=high_color, name=units) #, limits=[0,600])
   
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
n2o_upland_crop_ipcc_ha_g.save(filename="map_n2o_emissions_kg_ha_ipcc_2019.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_ipcc_ha_g


# %%
n2o_bhatia_sql = """
select *
from vwM_district_n2o_co2e_upland_crop_results_bhatia_2013
where gwp_time_period = 100"""
n2o_bhatia_df = load_map_data(db_client_input, n2o_bhatia_sql)
n2o_bhatia_df.head()


# %%
units = "$Gg\ CO_2e_{100}$"

n2o_bhatia_df['direct_Gg_co2e_map_p_score'] = n2o_bhatia_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_bhatia_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)


n2o_upland_crop_bhatia_Gg_g = (ggplot(n2o_bhatia_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total non-rice $N_2O$\nfertilizer induced emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   ## + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_upland_crop_bhatia_Gg_g.save(filename="map_total_n2o_emissions_Gg_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_bhatia_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_bhatia_df['fert_kg_n2o_co2e_ha_p_score'] = n2o_bhatia_df['fert_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_bhatia_df['fert_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_bhatia_ha_g = (ggplot(n2o_bhatia_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="fert_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare non-rice $N_2O$\nfertilizer induced emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=150, high=high_color, name=units) #, limits=[0,600])
   
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
n2o_upland_crop_bhatia_ha_g.save(filename="map_n2o_emissions_kg_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_bhatia_ha_g

# %%
n2o_shcherbak_sql = """
select *
from vwM_district_n2o_co2e_upland_crop_results_shcherbak_2014
where gwp_time_period = 100"""
n2o_shcherbak_df = load_map_data(db_client_input, n2o_shcherbak_sql)
n2o_shcherbak_df.head()
# %%

# %%
units = "$Gg\ CO_2e_{100}$"

n2o_shcherbak_df['direct_Gg_co2e_map_p_score'] = n2o_shcherbak_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_shcherbak_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_shcherbak_Gg_g = (ggplot(n2o_shcherbak_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total non-rice $N_2O$\nfertilizer induced emissions\n(Shcherbak et al., 2014)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=150, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_upland_crop_shcherbak_Gg_g.save(filename="map_total_n2o_emissions_Gg_shcherbak_2014.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_shcherbak_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_shcherbak_df['direct_kg_n2o_co2e_ha_p_score'] = n2o_shcherbak_df['direct_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_shcherbak_df['direct_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_upland_crop_shcherbak_ha_g = (ggplot(n2o_shcherbak_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare non-rice $N_2O$\nfertilizer induced emissions\n(Shcherbak et al., 2014)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=500, high=high_color, name=units) #, limits=[0,600])
   
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
n2o_upland_crop_shcherbak_ha_g.save(filename="map_n2o_emissions_kg_ha_shcherbak_2014.png", path=chart_dir,  units='cm', dpi=300)
n2o_upland_crop_shcherbak_ha_g

# %%

gA = pw.load_ggplot(n2o_upland_crop_bhatia_ha_g + labs(title='A')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gB = pw.load_ggplot(n2o_upland_crop_eagle_ha_g + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gC = pw.load_ggplot(n2o_upland_crop_ipcc_ha_g + labs(title='C')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gD = pw.load_ggplot(n2o_upland_crop_shcherbak_ha_g + labs(title='D')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))

gE = pw.load_ggplot(n2o_upland_crop_bhatia_Gg_g + labs(title='E')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gF = pw.load_ggplot(n2o_upland_crop_eagle_Gg_g + labs(title='F')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gG = pw.load_ggplot(n2o_upland_crop_ipcc_Gg_g + labs(title='G')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gH = pw.load_ggplot(n2o_upland_crop_shcherbak_Gg_g + labs(title='H')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))

#%%
g = (gA| gB| gC| gD)/(gE| gF| gG| gH)
g.savefig(path.join(chart_dir,"map_district_n2o_upland_crop_ha_Gg_plate.png"), dpi=300)
g










# %%
n2o_rice_eagle_sql = """
select *
from vwM_district_n2o_co2e_rice_results_eagle_2020
where gwp_time_period = 100"""
n2o_rice_eagle_df = load_map_data(db_client_input, n2o_rice_eagle_sql)
n2o_rice_eagle_df.head()

# %%
units = "$Gg\ CO_2e_{100}$"

n2o_rice_eagle_df['direct_Gg_co2e_map_p_score'] = n2o_rice_eagle_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_rice_eagle_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_eagle_Gg_g = (ggplot(n2o_rice_eagle_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $N_2O$\nfertilizer induced emissions\n(Eagle et al., 2020')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units, limits=[0,150])
   
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
n2o_rice_eagle_Gg_g.save(filename="map_rice_total_n2o_emissions_Gg_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_eagle_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_rice_eagle_df['direct_kg_n2o_co2e_ha_p_score'] = n2o_rice_eagle_df['direct_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_rice_eagle_df['direct_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_eagle_ha_g = (ggplot(n2o_rice_eagle_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $N_2O$\nfertilizer induced emissions\n(Eagle et al., 2020)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=250, high=high_color, name=units, limits=[0,600])
   
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
n2o_rice_eagle_ha_g.save(filename="map_rice_n2o_emissions_kg_ha_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_eagle_ha_g

# %%
n2o_rice_ipcc_sql = """
select *
from vwM_district_n2o_co2e_rice_results_ipcc_2019
where gwp_time_period = 100"""
n2o_rice_ipcc_df = load_map_data(db_client_input, n2o_rice_ipcc_sql)
n2o_rice_ipcc_df.head()


# %%
units = "$Gg\ CO_2e_{100}$"

n2o_rice_ipcc_df['direct_Gg_co2e_map_p_score'] = n2o_rice_ipcc_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_rice_ipcc_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_ipcc_Gg_g = (ggplot(n2o_rice_ipcc_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $N_2O$\nfertilizer induced emissions\n(IPCC Update 2019)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_rice_ipcc_Gg_g.save(filename="map_rice_total_n2o_emissions_Gg_ipcc_2019.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_ipcc_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_rice_ipcc_df['fert_kg_n2o_co2e_ha_p_score'] = n2o_rice_ipcc_df['fert_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_rice_ipcc_df['fert_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_ipcc_ha_g = (ggplot(n2o_rice_ipcc_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="fert_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $N_2O$\nfertilizer induced emissions\n(IPCC Update 2019)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=400, high=high_color, name=units) #, limits=[0,600])
   
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
n2o_rice_ipcc_ha_g.save(filename="map_rice_n2o_emissions_kg_ha_ipcc_2019.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_ipcc_ha_g



# %%
n2o_rice_bhatia_sql = """
select *
from vwM_district_n2o_co2e_rice_results_bhatia_2013
where gwp_time_period = 100"""
n2o_rice_bhatia_df = load_map_data(db_client_input, n2o_rice_bhatia_sql)
n2o_rice_bhatia_df.head()


# %%
units = "$Gg\ CO_2e_{100}$"

n2o_rice_bhatia_df['direct_Gg_co2e_map_p_score'] = n2o_rice_bhatia_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_rice_bhatia_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_bhatia_Gg_g = (ggplot(n2o_rice_bhatia_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $N_2O$\nfertilizer induced emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_rice_bhatia_Gg_g.save(filename="map_rice_total_n2o_emissions_Gg_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_bhatia_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_rice_bhatia_df['direct_kg_n2o_co2e_ha_p_score'] = n2o_rice_bhatia_df['direct_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_rice_bhatia_df['direct_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_bhatia_ha_g = (ggplot(n2o_rice_bhatia_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $N_2O$\nfertilizer induced emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=150, high=high_color, name=units) #, limits=[0,600])
   
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
n2o_rice_bhatia_ha_g.save(filename="map_rice_n2o_emissions_kg_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_bhatia_ha_g

# %%
n2o_rice_shcherbak_sql = """
select *
from vwM_district_n2o_co2e_rice_results_shcherbak_2014
where gwp_time_period = 100"""
n2o_rice_shcherbak_df = load_map_data(db_client_input, n2o_rice_shcherbak_sql)
n2o_rice_shcherbak_df.head()
# %%

# %%
units = "$Gg\ CO_2e_{100}$"

n2o_rice_shcherbak_df['direct_Gg_co2e_map_p_score'] = n2o_rice_shcherbak_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_rice_shcherbak_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_shcherbak_Gg_g = (ggplot(n2o_rice_shcherbak_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $N_2O$\nfertilizer induced emissions\n(Shcherbak et al., 2014)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=150, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_rice_shcherbak_Gg_g.save(filename="map_rice_total_n2o_emissions_Gg_shcherbak_2014.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_shcherbak_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_rice_shcherbak_df['direct_kg_n2o_co2e_ha_p_score'] = n2o_rice_shcherbak_df['direct_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_rice_shcherbak_df['direct_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_shcherbak_ha_g = (ggplot(n2o_rice_shcherbak_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $N_2O$\nfertilizer induced emissions\n(Shcherbak et al., 2014)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=500, high=high_color, name=units) #, limits=[0,600])
   
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
n2o_rice_shcherbak_ha_g.save(filename="map_rice_n2o_emissions_kg_ha_shcherbak_2014.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_shcherbak_ha_g


# %%
n2o_rice_akiyama_sql = """
select *
from [vwM_district_n2o_co2e_rice_results_hiroko_akiyama_2005]
where gwp_time_period = 100"""
n2o_rice_akiyama_df = load_map_data(db_client_input, n2o_rice_akiyama_sql)
n2o_rice_akiyama_df.head()
# %%

# %%
units = "$Gg\ CO_2e_{100}$"

n2o_rice_akiyama_df['direct_Gg_co2e_map_p_score'] = n2o_rice_akiyama_df['direct_Gg_co2e_map'].rank(pct=True)
breaks =  n2o_rice_akiyama_df['direct_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)

n2o_rice_akiyama_Gg_g = (ggplot(n2o_rice_akiyama_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $N_2O$\nfertilizer induced emissions\n(Akiyama et al., 2014)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=10, high=high_color, name=units) #, limits=[0,150])
   
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
n2o_rice_akiyama_Gg_g.save(filename="map_rice_total_n2o_emissions_Gg_akiyama_2014.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_akiyama_Gg_g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$"

n2o_rice_akiyama_df['direct_kg_n2o_co2e_ha_p_score'] = n2o_rice_akiyama_df['direct_kg_n2o_co2e_ha'].rank(pct=True)
breaks =  n2o_rice_akiyama_df['direct_kg_n2o_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
n2o_rice_akiyama_ha_g = (ggplot(n2o_rice_akiyama_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
    
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $N_2O$\nfertilizer induced emissions\n(Akiyama et al., 2014)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
 # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=midpoint,  high=high_color, name=units, rescaler=rescale) #, limits=[0,600])
   
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
n2o_rice_akiyama_ha_g.save(filename="map_rice_n2o_emissions_kg_ha_akiyama_2014.png", path=chart_dir,  units='cm', dpi=300)
n2o_rice_akiyama_ha_g


# %%

gA = pw.load_ggplot(n2o_rice_bhatia_ha_g + labs(title='A')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gB = pw.load_ggplot(n2o_rice_eagle_ha_g + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gC = pw.load_ggplot(n2o_rice_ipcc_ha_g + labs(title='C')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gD = pw.load_ggplot(n2o_rice_shcherbak_ha_g + labs(title='D')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gE = pw.load_ggplot(n2o_rice_akiyama_ha_g + labs(title='E')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))

gF = pw.load_ggplot(n2o_rice_bhatia_Gg_g + labs(title='F')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gG = pw.load_ggplot(n2o_rice_eagle_Gg_g + labs(title='G')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gH = pw.load_ggplot(n2o_rice_ipcc_Gg_g + labs(title='H')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gI = pw.load_ggplot(n2o_rice_shcherbak_Gg_g + labs(title='I')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
gJ = pw.load_ggplot(n2o_rice_akiyama_Gg_g + labs(title='J')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))

#%%
g = (gA| gB| gC| gD| gE)/(gF| gG| gH| gI| gJ)
g.savefig(path.join(chart_dir,"map_district_n2o_rice_ha_Gg_plate.png"), dpi=300)
g




# %%
upland_ha_corr_sql = """
select * from vwA_district_n2o_co2e_upland_crop_model_corr_wide
"""

upland_ha_corr_df = load_table_data(db_client_input, upland_ha_corr_sql)
upland_ha_corr_df.head()

# %%
corr_kg_co2e_ha_df = upland_ha_corr_df.loc[:, ['kg_n2o_co2e_ha__bhatia_2013','kg_n2o_co2e_ha__ipcc_2019','kg_n2o_co2e_ha__eagle_2020','kg_n2o_co2e_ha__shcherbak_2014']]
corr_kg_co2e_ha_df.head()     
                                    
# %%
def cap_first_letter(s):
    if 'ipcc' in s:
      return 'IPCC 2019\nUpdated\nMethodology' 
    return s[0].upper() + s[1:]

corr_kg_co2e_ha_matrix = corr_kg_co2e_ha_df.corr(method='spearman')
corr_kg_co2e_ha_matrix.columns = [cap_first_letter(x.replace('kg_n2o_co2e_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.columns]
corr_kg_co2e_ha_matrix.index = [cap_first_letter(x.replace('kg_n2o_co2e_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.index]
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
    labs(title='None-rice District Model Comparison\n(Spearman Correlation $Kg\ N_2O\ Ha^{-1}$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_upland_crop_kg_n2o_co2e_ha_spearman_heatmap.png"), dpi=300)


# %%
corr_Gg_co2e_df = upland_ha_corr_df.loc[:, ['total_Gg_co2e_map__bhatia_2013','total_Gg_co2e_map__ipcc_2019','total_Gg_co2e_map__eagle_2020','total_Gg_co2e_map__shcherbak_2014']]
corr_Gg_co2e_df.head() 

corr_Gg_co2e_matrix = corr_Gg_co2e_df.corr(method='spearman')
corr_Gg_co2e_matrix.columns = [cap_first_letter(x.replace('total_Gg_co2e_map__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.columns]
corr_Gg_co2e_matrix.index = [cap_first_letter(x.replace('total_Gg_co2e_map__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.index]
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
    labs(title='None-rice District Model Comparison\n(Spearman Correlation $Gg\ N_2O$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_upland_crop_n2o_Gg_co2e_spearman_heatmap.png"), dpi=300)


# %%
upland_ha_corr_sql = """
select * from vwA_district_n2o_co2e_rice_model_corr_wide
"""

upland_ha_corr_df = load_table_data(db_client_input, upland_ha_corr_sql)
upland_ha_corr_df.head()

# %%
corr_kg_co2e_ha_df = upland_ha_corr_df.loc[:, ['kg_n2o_co2e_ha__bhatia_2013','kg_n2o_co2e_ha__ipcc_2019','kg_n2o_co2e_ha__hiroko_akiyama_2005','kg_n2o_co2e_ha__eagle_2020','kg_n2o_co2e_ha__shcherbak_2014']]
corr_kg_co2e_ha_df.head()     
                                    
# %%
def cap_first_letter(s):
    if 'ipcc' in s:
      return 'IPCC 2019\nUpdated\nMethodology' 
    if 'hiroko' in s:
      return 'Akiyama\net al., 2005'
    return s[0].upper() + s[1:]

corr_kg_co2e_ha_matrix = corr_kg_co2e_ha_df.corr(method='spearman')
corr_kg_co2e_ha_matrix.columns = [cap_first_letter(x.replace('kg_n2o_co2e_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.columns]
corr_kg_co2e_ha_matrix.index = [cap_first_letter(x.replace('kg_n2o_co2e_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.index]
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
    labs(title='Rice District Model Comparison\n(Spearman Correlation $Kg\ N_2O\ Ha^{-1}$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_rice_kg_co2e_ha_spearman_heatmap.png"), dpi=300)


# %%
corr_Gg_co2e_df = upland_ha_corr_df.loc[:, ['total_Gg_co2e_map__bhatia_2013','total_Gg_co2e_map__ipcc_2019','total_Gg_co2e_map__hiroko_akiyama_2005','total_Gg_co2e_map__eagle_2020','total_Gg_co2e_map__shcherbak_2014']]
corr_Gg_co2e_df.head() 

corr_Gg_co2e_matrix = corr_Gg_co2e_df.corr(method='spearman')
corr_Gg_co2e_matrix.columns = [cap_first_letter(x.replace('total_Gg_co2e_map__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.columns]
corr_Gg_co2e_matrix.index = [cap_first_letter(x.replace('total_Gg_co2e_map__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.index]
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
    labs(title='Rice District Model Comparison\n(Spearman Correlation $Gg\ N_2O$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_rice_n2o_Gg_co2e_spearman_heatmap.png"), dpi=300)



# %%
national_summary_sql = """select *
from vwG_National_co2e_summmary_all_gases_all_models_long
where gwp_time_period = 100 
"""

national_summary_df = load_table_data(db_client_input, national_summary_sql)
national_summary_df.head()

#%%
def reverse_text(s):
    return s[::-1]
gases_dic = {"$Residue\ Burning\ CO_2e_{100}$": "Residue Burning\n$CO_2e_{100}$",
             "$N_2O\ CO_2e_{100}$, (rice)": "$N_2O\ CO_2e_{100}$\n(rice)",
             "$N_2O\ CO_2e_{100}$, (none-rice crops)": "$N_2O\ CO_2e_{100}$\n(none-rice crops)",
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


#%%
national_summary_df['gas'] = pd.Categorical(national_summary_df['gas'], categories=gases, ordered=True)
national_summary_df['model_ref'] = pd.Categorical(national_summary_df['model_ref'], categories=list(models_dic.values()), ordered=False)
national_summary_df.head()

#%%
model_name_dic = {v[0]:v[1] for v in national_summary_df[['model_ref','model_name']].drop_duplicates().values}
model_name_dic 

#%%
national_summary_df

#%%-
def x_lab(s):
  return [model_name_dic[x] for x in s]


g = (ggplot(national_summary_df)
      + geom_bar(aes(x='model_ref', y='value', fill='gas'), stat='identity', width=.8)
      + geom_errorbar(aes(x='model_ref', ymin='value - sd', ymax='value + sd'), width=.5)
      + labs(title='Total Cropping Emission by Model', x="Models",y="Kg $CO_2e_{100}\ Ha^{-1}$                  Tg $CO_2e_{100}$")
      #+ scale_y_continuous(limits=(0,250))
      + scale_x_discrete(legend=False, labels=x_lab)
      + scale_fill_discrete(legend=False, name='Gas')
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
g.save(os.path.join(chart_dir, "national_all_models_all_emissions_facet.png"), dpi=300)
g

# %%
total_national_summary_df = national_summary_df[national_summary_df['statistic'] == 'total']

total_g = (ggplot(total_national_summary_df)
      + geom_bar(aes(x='model_name', y='value', fill='gas'), stat='identity', width=.8)
      + geom_errorbar(aes(x='model_name', ymin='value - sd', ymax='value + sd'), width=.5)
      + labs(title='Total Cropping Emission by Model', x="Models", y="Tg $CO_2e_{100}$")
      + scale_y_continuous(limits=(0,250))
      + scale_x_discrete(legend=False)
      + scale_fill_discrete(legend=False, name='Gas')
      + theme_minimal()
      + theme(figure_size= (10,4), 
            title=element_text(size=22, backgroundcolor='white'), 
          #  rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
          axis_text_x=element_text(rotation=90, size=10, hjust='center'),
          axis_text_y=element_text(rotation=0, size=10, vjust='center', hjust='left'),
      )
   
)
total_g.save(os.path.join(chart_dir, "total_national_all_models_all_emissions.png"), dpi=300)
total_g

# %%
area_national_summary_df = national_summary_df[national_summary_df['statistic'] == 'mean']

mean_g = (ggplot(area_national_summary_df)
      + geom_bar(aes(x='model_ref', y='value', fill='gas'), stat='identity', width=.5)
      + geom_errorbar(aes(x='model_ref', ymin='value - sd', ymax='value + sd'), width=.3)
      + labs(title='Total Cropping Emission by Model', x="Models", y="Kg $CO_2e_{100}\ Ha^{-1}$")
      + scale_y_continuous()
      + scale_x_discrete(legend=False)
      + scale_fill_discrete(legend=False, name='Gas')
      + theme_minimal()
      + theme(figure_size= (10,4), 
            title=element_text(size=22, backgroundcolor='white'), 
        #    rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
          axis_text_x=element_text(rotation=90, size=8, hjust='center'),
          axis_text_y=element_text(rotation=0, size=8, vjust='center', hjust='center'),
      )
   
)

mean_g.save(os.path.join(chart_dir, "area_national_all_models_all_emissions.png"), dpi=300)
mean_g

# %%
pw.overwrite_axisgrid()
t = pw.load_ggplot(total_g + labs(title='A')  + theme(  plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))
m = pw.load_ggplot(mean_g + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32) , legend_text=element_text(size=22)))  
fig =  t/m
fig.savefig(path.join(chart_dir,"national_total_mean_all_emissions.png"), dpi=300)
fig


#%%
national_summary_df

# %%
national_summary_sql = """select *
from vwG_National_co2e_summmary_all_gases_all_models
where gwp_time_period = 100
	and model_ref in('Nikolaisen et al., 2023','Eagle et al., 2020 (rice)','Eagle et al., 2020 (none-rice crops)','Biomass Altas (44 crops)')
"""

national_summary_df = load_table_data(db_client_input, national_summary_sql)

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
national_summary_df.head()

#%%
ch4_g = (ggplot(national_summary_df)
      + geom_bar(aes(x='model_ref', y='total_Tg_co2e', fill='gas'), stat='identity', width=.8)
      + geom_errorbar(aes(x='model_ref', ymin='total_Tg_co2e - sd_total_Tg_co2e', ymax='total_Tg_co2e + sd_total_Tg_co2e'), width=.5)
      + labs(title='Total Cropping Emissions by Model', x="Models", y="Tg $CO_2e_{100}$")
      + scale_y_continuous(limits=(0,150))
      + scale_x_discrete(legend=False)
      + scale_fill_discrete(legend=False, name='Gas')
      + theme_minimal()
      + theme(figure_size= (8,8), 
            title=element_text(size=22, backgroundcolor='white'), 
            rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
            legend_text=element_text(size=18),
          axis_text_x=element_text(rotation=90, size=18, hjust='center'),
          axis_text_y=element_text(rotation=0, size=18, vjust='center', hjust='center'),
      )
   
)
ch4_g.save(os.path.join(chart_dir, "national_edf_models_all_emissions.png"), dpi=300)
ch4_g
# %%
