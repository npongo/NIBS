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

# colors = ['green', 'blue', 'red', 'orange', 'purple', 'violet', 'aqua']
# colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00',  '#cab2d6','#6a3d9a','#ffff99','#b15928']


#%%
low_color = "green"
mid_color= "yellow"
high_color = "red"
colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
#colors = ['#81C000',  '#F2F900',  '#FFE700',  '#FF8F00',  '#FF0500']


#%%
farm_size_area_sql = """select *
from vwM_district_farmsize_area_proportion
where farm_size = '<2ha'
"""
farm_size_area_df = load_map_data(db_client_input, farm_size_area_sql)
farm_size_area_df.head()

#%%
farm_size_area_df.describe()

#%% 
units = "Proportion of\nAgricuultural land"

farm_size_area_g = (ggplot(farm_size_area_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="prop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
    + labs(title="Proportion of Agricutlure\nLand in Farms <2Ha")
     
   + scale_fill_gradientn(colors=colors, name=units)
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
farm_size_area_g.save(filename="map_farm_size_proportion_ag_area.png", path=chart_dir,  units='cm', dpi=300)
farm_size_area_g
# %%
