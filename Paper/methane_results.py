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

# %%  national results 
ch4_sql_national = """select * from vwG_national_ch4_summary_all_models"""
ch4_df_national = load_table_data(db_client_input, ch4_sql_national)
ch4_df_national

# %%
# ch4_sql_bhatia_2013 = """
# select *
# from vwM_district_ch4_co2e_bhatia_2013
# where gwp_time_period = 100
# """

ch4_sql_bhatia_2013 = """
select *
from vwM_district_rice_ch4_co2e_bhatia_2013
where gwp_time_period = 100
"""
ch4_df_bhatia_2013 = load_map_data(db_client_input, ch4_sql_bhatia_2013)
ch4_df_bhatia_2013.head()

#%% 
units = "$Gg\ CO_2e_{100}$"
ch4_df_bhatia_2013['total_Gg_co2e_map_p_score'] = ch4_df_bhatia_2013['total_Gg_co2e_map'].rank(pct=True)
breaks =  ch4_df_bhatia_2013['total_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
t_bhatia_2013 = (ggplot(ch4_df_bhatia_2013)
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
            + labs(title='Total rice $CH_4$ emissions\n(Bhatia et al., 2013)')
    
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  ) 
   # + scale_fill_discrete(name='Crop')
   # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,2000])
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=3000, high=high_color, name=units, limits=[0,5000])
   
   
    + theme(
         figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical' 
         #, panel_border=element_rect(color="black", size=1.5)  # Add border around the plot
         ##, plot_background=element_rect(color="black", size=2)  # Add border around the entire plot
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
t_bhatia_2013.save(filename="map_total_ch4_emissions_Gg_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
t_bhatia_2013

#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_bhatia_2013['kg_ch4_co2e_ha_p_score'] = ch4_df_bhatia_2013['kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_bhatia_2013['kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
ha_bhatia_2013 = (ggplot(ch4_df_bhatia_2013)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
    + labs(title='Per hectare rice $CH_4$ emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=3000, high=high_color, name=units, limits=[0,5500])
   
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
ha_bhatia_2013.save(filename="map_ch4_emissions_kg_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
ha_bhatia_2013


#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_bhatia_2013['min_kg_ch4_co2e_ha_p_score'] = ch4_df_bhatia_2013['min_kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_bhatia_2013['min_kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
max_r = ch4_df_bhatia_2013['min_kg_ch4_co2e_ha'].max()*1.1
min_ha_bhatia_2013 = (ggplot(ch4_df_bhatia_2013)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="min_kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Minimum per hectare rice $CH_4$ emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=max_r/2.0, high=high_color, name=units, limits=[0,max_r])
   
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
min_ha_bhatia_2013.save(filename="map_ch4_emissions_min_kg_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
min_ha_bhatia_2013


#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_bhatia_2013['max_kg_ch4_co2e_ha_p_score'] = ch4_df_bhatia_2013['max_kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_bhatia_2013['max_kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
max_r = ch4_df_bhatia_2013['max_kg_ch4_co2e_ha'].quantile(.99)*1.1
min_r = ch4_df_bhatia_2013['max_kg_ch4_co2e_ha'].quantile(.01)*.9
median_r = ch4_df_bhatia_2013['max_kg_ch4_co2e_ha'].median()
max_ha_bhatia_2013 = (ggplot(ch4_df_bhatia_2013)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Maximum per hectare rice $CH_4$ emissions\n(Bhatia et al., 2013)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=5000, high=high_color, name=units, limits=[min_r,max_r])
   
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
max_ha_bhatia_2013.save(filename="map_ch4_emissions_max_kg_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
max_ha_bhatia_2013


#%%
# Open the image from my computer
# image = open_image_local(path.join(chart_dir, "map_total_ch4_emissions_Gg_bhatia_2013.png"))

# fig_bhatia_2013 = ha_bhatia_2013.draw()
# image_xaxis = 0.58
# image_yaxis = 0.001
# image_width = 0.4
# image_height = 0.4  # Same as width since our logo is a square

# # Define the position for the image axes
# ax_image = fig_bhatia_2013.add_axes([image_xaxis,
#                          image_yaxis,
#                          image_width,
#                          image_height]
#                        )

# # Display the image
# ax_image.imshow(image)
# ax_image.axis('off')  # Remove axis of the image
# fig_bhatia_2013.savefig(os.path.join(chart_dir, "map_ch4_emissions_kg_ha_total_bhatia_2013.png"), dpi=300)
# fig_bhatia_2013

## Yan 2005


# %%
ch4_sql_yan_2005 = """
select *
from vwM_district_rice_ch4_co2e_yan_2005
where gwp_time_period = 100
"""
ch4_df_yan_2005 = load_map_data(db_client_input, ch4_sql_yan_2005)
ch4_df_yan_2005.describe()

#%% 
units = "$Gg\ CO_2e_{100}$"
ch4_df_yan_2005['total_Gg_co2e_map_p_score'] = ch4_df_yan_2005['total_Gg_co2e_map'].rank(pct=True)
breaks =  ch4_df_yan_2005['total_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
t_yan_2005 = (ggplot(ch4_df_yan_2005)
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
            + labs(title='Total rice $CH_4$ emissions\n(Yan et al., 2005)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,2200])
   
    + theme(
         figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
         #, panel_border=element_rect(color="black", size=1.5)  # Add border around the plot
         ##, plot_background=element_rect(color="black", size=2)  # Add border around the entire plot
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
t_yan_2005.save(filename="map_total_ch4_emissions_Gg_yan_2005.png", path=chart_dir,  units='cm', dpi=300)
t_yan_2005



#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_yan_2005['kg_ch4_co2e_ha_p_score'] = ch4_df_yan_2005['kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_yan_2005['kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
ha_yan_2005 = (ggplot(ch4_df_yan_2005)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $CH_4$ emissions\n(Yan et al., 2005)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=2000, high=high_color, name=units, limits=[0,4000])
   
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
ha_yan_2005.save(filename="map_ch4_emissions_kg_ha_yan_2005.png", path=chart_dir,  units='cm', dpi=300)
ha_yan_2005

#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_yan_2005['min_kg_ch4_co2e_ha_p_score'] = ch4_df_yan_2005['min_kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_yan_2005['min_kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
min_ha_yan_2005 = (ggplot(ch4_df_yan_2005)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="min_kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Minimum per hectare rice $CH_4$ emissions\n(Yan et al., 2005)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=3000, high=high_color, name=units, limits=[0,5500])
   
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
min_ha_yan_2005.save(filename="map_ch4_emissions_min_kg_ha_yan_2005.png", path=chart_dir,  units='cm', dpi=300)
min_ha_yan_2005

#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_yan_2005['max_kg_ch4_co2e_ha_p_score'] = ch4_df_yan_2005['max_kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_yan_2005['max_kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
max_ha_yan_2005 = (ggplot(ch4_df_yan_2005)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Maximum per hectare rice $CH_4$ emissions\n(Yan et al., 2005)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=5000, high=high_color, name=units) # , limits=[0,10000])
   
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
max_ha_yan_2005.save(filename="map_ch4_emissions_max_kg_ha_yan_2005.png", path=chart_dir,  units='cm', dpi=300)
max_ha_yan_2005

#%%
# Open the image from my computer
# image = open_image_local(path.join(chart_dir, "map_total_ch4_emissions_Gg_yan_2005.png"))

# fig_yan_2005 = ha_yan_2005.draw()
# image_xaxis = 0.58
# image_yaxis = 0.001
# image_width = 0.4
# image_height = 0.4  # Same as width since our logo is a square

# # Define the position for the image axes
# ax_image = fig_yan_2005.add_axes([image_xaxis,
#                          image_yaxis,
#                          image_width,
#                          image_height]
#                        )

# # Display the image
# ax_image.imshow(image)
# ax_image.axis('off')  # Remove axis of the image
# fig_yan_2005.savefig(os.path.join(chart_dir, "map_ch4_emissions_kg_ha_total_yan_2005.png"), dpi=300)
# fig_yan_2005


# %%
ch4_sql_nikolaisen_2023 = """
select *
from vwM_district_rice_ch4_co2e_nikolaisen_2023
where gwp_time_period = 100
"""
ch4_df_nikolaisen_2023 = load_map_data(db_client_input, ch4_sql_nikolaisen_2023)
ch4_df_nikolaisen_2023.describe()

#%% 
units = "$Gg\ CO_2e_{100}$"
ch4_df_nikolaisen_2023['total_Gg_co2e_map_p_score'] = ch4_df_nikolaisen_2023['total_Gg_co2e_map'].rank(pct=True)
breaks =  ch4_df_nikolaisen_2023['total_Gg_co2e_map'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
t_nikolaisen_2023 = (ggplot(ch4_df_nikolaisen_2023)
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
            + labs(title='\nTotal rice $CH_4$ emissions\n(Nikolaisen et al., 2023)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=650, high=high_color, name=units, limits=[0,2000])
   
    + theme(
         figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
         #, panel_border=element_rect(color="black", size=1.5)  # Add border around the plot
         ###, plot_background=element_rect(color="black", size=2)  # Add border around the entire plot
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
t_nikolaisen_2023.save(filename="map_total_ch4_emissions_Gg_nikolaisen_2023.png", path=chart_dir,  units='cm', dpi=300)
t_nikolaisen_2023

#%% 
units = "$Kg\ CO_2e_{100}$ $Ha^{-1}$"
ch4_df_nikolaisen_2023['kg_ch4_co2e_ha_p_score'] = ch4_df_nikolaisen_2023['kg_ch4_co2e_ha'].rank(pct=True)
breaks =  ch4_df_nikolaisen_2023['kg_ch4_co2e_ha'].quantile([0.4, 0.7, 0.9,1]).round(0)
  
ha_nikolaisen_2023 = (ggplot(ch4_df_nikolaisen_2023)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_ch4_co2e_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $CH_4$ emissions\n(Nikolaisen et al., 2023)')
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, labels=breaks  )
   # + scale_fill_discrete(name='Crop')
    # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=3000, high=high_color, name=units)#, limits=[0,10000])
   
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
ha_nikolaisen_2023.save(filename="map_ch4_emissions_kg_ha_nikolaisen_2023.png", path=chart_dir,  units='cm', dpi=300)
ha_nikolaisen_2023


#%%
# # Open the image from my computer
# image = open_image_local(path.join(chart_dir, "map_total_ch4_emissions_Gg_nikolaisen_2023.png"))

# fig_nikolaisen_2023 = ha_nikolaisen_2023.draw()
# image_xaxis = 0.58
# image_yaxis = 0.001
# image_width = 0.4
# image_height = 0.4  # Same as width since our logo is a square

# # Define the position for the image axes
# ax_image = fig_nikolaisen_2023.add_axes([image_xaxis,
#                          image_yaxis,
#                          image_width,
#                          image_height]
#                        )

# # Display the image
# ax_image.imshow(image)
# ax_image.axis('off')  # Remove axis of the image
# fig_nikolaisen_2023.savefig(os.path.join(chart_dir, "map_ch4_emissions_kg_ha_total_nikolaisen_2023.png"), dpi=300)
# fig_nikolaisen_2023



# %%
gA = pw.load_ggplot(ha_bhatia_2013 + labs(title='A')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gB = pw.load_ggplot(ha_yan_2005 + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gC = pw.load_ggplot(ha_nikolaisen_2023 + labs(title='C')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))

gD = pw.load_ggplot(t_bhatia_2013 + labs(title='D')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gE = pw.load_ggplot(t_yan_2005 + labs(title='E')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gF = pw.load_ggplot(t_nikolaisen_2023 + labs(title='F')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
#%%
g = (gA| gB| gC)/(gD| gE| gF)
g.savefig(path.join(chart_dir,"map_ch4_emissions_plate.png"), dpi=300)
g



# %%
model_uncer_sql = """select 'Bhatia et al., 2013' as ref, mean_kg_ch4_ha, sd_kg_ch4_ha, mean_min_kg_ch4_ha, mean_max_kg_ch4_ha
from vwG_national_ch4_summary_bhatia_2013
union
select 'Nikolaisen et al., 2023' as ref, mean_kg_ch4_ha, sd_kg_ch4_ha, mean_min_kg_ch4_ha, mean_max_kg_ch4_ha
from vwG_national_ch4_summary_nikolaisen_2023
union
select 'Yan et al., 2005' as ref, mean_kg_ch4_ha, sd_kg_ch4_ha, mean_min_kg_ch4_ha, mean_max_kg_ch4_ha
from vwG_national_ch4_summary_yan_2005"""
model_uncer_df = load_table_data(db_client_input, model_uncer_sql)
model_uncer_df

#%%
dist_max_t_farm_size_sql_bhatia = """
select *
from vwM_district_max_ch4_farm_size_summary_bhatia_2013
"""
dist_max_t_farm_size_df_bhatia = load_map_data(db_client_input, dist_max_t_farm_size_sql_bhatia)
farm_size_replacements = {
    'MARGINAL (BELOW 1.0)': '<2.0 Ha',  # '<1.0 Ha',
    'SMALL (1.0 - 1.99)': '<2.0 Ha',  # '1.0 - 1.99 Ha',
    'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
    'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
    'LARGE (10 AND ABOVE)': '≥10 Ha'
}
dist_max_t_farm_size_df_bhatia['max_t_ch4_farm_size'] = dist_max_t_farm_size_df_bhatia['max_t_ch4_farm_size'].replace(farm_size_replacements)
dist_max_t_farm_size_df_bhatia['max_kg_ch4_ha_farm_size'] = dist_max_t_farm_size_df_bhatia['max_kg_ch4_ha_farm_size'].replace(farm_size_replacements)

dist_max_t_farm_size_df_bhatia['max_t_ch4_farm_size'] = pd.Categorical(dist_max_t_farm_size_df_bhatia['max_t_ch4_farm_size']
                                    , categories=[ '<2.0 Ha', '2.0 - 3.99 Ha', '4.0 - 9.99 Ha','≥10 Ha' ], ordered=True)
             
dist_max_t_farm_size_df_bhatia['max_kg_ch4_ha_farm_size'] = pd.Categorical(dist_max_t_farm_size_df_bhatia['max_kg_ch4_ha_farm_size']
                                              , categories=[ '<2.0 Ha', '2.0 - 3.99 Ha', '4.0 - 9.99 Ha','≥10 Ha' ], ordered=True)

dist_max_t_farm_size_df_bhatia.head()

#%% 
d_max_t_bhatia_2013 = (ggplot(dist_max_t_farm_size_df_bhatia)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_t_ch4_farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='\nRice, landholding size with\nlargest total $CH_4$\nemissions(Bhatia et al., 2013)')
     
    + scale_fill_discrete(name='Landholding\nSize')
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
d_max_t_bhatia_2013.save(filename="map_district_farm_size_max_t_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_bhatia_2013

#%% 
d_max_ha_bhatia_2013 = (ggplot(dist_max_t_farm_size_df_bhatia)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_kg_ch4_ha_farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Rice, landholding size\nwith largest $CH_4$ emissions per\nhectare(Bhatia et al., 2013)')
     
    + scale_fill_discrete(name='Landholding\nSize')
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
d_max_ha_bhatia_2013.save(filename="map_district_farm_size_max_ha_bhatia_2013.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_bhatia_2013

# %%

#%%
dist_max_t_farm_size_sql_nikolaisen = """
select *
from vwM_district_max_ch4_farm_size_summary_nikolaisen_2023
"""
dist_max_t_farm_size_df_nikolaisen = load_map_data(db_client_input, dist_max_t_farm_size_sql_nikolaisen)
farm_size_replacements = {
    'MARGINAL (BELOW 1.0)': '<2.0 Ha',  # '<1.0 Ha',
    'SMALL (1.0 - 1.99)': '<2.0 Ha',  # '1.0 - 1.99 Ha',
    'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
    'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
    'LARGE (10 AND ABOVE)': '≥10 Ha'
}
dist_max_t_farm_size_df_nikolaisen['max_t_ch4_farm_size'] = dist_max_t_farm_size_df_nikolaisen['max_t_ch4_farm_size'].replace(farm_size_replacements)
dist_max_t_farm_size_df_nikolaisen['max_kg_ch4_ha_farm_size'] = dist_max_t_farm_size_df_nikolaisen['max_kg_ch4_ha_farm_size'].replace(farm_size_replacements)

dist_max_t_farm_size_df_nikolaisen['max_t_ch4_farm_size'] = pd.Categorical(dist_max_t_farm_size_df_nikolaisen['max_t_ch4_farm_size']
                                    , categories=[ '<2.0 Ha', '2.0 - 3.99 Ha', '4.0 - 9.99 Ha','≥10 Ha' ], ordered=True)
             
dist_max_t_farm_size_df_nikolaisen['max_kg_ch4_ha_farm_size'] = pd.Categorical(dist_max_t_farm_size_df_nikolaisen['max_kg_ch4_ha_farm_size']
                                              , categories=[ '<2.0 Ha', '2.0 - 3.99 Ha', '4.0 - 9.99 Ha','≥10 Ha' ], ordered=True)

dist_max_t_farm_size_df_nikolaisen.head()

#%% 
d_max_t_nikolaisen_2023 = (ggplot(dist_max_t_farm_size_df_nikolaisen)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_t_ch4_farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
    + labs(title='\nRice, landholding size with\nlargest total $CH_4$emissions\n(Nikolaisen et al., 2023)')
     
    + scale_fill_discrete(name='Landholding\nSize')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
        # rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
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
d_max_t_nikolaisen_2023.save(filename="map_district_farm_size_max_t_nikolaisen_2023.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_nikolaisen_2023

#%% 
d_max_ha_nikolaisen_2023 = (ggplot(dist_max_t_farm_size_df_nikolaisen)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_kg_ch4_ha_farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Rice, landholding size with\nlargest $CH_4$ emissions per\nhectare (Nikolaisen et al., 2023)')
     
    + scale_fill_discrete(name='Landholding\nSize')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
        # rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=20),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
       #  , legend_position=(.12,.2)
         , legend_position=(.7,.02)
     )
)
d_max_ha_nikolaisen_2023.save(filename="map_district_farm_size_max_ha_nikolaisen_2023.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_nikolaisen_2023

#%%
dist_max_t_farm_size_sql_Yan = """
select *
from vwM_district_max_ch4_farm_size_summary_Yan_2005
"""
dist_max_t_farm_size_df_Yan = load_map_data(db_client_input, dist_max_t_farm_size_sql_Yan)
farm_size_replacements = {
    'MARGINAL (BELOW 1.0)': '<2.0 Ha',  # '<1.0 Ha',
    'SMALL (1.0 - 1.99)': '<2.0 Ha',  # '1.0 - 1.99 Ha',
    'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
    'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
    'LARGE (10 AND ABOVE)': '≥10 Ha'
}
dist_max_t_farm_size_df_Yan['max_t_ch4_farm_size'] = dist_max_t_farm_size_df_Yan['max_t_ch4_farm_size'].replace(farm_size_replacements)
dist_max_t_farm_size_df_Yan['max_kg_ch4_ha_farm_size'] = dist_max_t_farm_size_df_Yan['max_kg_ch4_ha_farm_size'].replace(farm_size_replacements)

dist_max_t_farm_size_df_Yan['max_t_ch4_farm_size'] = pd.Categorical(dist_max_t_farm_size_df_Yan['max_t_ch4_farm_size']
                                    , categories=[ '<2.0 Ha', '2.0 - 3.99 Ha', '4.0 - 9.99 Ha','≥10 Ha' ], ordered=True)
             
dist_max_t_farm_size_df_Yan['max_kg_ch4_ha_farm_size'] = pd.Categorical(dist_max_t_farm_size_df_Yan['max_kg_ch4_ha_farm_size']
                                              , categories=[ '<2.0 Ha', '2.0 - 3.99 Ha', '4.0 - 9.99 Ha','≥10 Ha' ], ordered=True)

dist_max_t_farm_size_df_Yan.head()

#%% 
d_max_t_yan_2005 = (ggplot(dist_max_t_farm_size_df_Yan)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_t_ch4_farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='\nRice, landholding size with largest\ntotal $CH_4$ emissions (Yan et al., 2005)')
     
    + scale_fill_discrete(name='Landholding\nSize')
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
d_max_t_yan_2005.save(filename="map_district_farm_size_max_t_yan_2005.png", path=chart_dir,  units='cm', dpi=300)
d_max_t_yan_2005

#%% 
d_max_ha_yan_2005 = (ggplot(dist_max_t_farm_size_df_Yan)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_kg_ch4_ha_farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Rice, landholding size with largest $CH_4$\nemissions per hectare (Yan et al., 2005)')
     
    + scale_fill_discrete(name='Landholding\nSize')
   # # + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,3000])
   
 + theme(
        figure_size=(7,8),
  #       rect=element_rect(fill=(0, 0, 0), color=(0, 0, 0)),
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
d_max_ha_yan_2005.save(filename="map_district_farm_size_max_ha_yan_2005.png", path=chart_dir,  units='cm', dpi=300)
d_max_ha_yan_2005



# %%
gA = pw.load_ggplot(d_max_ha_bhatia_2013 + labs(title='A')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gB = pw.load_ggplot(d_max_ha_yan_2005 + labs(title='B')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gC = pw.load_ggplot(d_max_ha_nikolaisen_2023 + labs(title='C')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))

gD = pw.load_ggplot(d_max_t_bhatia_2013 + labs(title='D')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gE = pw.load_ggplot(d_max_t_yan_2005 + labs(title='E')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
gF = pw.load_ggplot(d_max_t_nikolaisen_2023 + labs(title='F')  + theme( plot_title= element_text(ha='left', size=32)), figsize=(7,8))
#%%
g = (gA| gB| gC)/(gD| gE| gF)
g.savefig(path.join(chart_dir,"map_district_ch4_farm_size_max_plate.png"), dpi=300)
g


# %%
spearman_sql  = """select * from vwM_district_rank_ch4_farm_size_summary_all_models"""

spearman_df = load_map_data(db_client_input, spearman_sql)
spearman_df.head()

#%%
def spearman_corr(group):
    return group.drop(columns='geog_checksum').corr(method='spearman')


selected_columns = spearman_df.loc[:, ['geog_checksum', 'rank_kg_ch4_ha_farm_size_bhatia_2013', 'rank_kg_ch4_ha_farm_size_nikolaisen_2023','rank_kg_ch4_ha_farm_size_yan_2005']]
print(selected_columns.head())
grouped_spearman_corr = selected_columns.groupby('geog_checksum').apply(spearman_corr)

#%%
# Transform the correlation matrix into long format
grouped_spearman_corr_long = grouped_spearman_corr.stack().reset_index()
grouped_spearman_corr_long.columns = ['geog_checksum', 'model1', 'model2', 'Spearman_Correlation']
# Remove rows where model1 is equal to model2
# grouped_spearman_corr_long = grouped_spearman_corr_long[grouped_spearman_corr_long['model1'] != grouped_spearman_corr_long['model2']]
# grouped_spearman_corr_long = grouped_spearman_corr_long[(grouped_spearman_corr_long['model1'].apply(lambda x: not(x.endswith('yan_2005')))) 
#                                                             & (grouped_spearman_corr_long['model2'].apply(lambda x: not(x.endswith('bhatia_2013'))))]

grouped_spearman_corr_long.head(9)

#%%
# condition = (grouped_spearman_corr_long['model1'] == 'rank_kg_ch4_ha_farm_size_bhatia_2013') & (grouped_spearman_corr_long['model2'] == 'rank_kg_ch4_ha_farm_size_yan_2005')
# grouped_spearman_corr_long.loc[condition, ['model1', 'model2']] = grouped_spearman_corr_long.loc[condition, ['model2', 'model1']].values

# grouped_spearman_corr_long.head(9)

#%%
# Calculate the average Spearman correlation by model1 and model2
average_spearman_corr = grouped_spearman_corr_long.groupby(['model1', 'model2'])['Spearman_Correlation'].mean().reset_index()


rename_dict = {
    'rank_kg_ch4_ha_farm_size_bhatia_2013': 'Bhatia\net al., 2013',
    'rank_kg_ch4_ha_farm_size_nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'rank_kg_ch4_ha_farm_size_yan_2005': 'Yan et al.,\n2005'
}
average_spearman_corr['model1'] = average_spearman_corr['model1'].replace(rename_dict)
average_spearman_corr['model2'] = average_spearman_corr['model2'].replace(rename_dict)

average_spearman_corr['model1'] = pd.Categorical(values=average_spearman_corr['model1'], categories=average_spearman_corr['model1'].unique(), ordered=True)  
average_spearman_corr['model2'] = pd.Categorical(values=average_spearman_corr['model2'], categories=average_spearman_corr['model2'].unique(), ordered=True)        

print(average_spearman_corr.head())

#%%
heatmap_data = average_spearman_corr.pivot(index="model1", columns="model2", values="Spearman_Correlation")
heatmap_data.head()


# %%
# Reverse the order of the categories
model1_categories = average_spearman_corr['model1'].unique()[::-1]
model2_categories = average_spearman_corr['model2'].unique()#[::-1]
print(model1_categories, model2_categories)

#%%
heatmap_plot = (
    ggplot(average_spearman_corr, aes(x='model1', y='model2', fill='Spearman_Correlation')) +
    geom_tile() +
    geom_text(aes(label='round(Spearman_Correlation, 2)'), size=8, color='black') +
    scale_fill_gradient(low='blue', high='red',name="Corr") +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='Average District Spearman Correlation', x='', y='') +
    theme(figure_size=(5.5,5),
          axis_text_x=element_text(rotation=0, hjust='center'),
          axis_text_y=element_text(rotation=90, hjust='center'),
         panel_background=element_rect(fill='white') )
)

# Display the heatmap
print(heatmap_plot)

# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "ch4_farm_size_average_spearman_correlation_heatmap_plotnine.png"), dpi=300)


# %%
from scipy.stats import chi2_contingency

cramer_v_sql = """select * from vwM_district_max_ch4_farm_size_summary_all_models"""
cramer_v_df = load_map_data(db_client_input, cramer_v_sql)
cramer_v_df.head()

#%%
# Function to calculate Cramér's V for two categorical variables
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

categorical_columns = ['max_kg_ch4_ha_farm_size_bhatia_2013', 'max_kg_ch4_ha_farm_size_nikolaisen_2023', 'max_kg_ch4_ha_farm_size_yan_2005']
cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

for col1 in categorical_columns:
    for col2 in categorical_columns:
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
rename_dict = {
    'max_kg_ch4_ha_farm_size_bhatia_2013': 'Bhatia\net al., 2013',
    'max_kg_ch4_ha_farm_size_nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'max_kg_ch4_ha_farm_size_yan_2005': 'Yan et al.,\n2005'
}
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
    scale_fill_gradient(low='blue', high='red', name='Cramér\'s V') +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District farm size with highest methane emission\n($Kg\ Ch_4\ Ha^{-1}$ Cramér\'s V Heatmap)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, hjust='center'),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "farm_size_kg_ha_ch4_cramers_v_heatmap.png"), dpi=300)



#%%
categorical_columns = ['max_t_ch4_farm_size_bhatia_2013', 'max_t_ch4_farm_size_nikolaisen_2023', 'max_t_ch4_farm_size_yan_2005']
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
rename_dict = {
    'max_t_ch4_farm_size_bhatia_2013': 'Bhatia\net al., 2013',
    'max_t_ch4_farm_size_nikolaisen_2023': 'Nikolaisen\net al., 2023',
    'max_t_ch4_farm_size_yan_2005': 'Yan et al.,\n2005'
}
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
    scale_fill_gradient(low='blue', high='red', name='Cramér\'s V') +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District farm size with highest methane emission\n ($Total\ Gg\ Ch_4$ Cramér\'s V)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, hjust='center'),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "farm_size_t_ch4_cramers_v_heatmap.png"), dpi=300)




# # %%
# fert_type_by_farm_size_sql = """select * from vwG_inorganic_organic_fert_by_farm_size"""
# fert_type_by_farm_size_df = load_table_data(db_client_input, fert_type_by_farm_size_sql)

# fert_type_by_farm_size_df['fert_type'] = pd.Categorical(fert_type_by_farm_size_df['fert_type'], categories=['inorganic', 'organic'], ordered=True)
# farm_size_replacements = {
#     'MARGINAL (BELOW 1.0)': '<1.0 Ha',  # '<1.0 Ha',
#     'SMALL (1.0 - 1.99)': '1.0-1.9 Ha',  # '1.0 - 1.99 Ha',
#     'SEMI-MEDIUM (2.0 - 3.99)': '2.0-3.9 Ha',
#     'MEDIUM (4.0 - 9.99)': '4.0-9.9 Ha',
#     'LARGE (10 AND ABOVE)': '≥10 Ha'
# }
# fert_type_by_farm_size_df['farm_size'] = fert_type_by_farm_size_df['farm_size'].replace(farm_size_replacements)

# fert_type_by_farm_size_df['farm_size'] = pd.Categorical(fert_type_by_farm_size_df['farm_size']
#                                     , categories=[ '<1.0 Ha', '1.0-1.9 Ha', '2.0-3.9 Ha', '4.0-9.9 Ha','≥10 Ha' ], ordered=True)
# fert_type_by_farm_size_df.head()

# # %%
# g = (ggplot(fert_type_by_farm_size_df)
#      + geom_bar(aes(x='farm_size', y='mean_n_rate_kg_ha', fill='fert_type'), stat='identity', position='dodge')
#      + geom_errorbar(aes(x='farm_size', ymin='mean_n_rate_kg_ha-sd_n_rate_kg_ha', ymax='mean_n_rate_kg_ha+sd_n_rate_kg_ha', group='fert_type'), position='dodge')
#      + labs(title='Fertilizer rate by farm size', x='Farm size', y='Mean N application\nrate ($kg\ ha^{-1}$)')
#      + scale_fill_manual(values=['blue', 'green'], name='Fertilizer')
#      + theme(axis_text_x=element_text(rotation=0, hjust=0.5))
#      + theme(figure_size=(6, 4))
# )

# g.save(os.path.join(chart_dir, "farm_size_n_rate_by_fert_type.png"), dpi=300)
# g


# # %%
# import os

# sql_dir = r'E:\npongo Dropbox\benjamin clark\CIL\GHG Calcs\Ensemble'
# sql_files = [f for f in os.listdir(sql_dir) if f.endswith('.sql')]
# print(sql_files[:5])

# # %%
# def search_text_in_files(file, directory, text):
#     with open(os.path.join(directory, file), 'r') as f:
#         if text in f.read():
#             return True
#     return False

# text_to_search = "n2o_total_nitrogen_samples_summary"
# result = [f for f in sql_files if search_text_in_files(f, sql_dir, text_to_search)]
# print(result)
# # %%

# %%
ha_corr_sql = """
select * from [vwA_district_ch4_rice_model_corr_wide]
"""

ha_corr_df = load_table_data(db_client_input, ha_corr_sql)
ha_corr_df.head()

# %%
corr_kg_co2e_ha_df = ha_corr_df.loc[:, ['mean_kg_ch4_ha__bhatia_2013','mean_kg_ch4_ha__nikolaisen_2023','mean_kg_ch4_ha__yan_2005']]
corr_kg_co2e_ha_df.head()     
                                    
# %%
def cap_first_letter(s):
    if 'ipcc' in s:
      return 'IPCC 2019\nUpdated\nMethodology' 
    return s[0].upper() + s[1:]

corr_kg_co2e_ha_matrix = corr_kg_co2e_ha_df.corr(method='spearman')
corr_kg_co2e_ha_matrix.columns = [cap_first_letter(x.replace('mean_kg_ch4_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.columns]
corr_kg_co2e_ha_matrix.index = [cap_first_letter(x.replace('mean_kg_ch4_ha__','').replace('_','\net al., ')) for x in corr_kg_co2e_ha_matrix.index]
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
    labs(title='District Model Comparison\n(Spearman Correlation $Kg\ CH_4\ Ha^{-1}$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_rice_kg_ch4_ha_spearman_heatmap.png"), dpi=300)


# %%
corr_Gg_co2e_df = ha_corr_df.loc[:, ['total_t_ch4__bhatia_2013','total_t_ch4__nikolaisen_2023','total_t_ch4__yan_2005']]
corr_Gg_co2e_df.head() 
corr_Gg_co2e_matrix = corr_Gg_co2e_df.corr(method='spearman')
corr_Gg_co2e_matrix.columns = [cap_first_letter(x.replace('total_t_ch4__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.columns]
corr_Gg_co2e_matrix.index = [cap_first_letter(x.replace('total_t_ch4__','').replace('_','\net al., ')) for x in corr_Gg_co2e_matrix.index]
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
    labs(title='District Model Comparison\n(Spearman Correlation $Total\ District\ CH_4$)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10, vjust='center', hjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "district_rice_ch4_Gg_spearman_heatmap.png"), dpi=300)



# %%
