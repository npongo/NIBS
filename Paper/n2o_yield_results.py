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
crops = ['Rice','Wheat','Maize']
refs = ['shcherbak_2014', 'ipcc_2019', 'eagle_2020', 'bhatia_2013']

#%%
dfs = []
for ref in refs:
    for crop in crops:
        sql = f"""select * from [dbo].[vwM_district_n2o_co2e_apy_crop_results_{ref}]
        where apy_crop = '{crop}'
            and gwp_time_period = 100
        """
        print(sql)
        df = load_map_data(db_client_input, sql)
        dfs.append((ref, crop, df))


# %%     
units = "$\\frac{Kg\ CO_2e_{100}} {Kg\ Yield}$"
for ref, crop, df in dfs:

    df['kg_n2o_kg_yield_p_score'] = df['kg_n2o_kg_yield'].rank(pct=True)
    breaks =  df['kg_n2o_kg_yield'].quantile([0.4, 0.7, 0.9,1]).round(2)
    #print(breaks)
    g = (ggplot(df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     + coord_cartesian()
     + theme_void()
     
    + geom_map(aes(fill="kg_n2o_kg_yield_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=f"{crop} N$_2$O  Emissions By Yield\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
     
    + scale_fill_gradientn(colors=colors, values= [0,0.4, 0.7, 0.9,1], name=units, breaks=[0.4, 0.7, 0.9,1], labels=breaks  )
   
     + theme(
        figure_size=(7,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=22)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
         , legend_position=(.7,.02)
        )
    )

    print(g)
    g.save(filename=f"map_{crop}_kg_no2_co2e100_kg_yield_{ref}.png", path=chart_dir,  units='cm', dpi=300)
    g = None 
    df = None
    breaks = None





# %%

spearman_dfs = []
for crop in crops:
    spearman_sql  = f"""select * from [vwM_district_rank_n2o_yield_summary_all_models]
        where apy_crop = '{crop}'"""
    print(spearman_sql)
    df = load_map_data(db_client_input, spearman_sql)
    spearman_dfs.append((crop, df))


#%%
def spearman_corr(group):
    return group.drop(columns='apy_crop').corr(method='spearman')

grouped_spearman_corrs = []
for crop, df in spearman_dfs:
    selected_columns = df.loc[:, ['kg_n2o_kg_yield_bhatia_2013', 'kg_n2o_kg_yield_ipcc_2019','kg_n2o_kg_yield_eagle_2020','kg_n2o_kg_yield_shcherbak_2014']]
    print(selected_columns.head())
    grouped_spearman_corr = selected_columns.corr(method='spearman')
    grouped_spearman_corrs.append((crop, grouped_spearman_corr))
    # print(grouped_spearman_corr)


#%%
grouped_spearman_corrs[0][1]

#%%
rename_dict = {
    'kg_n2o_kg_yield_bhatia_2013': 'Bhatia\net al. 2013',
    'kg_n2o_kg_yield_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology',
    'kg_n2o_kg_yield_eagle_2020': 'Eagle et al.\n2020',
    'kg_n2o_kg_yield_shcherbak_2014': 'Shcherbak et al.\n2014'
}

corr_g_data = []
# Transform the correlation matrix into long format
for crop, spearman_corr in grouped_spearman_corrs:
    spearman_corr_long = spearman_corr.stack().reset_index()
    spearman_corr_long.columns = ['model1', 'model2', 'Spearman_Correlation']
    spearman_corr_long['model1'] = spearman_corr_long['model1'].replace(rename_dict)
    spearman_corr_long['model2'] = spearman_corr_long['model2'].replace(rename_dict)

    spearman_corr_long['model1'] = pd.Categorical(values=spearman_corr_long['model1'], categories=spearman_corr_long['model1'].unique(), ordered=True)  
    spearman_corr_long['model2'] = pd.Categorical(values=spearman_corr_long['model2'], categories=spearman_corr_long['model2'].unique(), ordered=True)        
    corr_g_data.append((crop, spearman_corr_long))

#%%


#%%

for crop, spearman_corr in corr_g_data:
    heatmap_data = spearman_corr.pivot(index="model1", columns="model2", values="Spearman_Correlation")

    model1_categories = spearman_corr['model1'].unique()[::-1]
    model2_categories = spearman_corr['model2'].unique()#[::-1]
    print(model1_categories, model2_categories)

    heatmap_plot = (
        ggplot(spearman_corr, aes(x='model1', y='model2', fill='Spearman_Correlation')) +
        geom_tile() +
        geom_text(aes(label='round(Spearman_Correlation, 2)'), size=12, color='black') +
        scale_fill_gradient(low='blue', high='red',name="Corr") +
        scale_x_discrete(limits=model1_categories) +
        scale_y_discrete(limits=model2_categories) +
        labs(title=f'{crop} Spearman Correlation of District Rankings', x='', y='') +
        theme(figure_size=(5.5,5),
            axis_text_x=element_text(rotation=0, hjust='center'),
            axis_text_y=element_text(rotation=90, hjust='center', vjust='center'),
            panel_background=element_rect(fill='white') )
    )

    # Display the heatmap
    print(heatmap_plot)

    # Save the heatmap
    heatmap_plot.save(os.path.join(chart_dir, f"n2o_yield_{crop}_spearman_correlation_heatmap_plotnine.png"), dpi=300)


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


apy_crop_replacements = {
    'other oilseeds': 'Oilseeds',  # '<1.0 Ha',
    'Moong(Green Gram)': 'Green Gram',  # '<1.0 Ha',
    'Rapeseed &Mustard': 'Rapeseed & Mustard',  # '<1.0 Ha',
}


#%%
dfs = []
for ref in refs:
    sql = f"""select * from [dbo].[vwM_district_max_n2o_6_class_apy_crop_yield_{ref}]
    """
    print(sql)
    df = load_map_data(db_client_input, sql)
    df['max_apy_crop'] = df['max_apy_crop'].replace(apy_crop_replacements)
    counts = df[df['max_apy_crop'] != 'Other Crops']['max_apy_crop'].value_counts()
    ordered_categories = list(counts.index) + ['Other Crops'] 
    df['max_apy_crop'] = pd.Categorical(df['max_apy_crop'], categories=ordered_categories, ordered=True)

    dfs.append((ref, df))

# %%
for ref,  df in dfs:

    g = (ggplot(df)
       + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="max_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=f"Crop with largest $N_2O$ emissions per yield\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
     
    + scale_fill_manual(values=crop_colors,name='Crop')
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

    print(g)
    g.save(filename=f"map_max_n2o_6_class_apy_crop_yield_{ref}.png", path=chart_dir,  units='cm', dpi=300)
# %%








# %%
rename_dict = {
    'apy_crop_bhatia_2013': 'Bhatia\net al., 2013',
    'apy_crop_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology',
    'apy_crop_eagle_2020': 'Eagle\net al.,2020',
    'apy_crop_shcherbak_2014': 'Shcherbak\net al.,2014',
}


categorical_columns = list(rename_dict.keys()) # ['max_kg_n2o_ha_farm_size_bhatia_2013', 'max_kg_n2o_ha_farm_size_eagle_2020', 'max_kg_n2o_ha_farm_size_ipcc_2019', 'max_kg_n2o_ha_farm_size_shcherbak_2014']  

# %%
print(categorical_columns)

# %%
from scipy.stats import chi2_contingency

cramer_v_sql = """select * from [vwM_district_max_n2o_apy_crop_yield_all_models]"""
cramer_v_df = load_map_data(db_client_input, cramer_v_sql)
cramer_v_df.head()

#%%
# Function to calculate Cramér's V for two categorical variables
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

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

cramers_v_long['model1'] = cramers_v_long['model1'].replace(rename_dict)
cramers_v_long['model2'] = cramers_v_long['model2'].replace(rename_dict)

# Convert to categorical with the new names
cramers_v_long['model1'] = pd.Categorical(values=cramers_v_long['model1'], categories=rename_dict.values(), ordered=True)
cramers_v_long['model2'] = pd.Categorical(values=cramers_v_long['model2'], categories=rename_dict.values(), ordered=True)

# Reverse the order of the categories
model1_categories = ['Shcherbak\net al.,2014',
 'Eagle\net al.,2020',
 'IPCC 2019\nUpdated\nMethodology',
 'Bhatia\net al., 2013']
model2_categories = model1_categories[::-1]

# Change the data type of the 'Cramers_V' column to float64
cramers_v_long['Cramers_V'] = cramers_v_long['Cramers_V'].astype('float64')

cramers_v_long.head()


#%%
heatmap_plot = (
    ggplot(cramers_v_long, aes(x='model1', y='model2', fill='Cramers_V')) +
    geom_tile() +
    geom_text(aes(label='round(Cramers_V, 2)'), size=16, color='black') +
    scale_fill_gradient(low='blue', high='red', name="Cramér's V") +
    scale_x_discrete(limits=model1_categories) +
    scale_y_discrete(limits=model2_categories) +
    labs(title='District crops with highest yield scaled $N_2O$ emission\n($\\frac{Kg\ N_2O}{Kg\ Yield}$ Cramér\'s V)', x='', y='') +
    theme(axis_text_x=element_text(rotation=0, size=10, hjust='center'),
          axis_text_y=element_text(rotation=90, size=10,   vjust='center'),
          axis_ticks=element_blank(),
          panel_background=element_rect(fill='white'))
)

# Display the heatmap
print(heatmap_plot)
# Save the heatmap
heatmap_plot.save(os.path.join(chart_dir, "apy_crop_kg_kg_yield_cramers_v_heatmap.png"), dpi=300)

# %%
model1_categories
# %%
