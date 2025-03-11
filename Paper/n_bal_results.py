#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import os
import numpy as np
import pandas as pd
import PostDoc.db_clients.mssql_db_client as mssql  
from PostDoc.Plotting.PlottingFunctions import *
from plotnine import ggplot, aes, labs
from PIL import Image
import numpy as np
import duckdb


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
ref = 'eagle_2020'

#%%
sql = f"""select * from [dbo].[vwM_district_n_balance_results_eagle_2020]
        where gwp_time_period = 100
        """
print(sql)
df = load_map_data(db_client_input, sql)
df.head()

#%%
units = "$Kg\ N\ Ha^{-1}$"
df['mean_n_balance_n_kg_ha_p_score'] = df['mean_n_balance_n_kg_ha'].rank(pct=True)
breaks =  df['mean_n_balance_n_kg_ha'].quantile([0.4, 0.7, 0.9,1]).round(2)
#print(breaks)
g = (ggplot(df)
+ geom_map(india, fill='grey', color=None, show_legend=False)
    + scale_x_continuous(limits=(67.5,97.5))
    + scale_y_continuous(limits=(7.5,37.5))
    + coord_cartesian()
    + theme_void()
    
+ geom_map(aes(fill="mean_n_balance_n_kg_ha_p_score"), color=None, show_legend=True)
+ geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
        + labs(title=f"All Crops Nitrogen Balance\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
    
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
g.save(filename=f"map_n_balance_all_crops_kg_ha_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)

#%%
units = "Gg N"
df['total_n_balance_n_Gg_n_map_p_score'] = df['total_n_balance_n_Gg_n_map'].rank(pct=True)
breaks =  df['total_n_balance_n_Gg_n_map'].quantile([0.4, 0.7, 0.9,1]).round(2)
#print(breaks)
g = (ggplot(df)
+ geom_map(india, fill='grey', color=None, show_legend=False)
    + scale_x_continuous(limits=(67.5,97.5))
    + scale_y_continuous(limits=(7.5,37.5))
    + coord_cartesian()
    + theme_void()
    
+ geom_map(aes(fill="total_n_balance_n_Gg_n_map_p_score"), color=None, show_legend=True)
+ geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
        + labs(title=f"All Crops Nitrogen Balance\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
    
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
g.save(filename=f"map_n_balance_all_crops_total_Gg_n_eagle_2020.png", path=chart_dir,  units='cm', dpi=300)

#%%
print(refs)
dfs = []
for crop in crops:
    sql = f"""select * from [dbo].[vwM_district_n_balance_apy_crop_results_{ref}]
    where apy_crop = '{crop}'
        and gwp_time_period = 100
    """
    print(sql)
    df = load_map_data(db_client_input, sql)
    dfs.append((ref, crop, df))


# %%     
for ref, crop, df in dfs:
    units = "$Kg\ N\ Ha^{-1}$"
    df['mean_n_balance_n_kg_ha_p_score'] = df['mean_n_balance_n_kg_ha'].rank(pct=True)
    breaks =  df['mean_n_balance_n_kg_ha'].quantile([0.4, 0.7, 0.9,1]).round(2)
    #print(breaks)
    g_ha = (ggplot(df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
    + scale_x_continuous(limits=(67.5,97.5))
    + scale_y_continuous(limits=(7.5,37.5))
    + coord_cartesian()
    + theme_void()
     
    + geom_map(aes(fill="mean_n_balance_n_kg_ha_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=f"{crop} Nitrogen Balance\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
     
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

    print(g_ha)
    g_ha.save(filename=f"map_{crop}_n_bal_kg_ha_{ref}.png", path=chart_dir,  units='cm', dpi=300)
    g_ha = None
    breaks = None

    units = "Gg N"
    df['total_n_balance_n_Gg_n_map_p_score'] = df['total_n_balance_n_Gg_n_map'].rank(pct=True)
    breaks =  df['total_n_balance_n_Gg_n_map'].quantile([0.4, 0.7, 0.9,1]).round(2)
    #print(breaks)
    g_Gg = (ggplot(df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     + coord_cartesian()
     + theme_void()
     
    + geom_map(aes(fill="total_n_balance_n_Gg_n_map_p_score"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=f"{crop} Nitrogen Balance\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
     
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

    print(g_Gg)
    g_Gg.save(filename=f"map_{crop}_n_bal_total_Gg_n_{ref}.png", path=chart_dir,  units='cm', dpi=300)
    g_Gg = None
    df = None
    breaks = None

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


apy_crop_replacements = {
    'other oilseeds': 'Oilseeds',  # '<1.0 Ha',
    'Moong(Green Gram)': 'Green Gram',  # '<1.0 Ha',
    'Rapeseed &Mustard': 'Rapeseed & Mustard',  # '<1.0 Ha',
    'Rapeseed &Mustard': 'Rapeseed & Mustard',  # '<1.0 Ha',
    'Cotton(lint)': 'Cotton',  # '<1.0 Ha',
}

#%%
dfs = []
for ref in refs:
    sql = f"""select * from [dbo].[vwM_district_max_n_bal_6_class_apy_crop_summary_{ref}]
    """
    print(sql)
    df = load_map_data(db_client_input, sql)
    df['mean_n_balance_n_kg_ha_apy_crop'] = df['mean_n_balance_n_kg_ha_apy_crop'].replace(apy_crop_replacements)
    counts = df[df['mean_n_balance_n_kg_ha_apy_crop'] != 'Other Crops']['mean_n_balance_n_kg_ha_apy_crop'].value_counts()
    ordered_categories = list(counts.index) + ['Other Crops'] 
    df['mean_n_balance_n_kg_ha_apy_crop'] = pd.Categorical(df['mean_n_balance_n_kg_ha_apy_crop'], categories=ordered_categories, ordered=True)

    df['total_n_balance_n_kg_ha_apy_crop'] = df['total_n_balance_n_kg_ha_apy_crop'].replace(apy_crop_replacements)
    counts = df[df['total_n_balance_n_kg_ha_apy_crop'] != 'Other Crops']['total_n_balance_n_kg_ha_apy_crop'].value_counts()
    ordered_categories = list(counts.index) + ['Other Crops'] 
    df['total_n_balance_n_kg_ha_apy_crop'] = pd.Categorical(df['total_n_balance_n_kg_ha_apy_crop'], categories=ordered_categories, ordered=True)


    dfs.append((ref, df))


# %%
for ref,  df in dfs:

    g_ha = (ggplot(df)
       + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="mean_n_balance_n_kg_ha_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=f"Crop with largest Nitrogen Balance\nper Hectare ({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
     
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

    print(g_ha)
    g_ha.save(filename=f"map_max_n_bal_6_class_apy_kg_ha_mean_{ref}.png", path=chart_dir,  units='cm', dpi=300)

    g_total = (ggplot(df)
       + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="total_n_balance_n_kg_ha_apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=f"Crop with largest Total Nitrogen Balance\n({(ref[0].upper() + ref.replace('_',' et al. ')[1:]).replace("Ipcc","IPCC")})")
     
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

    print(g_total)
    g_total.save(filename=f"map_max_n_bal_6_class_apy_total_mean_{ref}.png", path=chart_dir,  units='cm', dpi=300)
# %%
