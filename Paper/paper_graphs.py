

#%%
%reload_ext autoreload
%autoreload 2

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.extend(["..\\"])
from Shared.NIBSData2 import *
from plotnine import ggplot, aes,  labs
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"D:\Repos\NIBS\Data" , chart_dir=r"D:\Repos\NIBS\Graphs")


#%%
farm_size_area_sql = f"""select * exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_farmsize_area_proportion.parquet')}') 
where farm_size = '<2ha'
"""
farm_size_area_df = nibs.load_map_data(farm_size_area_sql)
farm_size_area_df['percent'] = farm_size_area_df['prop']*100

#%% 
india = nibs.get_india()
india_states = nibs.get_india_states()

      
#%%
units = "Agricuultural\nland <2 Ha (%)\n"
farm_size_area_g = (ggplot(farm_size_area_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
    + scale_x_continuous(limits=(67.5,97.5))
    + scale_y_continuous(limits=(7.5,37.5))
    + coord_cartesian()
    + theme_void()
    + geom_map(aes(fill="percent"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
    + labs(title="Agricutlure Land in Farms <2Ha")
    + scale_fill_gradientn(colors=nibs.colors, name=units)
    + theme(
        figure_size=(7,8),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
         , legend_direction='vertical'
         , legend_position=(.75,.08)
     )
)
farm_size_area_g.save(filename="map_farm_size_proportion_ag_area.png", path=nibs.chart_dir,  units='cm', dpi=nibs.dpi)
farm_size_area_g
# %%
