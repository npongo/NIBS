#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import sys
sys.path.extend(["..\\"])
from Shared.NIBSData2 import *
from plotnine import ggplot, aes,  labs
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"b:\Repos\NIBS\Data" ,
                chart_dir=r"b:\Repos\NIBS\Graphs")


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
units = "Agricultural\nland <2 Ha (%)\n"
farm_size_area_g = (ggplot(farm_size_area_df)
    + geom_map(india, fill='grey', color="black", show_legend=False)
    + scale_x_continuous(limits=(67.5,97.5))
    + scale_y_continuous(limits=(7.5,37.5))
    + coord_cartesian()
    + theme_void()
    + geom_map(aes(fill="percent"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
    + labs(title="\nAgricutlure Land in Farms <2Ha")
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
#emission graph 
# eagle non-rice 
# bhatia rice 
# niko ch4
# biomass v2.0 burning 
# ha and t 

g_ha_n2o_none_rice = (upland_graphs['eagle_2020'][1] + labs(title='\n     A')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
g_ha_n2o_rice = (rice_graphs['bhatia_2013'][1] + labs(title='\n     B')  
                + theme( plot_title= element_text(ha='left', size=32))
                )
g_ha_ch4 = (ha_nikolaisen_2023 + labs(title='\n     C')  
                + theme( plot_title= element_text(ha='left', size=32))
                )
g_ha_burn = (biomass_ha_g + labs(title='\n     D')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )

g_t_n2o_none_rice = (upland_graphs['eagle_2020'][0] + labs(title='     E')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )
g_t_n2o_rice = (rice_graphs['bhatia_2013'][0] + labs(title='     F')  
                + theme( plot_title= element_text(ha='left', size=32))
                )
g_t_ch4 = (t_nikolaisen_2023 + labs(title='     G')  
                + theme( plot_title= element_text(ha='left', size=32))
                )
g_t_burn = (biomass_Gg_g + labs(title='     H')  
                + theme( plot_title= element_text(ha='left', size=32))
                        )

g = ((g_ha_n2o_none_rice | g_ha_n2o_rice | g_ha_ch4 | g_ha_burn)/(g_t_n2o_none_rice | g_t_n2o_rice | g_t_ch4 | g_t_burn)
     
     + theme(figure_size=(28, 16)))
g.save(path.join(nibs.chart_dir, 'Figure2.png'), dpi=nibs.dpi)
g
#%%
# Figure 3
# eagle 2020 ha and total crops with highest emissions 
t_g = (crops_graphs['eagle_2020'][0]+ labs(title='\n     B')  
                + theme( plot_title= element_text(ha='left', size=32))
                )
ha_g = (crops_graphs['eagle_2020'][1]+ labs(title='\n     A')  
                + theme( plot_title= element_text(ha='left', size=32))
                )
g = ha_g | t_g + theme(figure_size=(14,8))
g.save(path.join(nibs.chart_dir, 'Figure3.png'), dpi=nibs.dpi)
g

# %%
# Figure 4
# land size all crops and rice, eagle, nik
g_farm_size_ha_n2o = (farm_size_graphs['eagle_2020'][1]
                       + labs(title='\n     A')   
                       + theme( plot_title= element_text(ha='left', size=32))
                        )
g_farm_size_t_n2o = (farm_size_graphs['eagle_2020'][0]
                     + labs(title='     B')  
                     + theme( plot_title= element_text(ha='left', size=32))
                       )
g_farm_size_ha_ch4 = (g_dist_max_ha_farm_size_nikolaisen
                       + labs(title='     C')   
                       + theme( plot_title= element_text(ha='left', size=32))
                        )
g_farm_size_t_ch4 = (g_dist_max_t_farm_size_nikolaisen
                       + labs(title='     D')   
                       + theme( plot_title= element_text(ha='left', size=32))
                        )

g = (g_farm_size_ha_n2o | g_farm_size_ha_ch4)/(g_farm_size_t_n2o | g_farm_size_t_ch4) + theme(figure_size=(14, 16))
g.save(path.join(nibs.chart_dir, 'Figure4.png'), dpi=nibs.dpi)
g


# %%
# Figure 5
# 
landhold_sql = f"""select * 
from read_parquet('{path.join(nibs.data_dir,'crop_proportion_by_farm_size.parquet')}')
where crop in('Bajra','cotton(lint)','gram','groundnut','jowar','maize','Rapeseed &mustard','rice','soyabean','wheat')"""
landhold_df = nibs.load_table_data(landhold_sql)


landhold_df.columns = [c if c in ["geog", "geometry"] else nibs.format_name(c).replace("Organic\nfertilizer", "Crop").replace(
    "organic\nfertilizer", "crop") for c in landhold_df.columns]

landhold_df = nibs.farm_size_to_catagorical(landhold_df, ['Landhold'])
landhold_df['Crop'] = landhold_df['Crop'].replace(nibs.apy_crop_replacements)
landhold_df.loc[landhold_df.Crop=="Rapeseed & Mustard",'Crop'] = 'Rapeseed\nand Mustard'


#%%
l = (ggplot(landhold_df,aes(x="Landhold",y='Crop area ha', fill='Crop'))
     + geom_col(position="fill")
     + theme(figure_size=(8,6),axis_text_x=element_text(rotation=0, hjust=.5),rect=element_rect(color=(0,0,0,0),fill=(0,0,0,0))) # panel_background=element_rect(fill="red"))
     + labs(title="Crop area by Size of Landholding", y="Proportion",x="Landholding Size")
     + scale_y_continuous(name="Proportion")
     + scale_fill_brewer(type="div", palette ="RdYlGn") 
    + theme(figure_size=(8, 8),
                title=element_text(size=22), 
                axis_text_x=element_text(rotation=45, size=18, hjust='center'),
                axis_text_y=element_text(rotation=0, size=18, hjust='center'),
                axis_title_x=element_text(size=22),
                axis_title_y=element_text(size=22),
                legend_text=element_text(size=18)
    )
)
l.save(filename="CropAreaProportionByLandhold.png", path=nibs.chart_dir, width=12, height=12, units='cm', dpi=300)
l

# %%
fs = (l + labs(title='\nB')   
                       + theme( plot_title= element_text(ha='left', size=32))
                        )
fert = (fert_type_by_farm_size_g + labs(title='\nA')   
                       + theme( plot_title= element_text(ha='left', size=32))
                        )

g = (fert | fs) + theme(figure_size=(16, 10))
g.save(path.join(nibs.chart_dir, 'Figure5.png'), dpi=nibs.dpi)
g
# %%
# figure 1 
# national ghg 
national_facet_g.save(path.join(nibs.chart_dir, 'Figure1.png'), dpi=nibs.dpi)
national_facet_g
# %%
