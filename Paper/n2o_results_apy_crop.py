
#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import sys
sys.path.extend(["..\\"])
from Shared.NIBSData2 import *
from plotnine import labs
#import patchworklib as pw
from os import path
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"B:\Repos\NIBS\Data" , chart_dir=r"B:\Repos\NIBS\Graphs")

#%%
dfs = []
refs = {'shcherbak_2014': 'Shcherbak et al., 2014'
        , 'ipcc_2019': 'IPCC 2019 Updated Methodology'
        , 'eagle_2020': 'Eagle et al., 2020'
        , 'bhatia_2013': 'Bhatia et al., 2012'
        }
for ref in refs.keys():
    sql = f"""
    select * exclude geog, cast(geog as string) as geog
    from read_parquet('{path.join(nibs.data_dir,f'vwM_district_max_n2o_6_class_apy_crop_summary_{ref}.parquet')}') 
    """
    df = nibs.load_map_data(sql)
    nibs.apy_crop_6_class_to_catagorical(df, ['max_fert_induced_Tg_n2o_n_apy_crop','max_fert_induced_kg_n2o_n_ha_apy_crop'] )
    dfs.append((ref, df))

#%%
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
crops_graphs = {}
for (ref, df) in dfs:
    title = f'Crop with largest \ntotal $N_2O$ emissions\n({refs[ref]})'
    file_name = f"map_district_n2o_apy_crop_max_t_{ref}.svg"
    g_t = nibs.manual_catagorical_map(df
                        , 'max_fert_induced_Tg_n2o_n_apy_crop'
                        ,  title
                        , 'Crop'
                        , crop_colors
                        , file_name= file_name
                        , legend_position=(.7,.1)
                        ,dpi=nibs.dpi)
    g_t.show()

    title = f'Crop with largest $N_2O$\nemissions per hectare\n({refs[ref]})'
    file_name = f"map_district_n2o_apy_crop_max_ha_{ref}.svg"
    g_ha = nibs.manual_catagorical_map(df
                        , 'max_fert_induced_kg_n2o_n_ha_apy_crop'
                        ,  title
                        , 'Crop'
                        , crop_colors
                        , file_name= file_name
                        , legend_position=(.7,.1))
    g_ha.show()
    crops_graphs[ref] = (g_t, g_ha)

# %%
gA = crops_graphs['bhatia_2013'][1] + labs(title='\n     A')  + theme( plot_title= element_text(ha='left', size=32))

gB = crops_graphs['ipcc_2019'][1] + labs(title='\n     B') + theme( plot_title= element_text(ha='left', size=32))

gC = crops_graphs['eagle_2020'][1] + labs(title='\n     C') + theme( plot_title= element_text(ha='left', size=32))

gD = crops_graphs['shcherbak_2014'][1] + labs(title='\n     D') + theme( plot_title= element_text(ha='left', size=32))

gE = crops_graphs['bhatia_2013'][0] + labs(title='     E') + theme( plot_title= element_text(ha='left', size=32))

gF = crops_graphs['ipcc_2019'][0] + labs(title='     F') + theme( plot_title= element_text(ha='left', size=32))

gG = crops_graphs['eagle_2020'][0] + labs(title='     G') + theme( plot_title= element_text(ha='left', size=32))

gH = crops_graphs['shcherbak_2014'][0] + labs(title='     H') + theme( plot_title= element_text(ha='left', size=32))


#%%
g = (gA| gB| gC| gD)/(gE| gF| gG| gH) + theme(figure_size=(32, 16))
g.save(path.join(nibs.chart_dir,"map_district_n2o_apy_crop_max_plate.svg"), dpi=nibs.dpi)
g


# %%
rename_ha_dict = {
    'max_total_kg_n2o_n_ha_apy_crop_bhatia_2013': 'Bhatia\net al., 2013\n',
    'max_total_kg_n2o_n_ha_apy_crop_eagle_2020': 'Eagle\net al.,2020\n',
    'max_total_kg_n2o_n_ha_apy_crop_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n',
    'max_total_kg_n2o_n_ha_apy_crop_shcherbak_2014': 'Shcherbak\net al.,2014\n',
}

rename_total_dict = {
    'max_total_Tg_n2o_n_apy_crop_bhatia_2013': 'Bhatia\net al., 2013\n',
    'max_total_Tg_n2o_n_apy_crop_eagle_2020': 'Eagle\net al., 2020\n',
    'max_total_Tg_n2o_n_apy_crop_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n',
    'max_total_Tg_n2o_n_apy_crop_shcherbak_2014': 'Shcherbak\net al.,2014\n'
}

cramer_v_sql = f"""select * exclude geog, cast(geog as string) as geog
               from read_parquet('{path.join(nibs.data_dir,'vwM_district_max_n2o_apy_crop_summary_all_models.parquet')}') """
cramer_v_df = nibs.load_map_data(cramer_v_sql)

#%%

crop_ha_cramer_v = nibs.cramers_v_plot(cramer_v_df
                    , "apy_crop_kg_ha_cramers_v_matrix.svg"
                    , rename_ha_dict
                    , 'District crops with\nhighest $N_2O$ emission\n($Kg\ N_2O\ Ha^{-1}$ Cramér\'s V)'
                    , figure_size=(7,6)
                    ,legend_title="Cramér\'s V"
                    )
crop_ha_cramer_v.show()

crop_t_cramer_v = nibs.cramers_v_plot(cramer_v_df
                    , "apy_crop_t_n2o_cramers_v_matrix.svg"
                    , rename_total_dict
                    , 'District crops with\nhighest $N_2O$ emission\n ($Total\ Gg\ N_2O$ Cramér\'s V)'
                    , figure_size=(7,6)
                    , legend_title="Cramér\'s V"
                    )
crop_t_cramer_v.show()                                                               
# %%
