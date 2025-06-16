
#%%
%reload_ext autoreload
%autoreload 2

get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.extend(["..\\"])
import pandas as pd
from Shared.NIBSData2 import *
from plotnine import ggplot, aes, labs, scale_fill_manual
import patchworklib as pw
from os import path
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
print('Python %s on %s' % (sys.version, sys.platform))

nibs = NIBSData(data_dir=r"D:\Repos\NIBS\Data" , chart_dir=r"D:\Repos\NIBS\Graphs")

refs = {'shcherbak_2014': 'Shcherbak et al., 2014'
        , 'ipcc_2019': 'IPCC 2019 Updated Methodology'
        , 'eagle_2020': 'Eagle et al., 2020'
        , 'bhatia_2013': 'Bhatia et al., 2012'
        }

# %%
dfs = []
for ref in refs.keys():
    sql = f"""
    select * exclude geog, cast(geog as string) as geog
    from read_parquet('{path.join(nibs.data_dir,f'vwM_district_max_n2o_farm_size_summary_{ref}.parquet')}')
    """
    df = nibs.load_map_data(sql)
    df = nibs.farm_size_to_catagorical(df, ['max_fert_induced_Tg_n2o_n_farm_size','max_fert_induced_kg_n2o_n_ha_farm_size'])
    dfs.append((ref, df))

#%%
graphs = {}
for ref, df in dfs: 
    file_name = f'map_district_n2o_farm_size_max_t_{ref}.png'
    title = f'All crops, Landholding size with\nlargest total $N_2O$ emissions\n({refs[ref]})'
    g_max_t = nibs.catagorical_map(df, 'max_fert_induced_Tg_n2o_n_farm_size'
                        , title
                        , 'Landholding\nSize\n'
                        , file_name
                        , legend_position=(.65,.12))

    file_name = f'map_district_n2o_farm_size_max_kg_ha_{ref}.png'
    title = f'All crops, Landholding size\nwith largest $N_2O$ emissions per hectare\n({refs[ref]})'
    g_max_ha = nibs.catagorical_map(df, 'max_fert_induced_kg_n2o_n_ha_farm_size'
                        ,'All crops, Landholding size\nwith largest $N_2O$ emissions per hectare\n(IPCC 2019 Updated Methodology)'
                        , 'Landholding\nSize\n'
                        , 'map_district_n2o_farm_size_max_t_ipcc_2019.png'
                        , legend_position=(.65,.12))

    graphs[ref] = (g_max_t, g_max_ha)
    g_max_ha.show()
    g_max_t.show()


# %%

gA = pw.load_ggplot(graphs['bhatia_2013'][1] + labs(title='A')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )
gB = pw.load_ggplot(graphs['eagle_2020'][1] + labs(title='B')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )
gC = pw.load_ggplot(graphs['ipcc_2019'][1] + labs(title='C')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )
gD = pw.load_ggplot(graphs['shcherbak_2014'][1] + labs(title='D')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )

gE = pw.load_ggplot(graphs['bhatia_2013'][0] + labs(title='E')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )
gF = pw.load_ggplot(graphs['eagle_2020'][0] + labs(title='F')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )
gG = pw.load_ggplot(graphs['ipcc_2019'][0] + labs(title='G')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )
gH = pw.load_ggplot(graphs['shcherbak_2014'][0] + labs(title='H')   
                + theme(plot_title=element_text(ha='left', size=32) 
                        , legend_text_colorbar=element_text(size=18)
                        , legend_title=element_text(size=22)
                        , legend_position=(.7,.16)
                        , legend_key_height=100
                        , legend_key_width=25)
                        )

g = (gA| gB|gC| gD)/(gE| gF| gG| gH)
g.savefig(path.join(nibs.chart_dir,"map_district_n2o_apy_crop_max_plate.png"), dpi=nibs.dpi)
g

# %%
cramer_v_sql = f"""select *  exclude geog, cast(geog as string) as geog
from read_parquet('{path.join(nibs.data_dir,'vwM_district_max_n2o_farm_size_summary_all_models.parquet')}')"""
cramer_v_df = nibs.load_map_data(cramer_v_sql)

# %%

rename_ha_dict = {
    'max_total_kg_n2o_n_ha_farm_size_bhatia_2013': 'Bhatia\net al., 2013\n',
    'max_total_kg_n2o_n_ha_farm_size_eagle_2020': 'Eagle\net al.,2020\n',
    'max_total_kg_n2o_n_ha_farm_size_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n',
    'max_total_kg_n2o_n_ha_farm_size_shcherbak_2014': 'Shcherbak\net al.,2014\n',
}
ha_cramers_v = nibs.cramers_v_plot(cramer_v_df, "farm_size_kg_ha_cramers_v_matrix.png"
                    , rename_ha_dict
                    , 'District farm size with\nhighest $N_2O$ emission\n($Kg\ N_2O\ Ha^{-1}$ Cramér\'s V)'
                    , legend_title="Cramér\'s V"
                    ,figure_size=(7,6)
                    )
ha_cramers_v.show()

rename_total_dict = {
    'max_total_Tg_n2o_n_farm_size_bhatia_2013': 'Bhatia\net al., 2013\n',
    'max_total_Tg_n2o_n_farm_size_eagle_2020': 'Eagle\net al., 2020\n',
    'max_total_Tg_n2o_n_farm_size_ipcc_2019': 'IPCC 2019\nUpdated\nMethodology\n\n',
    'max_total_Tg_n2o_n_farm_size_shcherbak_2014': 'Shcherbak\net al.,2014\n'
}

t_cramers_v = nibs.cramers_v_plot(cramer_v_df, "farm_size_t_n2o_cramers_v_matrix.png"
                    , rename_total_dict
                    , 'District farm size with\nhighest $N_2O$ emission\n ($Total\ Gg\ N_2O$ Cramér\'s V)'
                    , legend_title="Cramér\'s V"
                    ,figure_size=(7,6)
                    )
t_cramers_v.show()

# %%

fert_farm_size_sql = f"""select *
from read_parquet('{path.join(nibs.data_dir,'vwG_inorganic_organic_fert_by_farm_size.parquet')}') 
"""
fert_farm_size_df = nibs.load_table_data(fert_farm_size_sql)
fert_farm_size_df = nibs.farm_size_to_catagorical(fert_farm_size_df, ['farm_size'])
fert_farm_size_df['fert_type'] = fert_farm_size_df['fert_type'].replace({'inorganic': 'Inorganic', 'organic': 'Organic'})
fert_farm_size_df['fert_type'] = pd.Categorical(fert_farm_size_df['fert_type'], categories=['Inorganic', 'Organic'], ordered=True)  
fert_farm_size_df.head()

# %%
g = (ggplot(fert_farm_size_df)
        + geom_bar(aes(x='farm_size', y='mean_n_rate_kg_ha', fill='fert_type'), stat='identity', position='dodge')
        + labs(title='Fertilizer application by farm size', x='Farm Size', y='Fertilizer application ($Kg\ N\ Ha^{-1}$)')
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
        + scale_fill_manual(values=['#F46D43', '#1A9850'], name="Fertilizer")
        + theme_minimal()
        + theme(figure_size=(8, 8),
                title=element_text(size=22), 
                axis_text_x=element_text(rotation=45, size=18, hjust='center'),
                axis_text_y=element_text(rotation=0, size=18, hjust='center'),
                axis_title_x=element_text(size=22),
                axis_title_y=element_text(size=22),
                legend_text=element_text(size=18)
        )
    )
g.show()
g.save(path.join(nibs.chart_dir, " fert_type_by_farm_size.png"), dpi=nibs.dpi)


# %%
fert_farm_size_sql = f"""select *
from read_parquet('{path.join(nibs.data_dir,'vwG_rice_inorganic_organic_fert_by_farm_size.parquet')}') 
"""
fert_farm_size_df = nibs.load_table_data(fert_farm_size_sql)
fert_farm_size_df['fert_type'] = fert_farm_size_df['fert_type'].replace({'inorganic': 'Inorganic', 'organic': 'Organic'})
fert_farm_size_df = nibs.farm_size_to_catagorical(fert_farm_size_df, ['farm_size'])
fert_farm_size_df['fert_type'] = pd.Categorical(fert_farm_size_df['fert_type'], categories=['Inorganic', 'Organic'], ordered=True)  


# %%
g = (ggplot(fert_farm_size_df)
        + geom_bar(aes(x='farm_size', y='mean_n_rate_kg_ha', fill='fert_type'), stat='identity', position='dodge')
        + labs(title='Rice fertilizer application by farm size', x='Farm Size', y='Fertilizer application ($Kg\ N\ Ha^{-1}$)')
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
        + scale_fill_manual(values=['#F46D43', '#1A9850'], name="Fertilizer")
        + theme_minimal()
        + theme(figure_size=(8, 8),
                title=element_text(size=22), 
                axis_text_x=element_text(rotation=45, size=18, hjust='center'),
                axis_text_y=element_text(rotation=0, size=18, hjust='center'),
                axis_title_x=element_text(size=22),
                axis_title_y=element_text(size=22),
                legend_text=element_text(size=18)
        )
    )
g.show()
g.save(path.join(nibs.chart_dir, "rice_fert_type_by_farm_size.png"), dpi=nibs.dpi)

# %%
fert_farm_size_sql = f"""select *
from read_parquet('{path.join(nibs.data_dir,'vwG_upland_crop_inorganic_organic_fert_by_farm_size.parquet')}') 
"""
fert_farm_size_df = nibs.load_table_data(fert_farm_size_sql)
fert_farm_size_df['fert_type'] = fert_farm_size_df['fert_type'].replace({'inorganic': 'Inorganic', 'organic': 'Organic'})
fert_farm_size_df['fert_type'] = pd.Categorical(fert_farm_size_df['fert_type'], categories=['Inorganic', 'Organic'], ordered=True) 
fert_farm_size_df = nibs.farm_size_to_catagorical(fert_farm_size_df, ['farm_size']) 

# %%
g = (ggplot(fert_farm_size_df)
        + geom_bar(aes(x='farm_size', y='mean_n_rate_kg_ha', fill='fert_type'), stat='identity', position='dodge')
        + labs(title='None-rice crop fertilizer application by farm size', x='Farm Size', y='Fertilizer application ($Kg\ N\ Ha^{-1}$)')
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
        + scale_fill_manual(values=['#F46D43', '#1A9850'], name="Fertilizer")
        + theme_minimal()
        + theme(figure_size=(8, 8),
                title=element_text(size=22), 
                axis_text_x=element_text(rotation=45, size=18, hjust='center'),
                axis_text_y=element_text(rotation=0, size=18, hjust='center'),
                axis_title_x=element_text(size=22),
                axis_title_y=element_text(size=22),
                legend_text=element_text(size=18)
        )
    )
g.show()
g.save(path.join(nibs.chart_dir, "none_rice_crop_fert_type_by_farm_size.png"), dpi=nibs.dpi)

