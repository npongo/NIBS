#%%
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


chart_dir = r"E:\npongo Dropbox\benjamin clark\CIL\Products\EDF20240913"
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



# db_conn = {'server': '.\\npongo22', 'database': 'india_cost_of_cultivation_ghg_results_v1'}
# db_client = mssql.SqlServerClient(db_conn) 

db_conn_input = {'server': '.\\npongo22', 'database': 'india_agriculture_census_ghg_results_v1'}
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
bur_sql = """
    select * from india_bur_3_emission_data
    where sector not in('Enteric Fermentation','Manure Management')
"""
bur_df = load_table_data(db_client_input, bur_sql)
print(list(bur_df['sector']))
print({i:i for i in bur_df['sector']})  

#%%
bur_df['sector'] = pd.Categorical(bur_df['sector']
                                    , categories=[ 'Field Burning', 'Total N2O Emissions','Rice Cultivation'], ordered=True)
bur_df['sector'] = bur_df['sector'].cat.rename_categories({'Rice Cultivation': 'Rice\nCultivation', 'Total N2O Emissions': 'Total N2O\nEmissions', 'Field Burning': 'Field\nBurning'})

bur_df.head()
#%%
p = (ggplot(bur_df)
            # + geom_violin(filter_df,aes(x='dataset', y='kg_inorganic_n_ha'),draw_quantiles=[0.25, 0.5, 0.75])
            + geom_col(aes(x='sector', y='mt_co2e_100',fill='sector'))
            + geom_text(aes(x='sector', y='mt_co2e_100', label='mt_co2e_100'), 
                        va='bottom', ha='center', size=8, format_string='{:.2f}')
    
            # + geom_jitter(filter_df, aes(x='dataset', y='kg_inorganic_n_ha'), color='blue', alpha=.01, size=.001, width=0.3)
             + labs(title='2016 India BUR UNFCCC Emissions', x='Emission Type', y="$CO_2e_{100}\ Mt\ Year^{-1}$")
            # + scale_y_log10()
            + scale_y_continuous(expand=[0,0], limits=[0, 100])
            + guides(fill=None) 
             + theme(axis_text_x=element_text(angle=0, va="top", ha="center", size=8), plot_title=element_text(ha='center', size=14))
       
            )
print(p)
p.save(filename=f"india_bur_3_emissions.png", path=chart_dir,height=7, width=12, units='cm', dpi=92)
# %%

# %%
national_sql = """
select *
from vwG_national_cropping_ghg
where parameter not in('mean_total_eagle_2020_n2o_co2e_Mt','mean_total_co2e_Mt')
"""
national_df  = load_table_data(db_client_input, national_sql)
#national_df = national_df.sort_values(by=['gwp_time_period', 'mean_co2e_Mt'], ascending=[True, True])
national_df['label_simple'] = pd.Categorical(national_df['label_simple']
                                 , categories=[ 'Rice $N_2O$',
                                            'Residue Burning',
                                            'Upland Crops $N_2O$', 
                                            'Rice $CH_4$']
                                            , ordered=True)

national_df['gwp_time_period'] = pd.Categorical(national_df['gwp_time_period']
                                 , categories=[ 20,
                                            100]
, ordered=True)
gwp_label={20:'$GWP_{20}$', 100:'$GWP_{100}$'}
national_df['gwp_label'] = national_df['gwp_time_period'].map(gwp_label)

print({i:i for i in national_df['label_simple'].unique()})  

#%%
national_df['label_simple'] = national_df['label_simple'].cat.rename_categories({'Upland Crops $N_2O$': 'Upland\nCrops\n$N_2O$',
                                                                                 'Rice $N_2O$': 'Rice\n$N_2O$',
                                                                                  'Residue Burning': 'Residue\nBurning',
                                                                                   'Rice $CH_4$': 'Rice\n$CH_4$'}
)

national_df.head()

#%%
national_df['gwp_time_period'].unique()

#%%
p = (ggplot(national_df)
            # + geom_violin(filter_df,aes(x='dataset', y='kg_inorganic_n_ha'),draw_quantiles=[0.25, 0.5, 0.75])
            + geom_col(aes(x='label_simple', y='mean_co2e_Mt',fill='label_simple'))
            + geom_errorbar(aes(x='label_simple', ymin='mean_co2e_Mt - sd_co2e_Mt', ymax='mean_co2e_Mt + sd_co2e_Mt'), width=0.2)
          + geom_text(aes(x='label_simple', y='mean_co2e_Mt', label='mean_co2e_Mt'), 
                 va='bottom', ha='center', size=8, format_string='{:.2f}')
    
            # + geom_jitter(filter_df, aes(x='dataset', y='kg_inorganic_n_ha'), color='blue', alpha=.01, size=.001, width=0.3)
             + labs(title='2016-17 Cropping GHG Emissions', x="Emission Type", y="$CO_2e\ Mt\ Year^{-1}$")
            # + scale_y_log10()
             +scale_y_continuous(limits=(0, 375))
            + guides(fill=None) 
             + theme(axis_text_x=element_text(angle=0, va="top", ha="center", size=10),
             plot_title=element_text(ha='center', size=14))
            + facet_wrap('~gwp_label')
            )
print(p)
p.save(filename=f"national_ghg_15x18.png", path=chart_dir,height=12, width=15, units='cm', dpi=92)
g = p + facet_wrap('~gwp_label', scales='free_y')
print(g)
g.save(filename=f"national_ghg_15x18_free_y.png", path=chart_dir,height=12, width=15, units='cm', dpi=92)


# %%
district_sql = """select district_name, geog.STAsBinary() as geog from district_boundaries"""
district_df = load_map_data(db_client_input, district_sql)
district_df.head()

#%%
p =  plot_map(None, district_df, None, 'base map', '')
p

# %%

# %%
district_climate_sql = """select  geog.STAsBinary() as geog from district_climate_temp"""
district_climate_df = load_map_data(db_client_input, district_climate_sql)
district_climate_df.head()

#%%
p =  plot_map(None, district_climate_df, None, 'base map', '')
p

# %%
rice_wheat_sql = """select  * from vwR_rice_wheat_n_balance"""
rice_wheat_df = load_map_data(db_client_input, rice_wheat_sql)
rice_wheat_df.head()

#%%
p = (ggplot(rice_wheat_df)
            # + geom_violin(filter_df,aes(x='dataset', y='kg_inorganic_n_ha'),draw_quantiles=[0.25, 0.5, 0.75])
            + geom_point(aes(x='mean_total_n_balance_n_kg_ha', y='mean_yield_t_ha',fill='apy_crop'))
        #     + geom_errorbar(aes(x='label_simple', ymin='mean_co2e_Mt - sd_co2e_Mt', ymax='mean_co2e_Mt + sd_co2e_Mt'), width=0.2)
        #   + geom_text(aes(x='label_simple', y='mean_co2e_Mt', label='mean_co2e_Mt'), 
        #          va='bottom', ha='center', size=8, format_string='{:.2f}')
    
            + scale_fill_discrete(name='Crop')
            # + geom_jitter(filter_df, aes(x='dataset', y='kg_inorganic_n_ha'), color='blue', alpha=.01, size=.001, width=0.3)
             + labs(title='Rice-wheat rotation districts', x="N-balance $Kg\ Ha^{-1}$", y="$Yield\ Ton\ Ha^{-1}$")
            # + scale_y_log10()
            #  +scale_y_continuous(limits=(0, 375))
            #+ guides(fill=None) 
             + theme(axis_text_x=element_text(angle=0, va="top", ha="center", size=10),
             plot_title=element_text(ha='center', size=14))
            # + facet_wrap('~gwp_label')
            )
print(p)
p.save(filename=f"rice_wheat_n_balance_target_full.png", path=chart_dir,height=12, width=22, units='cm', dpi=92)


#%%
filter_df = rice_wheat_df[(rice_wheat_df['mean_total_n_balance_n_kg_ha'] > 200) & (rice_wheat_df['mean_yield_t_ha'] < 2)]


grouped_df = filter_df.groupby('rotation_id').size().reset_index(name='counts')
line_rid = grouped_df[grouped_df['counts'] > 1]['rotation_id']

print(line_rid)
print(len(line_rid))
line_df = filter_df[filter_df['rotation_id'].isin(line_rid)] 
line_df['mean_total_n_balance_n_kg_kg_yield'] =line_df['mean_total_n_balance_n_kg_ha']/line_df['mean_yield_t_ha'] 


line_cnt_df = line_df.groupby('label').size().reset_index(name='counts')
print(line_cnt_df)


#%%
p = (ggplot(filter_df)
            # + geom_violin(filter_df,aes(x='dataset', y='kg_inorganic_n_ha'),draw_quantiles=[0.25, 0.5, 0.75])
            + geom_point(aes(x='mean_total_n_balance_n_kg_ha', y='mean_yield_t_ha',fill='apy_crop'))
        #     + geom_errorbar(aes(x='label_simple', ymin='mean_co2e_Mt - sd_co2e_Mt', ymax='mean_co2e_Mt + sd_co2e_Mt'), width=0.2)
        #   + geom_text(aes(x='label_simple', y='mean_co2e_Mt', label='mean_co2e_Mt'), 
        #          va='bottom', ha='center', size=8, format_string='{:.2f}')
    
            # + geom_jitter(filter_df, aes(x='dataset', y='kg_inorganic_n_ha'), color='blue', alpha=.01, size=.001, width=0.3)
             + labs(title='Rice-wheat rotation districts', x="N-balance $Kg\ Ha^{-1}$", y="$Yield\ Ton\ Ha^{-1}$")
            # + scale_y_log10()
            + scale_fill_discrete(name='Crop')
            + scale_y_continuous(limits=(0, 2))
            + scale_x_continuous(limits=(200, 600))
            + geom_line(line_df, aes(x='mean_total_n_balance_n_kg_ha', y='mean_yield_t_ha', group='rotation_id', color='label'))
    
            + scale_color_discrete(name='District')
            #  +scale_y_continuous(limits=(0, 375))
            #+ guides(fill=None) 
             + theme(figure_size=(8,8),axis_text_x=element_text(angle=0, va="top", ha="center", size=10),
             plot_title=element_text(ha='center', size=14))
            # + facet_wrap('~gwp_label')
            )
print(p)
p.save(filename=f"rice_wheat_n_balance_target_zoom.png", path=chart_dir, units='cm', dpi=92)

# %%,height=12, width=12
units = "Yield($ton Ha^{-1}$) scaled\nN-balance $Kg\ Ha^{-1}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot()
     + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(line_df, aes(fill="mean_total_n_balance_n_kg_kg_yield"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
             + labs(title='Rice-wheat rotation districts')
          
     + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=200, high=high_color, name=units, limits=[100,300])
     #+ facet_wrap('~farm_size')
     + theme(
         figure_size=(10,10),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=14)
     )
)

g.save(filename="map_rice_wheat_targets.png", path=chart_dir,  units='cm', dpi=300)
g

# %%
target_df = line_df[(line_df['state_code'] == 'MH') | (line_df['state_code'] == 'MP') | (line_df['state_code'] == 'CG')]
target_df.loc[target_df['apy_crop'] == 'Rice', 'label'] = ''
target_df
target_df['farm_size'] = pd.Categorical(target_df['farm_size']
                                    , categories=[ 'MARGINAL (BELOW 1.0)', 'SMALL (1.0 - 1.99)', 'SEMI-MEDIUM (2.0 - 3.99)','MEDIUM (4.0 - 9.99)','LARGE (10 AND ABOVE)' ], ordered=True)
target_df['farm_size'] = target_df['farm_size'].cat.rename_categories({'MARGINAL (BELOW 1.0)': '<1.0 Ha',
                                                                        'SMALL (1.0 - 1.99)': '1.0 - 1.99 Ha',
                                                                        'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
                                                                        'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
                                                                        'LARGE (10 AND ABOVE)': '≥10 Ha'    })
target_df['irrigated'] = pd.Categorical(target_df['irrigated']
                                    , categories=['UI','I'], ordered=True)
target_df['irrigated'] = target_df['irrigated'].cat.rename_categories({'UI': 'Unirrigated', 'I': 'Irrigated'})
target_df


# %%
g = (ggplot(target_df,aes(x='mean_total_n_balance_n_kg_ha', y='mean_yield_t_ha'))
    + geom_line(aes(group='rotation_id', color='irrigated'))
    + geom_point(aes(fill='apy_crop', size='farm_size')) 
    + geom_text(aes( label='label', adjust_text=True), va='bottom', ha='left', size=8, format_string='{:}', nudge_x=0.02, nudge_y=0.02)
    + labs(title='Rice-wheat rotation districts', x="N-balance $Kg\ Ha^{-1}$", y="$Yield\ Ton\ Ha^{-1}$")
    
     + scale_color_manual(name='Irriagted',values={'Unirrigated': 'blue', 'Irrigated': 'green'})
     + scale_fill_manual(name='Crop',values={'Rice': 'red', 'Wheat': 'yellow'})
     + scale_size_discrete(name='Farm Size', range=[3,13])
     + scale_x_continuous(limits=(200, 400))
     + scale_y_continuous(limits=(0, 2.2))
     + theme(
         figure_size=(10, 6),
        # rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         #strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=14)
     )
)
g.save(filename="rice_wheat_targets_detail.png", path=chart_dir, units='cm', dpi=300)
g


# %%,  width=34, height=20
target_npk_sql = """

select link_id, district, geog_checksum, apy_crop, upper(substring(nutrient,4,1)) as nutrient, kg_ha 
from (
select  link_id, district, geog_checksum, apy_crop
	,shc_max_unit_corrected_kg_inorganic_kg_n_ha as kg_n_ha
	,shc_max_unit_corrected_kg_p_ha as kg_p_ha
	,shc_max_unit_corrected_kg_k_ha as kg_k_ha
from [india_agriculture_census_ghg_results_v1].dbo.fertilizer_input
where link_id in(<link_ids>)
) as src
unpivot(kg_ha for nutrient in(kg_n_ha,kg_p_ha,kg_k_ha)) as pvt"""

link_ids = ",".join([str(link_id) for link_id in target_df['link_id'].unique()])
target_npk_sql = target_npk_sql.replace('<link_ids>', link_ids)
target_npk_df = load_table_data(db_client_input, target_npk_sql)

target_npk_df['nutrient'] = pd.Categorical(target_npk_df['nutrient']
                                    , categories=['N','P','K'], ordered=True)

target_npk_df.head()
#print(target_npk_sql)


# %%
shc_npk_sql = """
select state_code, district, district_checksum, irrigated, apy_crop, substring(nutrient,4,1) as nutrient, kg_ha
from(
select state_code, district, district_checksum, irrigated, apy_crop, kg_n_ha, kg_p_ha, kg_k_ha 
from shc_crop_npk_requirements
where state_code in ('cg','mh','mp')
	and apy_crop in('rice','wheat') ) as src
unpivot(kg_ha for nutrient in( kg_n_ha, kg_p_ha, kg_k_ha )) as pvt
"""
shc_npk_df = load_table_data(db_client_input, shc_npk_sql)

shc_npk_df['nutrient'] = pd.Categorical(shc_npk_df['nutrient']
                                    , categories=['n','p','k'], ordered=True)
shc_npk_df['nutrient'] = shc_npk_df['nutrient'].cat.rename_categories({'n': 'N', 'p': 'P', 'k': 'K'})

shc_npk_df.head()

# %%
g = (ggplot()
    + geom_boxplot(shc_npk_df, aes(x='apy_crop', y='kg_ha'))
    + scale_fill_manual(name='Nutrient', values={'N': 'red', 'P': 'blue', 'K': 'green'})
    + geom_jitter(target_npk_df, aes(x='apy_crop', y='kg_ha', fill='district'), width=0.2, size=4)
    #+ geom_text(target_npk_df, aes(x='apy_crop', y='kg_ha'), label='label', position=position_jitter(width=1,height=1))
    + labs(title='Macro nutrient recommendations distributions compared with target districts', x='Crop', y='Nutrient (kg/ha)')
    + scale_fill_discrete(name='District') #, values={'N': 'red', 'P': 'blue', 'K': 'green'})
    + theme(
        figure_size=(10, 4),
        plot_title=element_text(ha='center', size=14)
    )
    +facet_wrap('~nutrient')
)

g.save(filename="macro_nutrient_distribution_by_crop.png", path=chart_dir, units='cm', dpi=300)
g

#%%
target_npk_df.head()


# %%
geog_checksums = ",".join([str(geog_checksum) for geog_checksum in target_df['geog_checksum'].unique()])

shc_micro_sql = """select Parameter, cast([Raigarh(CG)] as varchar(32)) as [Raigarh(CG)],cast([Nagpur(MH)] as varchar(32)) as [Nagpur(MH)],cast([Chandrapur(MH)] as varchar(32)) as [Chandrapur(MH)],cast([Amravati(MH)] as varchar(32)) as [Amravati(MH)],cast([Nashik(MH)] as varchar(32)) as [Nashik(MH)],cast([Balaghat(MP)] as varchar(32)) as [Balaghat(MP)]
from (
select replace(concat(replace(parameter_name,concat('(',mn.parameter,')'),''),' (',unit,')'),' ()','') as parameter
	,concat(district_name,'(',state_code,')') as district 
	,cast(concat(round(mean,1),' ±',round(sd,1),' (',class,')') as varbinary(128)) as v
	--,1 as v
from shc_district_micro_nutrient_summary mn
	join shc_parameter_units u on mn.parameter = u.parameter 
	join district_boundaries d on d.geog_checksum = mn.district_checksum 
	join state_codes sc on sc.state_name = d.state_name 
where district_checksum in(<geog_checksums>)
) as src
pivot(min(v) for district in([Raigarh(CG)],[Nagpur(MH)],[Chandrapur(MH)],[Amravati(MH)],[Nashik(MH)],[Balaghat(MP)])) as pvt
"""
shc_micro_sql = shc_micro_sql.replace('<geog_checksums>', geog_checksums)
shc_micro_df = load_table_data(db_client_input, shc_micro_sql)
shc_micro_df.head()
latex_output_micro = shc_micro_df.to_latex(index=False)
latex_output_micro = shc_micro_df.to_latex(index=False)
with open(f"{chart_dir}\shc_mean_micro_nutrient_table.tex", "w") as f:
    f.write(latex_output_micro)
latex_output_micro

# %%
geog_checksums = ",".join([str(geog_checksum) for geog_checksum in target_df['geog_checksum'].unique()])

shc_micro_sql = """select Parameter, cast([Raigarh(CG)] as varchar(32)) as [Raigarh(CG)],cast([Nagpur(MH)] as varchar(32)) as [Nagpur(MH)],cast([Chandrapur(MH)] as varchar(32)) as [Chandrapur(MH)],cast([Amravati(MH)] as varchar(32)) as [Amravati(MH)],cast([Nashik(MH)] as varchar(32)) as [Nashik(MH)],cast([Balaghat(MP)] as varchar(32)) as [Balaghat(MP)]
from (
select replace(concat(replace(parameter_name,concat('(',mn.parameter,')'),''),' (',unit,')'),' ()','') as parameter
	,concat(district_name,'(',state_code,')') as district 
	,cast(concat(round(p50,1),'(iqr ',round(p25,1),'-',round(p75,1),') ',p50_class) as varbinary(128)) as v
	--,1 as v
from shc_district_micro_nutrient_summary mn
	join shc_parameter_units u on mn.parameter = u.parameter 
	join district_boundaries d on d.geog_checksum = mn.district_checksum 
	join state_codes sc on sc.state_name = d.state_name 
where district_checksum in(<geog_checksums>)
) as src
pivot(min(v) for district in([Raigarh(CG)],[Nagpur(MH)],[Chandrapur(MH)],[Amravati(MH)],[Nashik(MH)],[Balaghat(MP)])) as pvt
"""
shc_micro_sql = shc_micro_sql.replace('<geog_checksums>', geog_checksums)
shc_micro_df = load_table_data(db_client_input, shc_micro_sql)
shc_micro_df.head()
latex_output_micro = shc_micro_df.to_latex(index=False)
latex_output_micro = shc_micro_df.to_latex(index=False)
with open(f"{chart_dir}\shc_median_micro_nutrient_table.tex", "w") as f:
    f.write(latex_output_micro)
latex_output_micro


#%%
apy_sql = """with a as
	(
	select concat(district_name,'(',state_code,')') as district 
		,apy_crop, year
		,area, production, yield	
		,season
		,row_number() over(partition by apy.state_name,district_name,apy_crop,year order by iif(season='total',0,1)) as rno
	from apy 
		join state_codes sc on sc.state_name = apy.state_name
	where geog_checksum in(357371350,-2066345052,931355363,1626958676,2076785716,2072171870)--,384966572,1310434990
		and apy_crop in('rice','wheat')
	)
,b as (
	select 'All India' as district 
		,apy_crop, year
		,area, production, yield	
		,season
		,row_number() over(partition by apy.state_name,district_name,apy_crop,year order by iif(season='total',0,1)) as rno
	from apy 
		join state_codes sc on sc.state_name = apy.state_name
	--where geog_checksum not in(357371350,-2066345052,931355363,1626958676,2076785716,2072171870)--,384966572,1310434990
		and apy_crop in('rice','wheat')
        and year <> '2019-20'   
    )
select district, apy_crop, year, area, production, yield
from a
where rno = 1
union
select district, apy_crop, year, area, production, yield
from b
where rno = 1
"""
apy_df = load_table_data(db_client_input, apy_sql)

apy_df['district'] = pd.Categorical(apy_df['district'], categories=['RAIGARH(CG)', 'BALAGHAT(MP)', 'AMRAVATI(MH)', 'CHANDRAPUR(MH)', 'NAGPUR(MH)', 'NASHIK(MH)', 'All India'], ordered=True)
apy_df.head()

#%%
{d:d for d in apy_df['district']}

#%%
g = (ggplot(apy_df)
    + geom_smooth(aes(x='year', y='yield', color='district', group='district'), method='loess')
    + labs(title='Target districts yield', x='Year', y='Yield($ton\ Ha^{-1}$)')
    + scale_color_manual(name='Crop', values={'RAIGARH(CG)': 'red', 'BALAGHAT(MP)': 'blue', 'AMRAVATI(MH)': 'green', 'CHANDRAPUR(MH)': 'yellow', 'NAGPUR(MH)': 'orange', 'NASHIK(MH)': 'purple', 'All India': 'gray',})                                  
    + theme(
        figure_size=(10, 3),
        axis_text_x=element_text(angle=90, va="top", ha="center", size=8),
        plot_title=element_text(ha='center', size=14)
    )
    + facet_grid('.~apy_crop')
)
g.save(filename="target_district_yield.png", path=chart_dir, units='cm', dpi=300)
g

# %%
g = (ggplot(apy_df)
    + geom_smooth(aes(x='year', y='area/1000', color='district', group='district'), method='loess')
    + labs(title='Target districts crop area', x='Year', y='Area (Ha x 1000)')
    + scale_color_manual(name='Crop', values={'RAIGARH(CG)': 'red', 'BALAGHAT(MP)': 'blue', 'AMRAVATI(MH)': 'green', 'CHANDRAPUR(MH)': 'yellow', 'NAGPUR(MH)': 'orange', 'NASHIK(MH)': 'purple', 'All India': 'gray',})                                  
    + theme(
        figure_size=(10, 3),
        axis_text_x=element_text(angle=90, va="top", ha="center", size=8),
        plot_title=element_text(ha='center', size=14)
    )
    + facet_grid('.~apy_crop')
)
g.save(filename="target_district_area.png", path=chart_dir, units='cm', dpi=300)
g



# %%
largest_farm_size_sql = """select * from vwM_farm_size_largest_n2o_n_top_15_crops_eagle_2020_long"""
largest_farm_size_df = load_map_data(db_client_input, largest_farm_size_sql)
largest_farm_size_df['farm_size'] = pd.Categorical(largest_farm_size_df['farm_size']
                                    , categories=['MARGINAL (BELOW 1.0)', 'SMALL (1.0 - 1.99)', 'SEMI-MEDIUM (2.0 - 3.99)','MEDIUM (4.0 - 9.99)','LARGE (10 AND ABOVE)' ], ordered=True)

largest_farm_size_df['farm_size'] = largest_farm_size_df['farm_size'].cat.rename_categories({'MARGINAL (BELOW 1.0)': '<1.0 Ha',
                                                                        'SMALL (1.0 - 1.99)': '1.0 - 1.99 Ha',
                                                                        'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
                                                                        'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
                                                                        'LARGE (10 AND ABOVE)': '≥10 Ha'    })




largest_farm_size_df.head()


# %%
units = "Farm size"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot()
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(largest_farm_size_df, aes(fill="farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
             + labs(title='Farm size with largest $N_2O$ emissions (Eagle et al. 2020)')
     
    + scale_fill_manual(name='Farm size', values={'<1.0 Ha': 'red', '1.0 - 1.99 Ha': 'blue', '2.0 - 3.99 Ha': 'green', '4.0 - 9.99 Ha': 'yellow', '≥10 Ha': 'purple'})
     #+ scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=200, high=high_color, name=units) #, limits=[100,300])
     + facet_wrap('~measure')
     + theme(
         figure_size=(16,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=14),
         strip_text=element_text(size=12)
     )
)
g.save(filename="map_largest_n2o_n_farm_size.png", path=chart_dir,  units='cm', dpi=300)
g


# %%

# %%
largest_farm_size_crop_sql = """select * from vwM_farm_size_largest_n2o_n_by_crops_eagle_2020_long"""
largest_farm_size_crop_df = load_map_data(db_client_input, largest_farm_size_crop_sql)
largest_farm_size_crop_df['farm_size'] = pd.Categorical(largest_farm_size_crop_df['farm_size']
                                    , categories=['MARGINAL (BELOW 1.0)', 'SMALL (1.0 - 1.99)', 'SEMI-MEDIUM (2.0 - 3.99)','MEDIUM (4.0 - 9.99)','LARGE (10 AND ABOVE)' ], ordered=True)

largest_farm_size_crop_df['farm_size'] = largest_farm_size_crop_df['farm_size'].cat.rename_categories({'MARGINAL (BELOW 1.0)': '<1.0 Ha',
                                                                        'SMALL (1.0 - 1.99)': '1.0 - 1.99 Ha',
                                                                        'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
                                                                        'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
                                                                        'LARGE (10 AND ABOVE)': '≥10 Ha'    })




largest_farm_size_crop_df.head()


# %%
units = "Farm size"
low_color = "green"
mid_color= "yellow"
high_color = "red"
largest_farm_size_rice_wheat_df = largest_farm_size_crop_df[largest_farm_size_crop_df['apy_crop'].isin(['Rice','Wheat'])] 
g = (ggplot()
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(largest_farm_size_rice_wheat_df, aes(fill="farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
             + labs(title='Farm size with largest $N_2O$ emissions')
     
    + scale_fill_manual(name='Farm size', values={'<1.0 Ha': 'red', '1.0 - 1.99 Ha': 'blue', '2.0 - 3.99 Ha': 'green', '4.0 - 9.99 Ha': 'yellow', '≥10 Ha': 'purple'})
     #+ scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=200, high=high_color, name=units) #, limits=[100,300])
     + facet_grid('measure~apy_crop')
     + theme(
         figure_size=(9,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=14),
         strip_text=element_text(size=12)
     )
)
g.save(filename="map_largest_rice_wheat_n2o_n_farm_size.png", path=chart_dir,  units='cm', dpi=300)
g

# %%
farm_size_ha_sql = """with z as
	(
	select top 15 apy_crop 
		,sum(adj_crop_area) as crop_area 
	from vwA_n2o_n_results
	group by apy_crop
	order by crop_area desc
	)
select farm_size
	,sum(adj_crop_area) as adj_crop_area
	, sum(adj_crop_area*[total_mean_n2o_n_eagle_2020_kg_ha])/sum(adj_crop_area) as [mean_total_kg_n2o_n_eagle_2020_ha]
	, sqrt(sum(adj_crop_area*power([total_sd_n2o_n_eagle_2020_kg_ha],2))/sum(adj_crop_area)) as [sd_total_kg_n2o_n_eagle_2020_ha]
from vwA_n2o_n_results
where 1=1--apy_crop <> 'rice'
	and apy_crop in(select apy_crop from z)
group by farm_size """

farm_size_ha_df = load_table_data(db_client_input, farm_size_ha_sql)
farm_size_ha_df['farm_size'] = pd.Categorical(farm_size_ha_df['farm_size']
                                    , categories=['MARGINAL (BELOW 1.0)', 'SMALL (1.0 - 1.99)', 'SEMI-MEDIUM (2.0 - 3.99)','MEDIUM (4.0 - 9.99)','LARGE (10 AND ABOVE)' ], ordered=True)    
farm_size_ha_df['farm_size'] = farm_size_ha_df['farm_size'].cat.rename_categories({'MARGINAL (BELOW 1.0)': '<1.0 Ha',
                                                                        'SMALL (1.0 - 1.99)': '1.0 - 1.99 Ha',
                                                                        'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
                                                                        'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
                                                                        'LARGE (10 AND ABOVE)': '≥10 Ha'    })

farm_size_ha_df.head()

# %%
g = (ggplot(farm_size_ha_df)
    + geom_col(aes(x='farm_size', y='mean_total_kg_n2o_n_eagle_2020_ha', fill='farm_size'))
    + geom_errorbar(aes(x='farm_size', ymin='mean_total_kg_n2o_n_eagle_2020_ha - sd_total_kg_n2o_n_eagle_2020_ha', ymax='mean_total_kg_n2o_n_eagle_2020_ha + sd_total_kg_n2o_n_eagle_2020_ha'), width=0.2)
    #+ geom_text(aes(x='farm_size', y='mean_total_kg_n2o_n_eagle_2020_ha', label='mean_total_kg_n2o_n_eagle_2020_ha'), va='bottom', ha='center', size=8, format_string='{:.2f}')
    + labs(title='Farm size with largest $N_2O$ emissions', y='$Kg\ N_2O-N\ Ha^{-1}$', x='Farm size')
    + scale_y_continuous(limits=(0, 3))
    + scale_fill_manual(name='Farm size', values={'<1.0 Ha': 'red', '1.0 - 1.99 Ha': 'blue', '2.0 - 3.99 Ha': 'green', '4.0 - 9.99 Ha': 'yellow', '≥10 Ha': 'purple'})
    + theme(
        figure_size=(8,8),
        axis_text_x=element_text(angle=90, va="top", ha="center", size=8),
        plot_title=element_text(ha='center', size=14)
    )
)
g.save(filename="farm_size_n2o_n.png", path=chart_dir, units='cm', dpi=300)
g


# %%
largest_farm_size_sql = """select * from vwM_farm_size_largest_ch4_rice_bhatia_2013_long"""
largest_farm_size_df = load_map_data(db_client_input, largest_farm_size_sql)
largest_farm_size_df['farm_size'] = pd.Categorical(largest_farm_size_df['farm_size']
                                    , categories=['MARGINAL (BELOW 1.0)', 'SMALL (1.0 - 1.99)', 'SEMI-MEDIUM (2.0 - 3.99)','MEDIUM (4.0 - 9.99)','LARGE (10 AND ABOVE)' ], ordered=True)

largest_farm_size_df['farm_size'] = largest_farm_size_df['farm_size'].cat.rename_categories({'MARGINAL (BELOW 1.0)': '<1.0 Ha',
                                                                        'SMALL (1.0 - 1.99)': '1.0 - 1.99 Ha',
                                                                        'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
                                                                        'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
                                                                        'LARGE (10 AND ABOVE)': '≥10 Ha'    })




largest_farm_size_df.head()


# %%
units = "Farm size"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot()
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(largest_farm_size_df, aes(fill="farm_size"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
             + labs(title='Farm size with largest $CH_4$ emissions (Bhatia et al., 2013)')
     
    + scale_fill_manual(name='Farm size', values={'<1.0 Ha': 'red', '1.0 - 1.99 Ha': 'blue', '2.0 - 3.99 Ha': 'green', '4.0 - 9.99 Ha': 'yellow', '≥10 Ha': 'purple'})
     #+ scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=200, high=high_color, name=units) #, limits=[100,300])
     + facet_wrap('~measure')
     + theme(
         figure_size=(16,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=14),
         strip_text=element_text(size=12)
     )
)
g.save(filename="map_largest_ch4_farm_size.png", path=chart_dir,  units='cm', dpi=300)
g

# %%
largest_crop_sql = """select * from vwM_crop_largest_n2o_n_top_15_crops_eagle_2020_long"""
largest_crop_df = load_map_data(db_client_input, largest_crop_sql)

largest_crop_df.head()

# %%
units = "Crop"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot()
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(largest_crop_df, aes(fill="apy_crop"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
             + labs(title='Crop with largest $N_2O$ emissions (Eagle et al. 2020)')
     
    + scale_fill_discrete(name='Crop')
     #+ scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=200, high=high_color, name=units) #, limits=[100,300])
     + facet_wrap('~measure')
     + theme(
         figure_size=(16,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=14),
         strip_text=element_text(size=12)
     )
)
g.save(filename="map_largest_n2o_n_crop.png", path=chart_dir,  units='cm', dpi=300)
g

# %%
all_total_emission_sql  = """
select geog, log(co2e) as co2e, emission_label, units
from vwM_district_all_crop_emission_co2e_long
where gwp_time_period = 100
    and measure = 'total_ton_co2e'
"""
all_total_emission_df = load_map_data(db_client_input, all_total_emission_sql)
all_total_emission_df.head()


# %%
units = "log($CO_2e_{100}$)"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(all_total_emission_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="co2e"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
             + labs(title='District Crop Emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=10, high=high_color, name=units, limits=[0,15])
     + facet_grid('units~emission_label')
     + theme(
         figure_size=(24,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=element_text(ha='center', size=24),
         strip_text=element_text(size=22)
         , legend_title=element_text(size=20)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_crop_emission_total_ton.png", path=chart_dir,  units='cm', dpi=300)
g

#%%
all_kg_ha_emission_sql  = """
select geog, log(co2e) as co2e, emission_label, units
from vwM_district_all_crop_emission_co2e_long
where gwp_time_period = 100
    and measure = 'kg_co2e_ha'
"""
all_kg_ha_emission_df = load_map_data(db_client_input, all_kg_ha_emission_sql)
all_kg_ha_emission_df.head()


# %%
units = "log($CO_2e_{100}\ Ha^{-1}$)"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(all_kg_ha_emission_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="co2e"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            # + labs(title='Crop Emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=6, high=high_color, name=units, limits=[0,9])
     + facet_grid('units~emission_label')
     + theme(
         figure_size=(24,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title=None, # element_text(ha='center', size=24),
         strip_text=element_text(size=22)
         , legend_title=element_text(size=20)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_crop_emissions_kg_ha.png", path=chart_dir,  units='cm', dpi=300)
g
# %%
all_kg_ha_emission_df['co2e'].describe()

# %%
all_model_sql = """
select concat(replace(replace(crop,'All Crops',''),'Upland crops','Non-rice'), emission,'(',replace(model,' et al.,',''),')') as grp, model, total_Mt_co2e, sd_total_Mt_co2e
from vwG_national_co2e_all_results
where gwp_time_period = 100 
	--and not(model = 'Bhatia et al., 2013' and emission = 'N2O')
	and not(model = 'IPCC 2019 Update' and crop = 'Upland crops')"""



all_model_df = load_table_data(db_client_input, all_model_sql)  

# %%
models = all_model_df['grp'].unique()
print(models)
print({m:m for m in models})

# %%
all_model_df['grp'] = pd.Categorical(all_model_df['grp'], categories=[
 'Residue Burning CO2e(Residue Burning)', 
 'RiceN2O(Shcherbak 2014)',
 'RiceN2O(IPCC 2019 Update)', 
 'RiceN2O(Eagle 2020)',
 'RiceN2O(Hiroko Akiyama 2005)',
 'RiceN2O(Bhatia 2013)',
 'Non-riceN2O(Bhatia 2013)',
 'Non-riceN2O(Shcherbak 2014)',
 'Non-riceN2O(Eagle 2020)', 
 'RiceCH4(Yan 2005)',
 'RiceCH4(Bhatia 2013)',
 'RiceCH4(Nikolaisen 2023)'
 ], ordered=True)


all_model_df['grp'] = all_model_df['grp'].cat.rename_categories({
 'Residue Burning CO2e(Residue Burning)': 'Residue Burning',
 'RiceN2O(Shcherbak 2014)': 'Rice $N_2O$\n(Shcherbak 2014)',
 'RiceN2O(IPCC 2019 Update)': 'Rice $N_2O$\n(IPCC 2019)',
 'RiceN2O(Eagle 2020)': 'Rice $N_2O$\n(Eagle 2020)',
 'RiceN2O(Hiroko Akiyama 2005)': 'Rice $N_2O$\n(Akiyama 2005)',
 'RiceN2O(Bhatia 2013)':'Rice $N_2O$\n(Bhatia 2013)', 
 'Non-riceN2O(Bhatia 2013)': 'Non-rice $N_2O$\n(Bhatia 2013)',
 'Non-riceN2O(Shcherbak 2014)': 'Non-rice $N_2O$\n(Shcherbak 2014)',
 'Non-riceN2O(Eagle 2020)': 'Non-rice $N_2O$\n(Eagle 2020)',
 'RiceCH4(Yan 2005)': 'Rice $CH_4$\n(Yan 2005)',
 'RiceCH4(Bhatia 2013)': 'Rice C$CH_4$H4\n(Bhatia 2013)',
 'RiceCH4(Nikolaisen 2023)': 'Rice $CH_4$\n(Nikolaisen 2023)'
  })
all_model_df


#%%
p = (ggplot(all_model_df)
            # + geom_violin(filter_df,aes(x='dataset', y='kg_inorganic_n_ha'),draw_quantiles=[0.25, 0.5, 0.75])
            + geom_col(aes(x='grp', y='total_Mt_co2e',fill='grp'))
            + geom_errorbar(aes(x='grp', ymin='total_Mt_co2e - sd_total_Mt_co2e', ymax='total_Mt_co2e + sd_total_Mt_co2e'), width=0.2)
          + geom_text(aes(x='grp', y='total_Mt_co2e', label='total_Mt_co2e'), 
                 va='bottom', ha='center', size=8, format_string='{:.2f}')
    
            # + geom_jitter(filter_df, aes(x='dataset', y='kg_inorganic_n_ha'), color='blue', alpha=.01, size=.001, width=0.3)
             + labs(title='', x="Emissions", y="$CO_2e\ Mt\ Year^{-1}$")
            # + scale_y_log10()
             +scale_y_continuous(limits=(0, 150))
            + guides(fill=None) 
             + theme(
            figure_size=(8,6),
            axis_text_x=element_text(angle=90, va="top", ha="center", size=12),
             plot_title=element_text(ha='center', size=14))
           # + facet_grid('.~crop + emission')
            )
print(p)
p.save(filename=f"national_rice_n2o_ghg_15x18_free_y.png", path=chart_dir, dpi=92)

#%%
#%%
ch4_emisions_df = all_model_df[(all_model_df['emission'] == 'CH4')]
p = (ggplot(ch4_emisions_df)
            # + geom_violin(filter_df,aes(x='dataset', y='kg_inorganic_n_ha'),draw_quantiles=[0.25, 0.5, 0.75])
            + geom_col(aes(x='model', y='total_Mt_co2e',fill='model'))
            + geom_errorbar(aes(x='model', ymin='total_Mt_co2e - sd_total_Mt_co2e', ymax='total_Mt_co2e + sd_total_Mt_co2e'), width=0.2)
          + geom_text(aes(x='model', y='total_Mt_co2e', label='total_Mt_co2e'), 
                 va='bottom', ha='center', size=8, format_string='{:.2f}')
    
            # + geom_jitter(filter_df, aes(x='dataset', y='kg_inorganic_n_ha'), color='blue', alpha=.01, size=.001, width=0.3)
             + labs(title='', x="$CH_4$", y="$CO_2e\ Mt\ Year^{-1}$")
            # + scale_y_log10()
             +scale_y_continuous(limits=(0, 120))
            + guides(fill=None) 
             + theme(
            figure_size=(1,6),
            axis_text_x=element_text(angle=90, va="top", ha="center", size=12),
             plot_title=element_text(ha='center', size=14))
           # + facet_grid('.~crop + emission')
            )
print(p)
p.save(filename=f"national_ch4_ghg_15x18_free_y.png", path=chart_dir, dpi=92)


# %%
residue_burning_sql = """select *
from vwM_district_residue_burning_co2e
where gwp_time_period = 100
"""
residue_burning_df = load_map_data(db_client_input, residue_burning_sql)
residue_burning_df.head()

#%% 
units = "$Gg\ CO_2e_{100}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(residue_burning_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="total_Gg_co2e_map"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total residue burning emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=35, high=high_color, name=units, limits=[0,130])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
              plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_total_residue_burning_emissions_Gg.png", path=chart_dir,  units='cm', dpi=300)
g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(residue_burning_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_co2e_ha"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare residue burning emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=175, high=high_color, name=units, limits=[0,500])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_residue_burning_emissions_kg_ha.png", path=chart_dir,  units='cm', dpi=300)
g

# %%
ch4_sql = """
select *
from vwM_district_ch4_co2e_bhatia_2013
where gwp_time_period = 100
"""
ch4_df = load_map_data(db_client_input, ch4_sql)
ch4_df.head()

#%% 
units = "$Gg\ CO_2e_{100}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(ch4_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="total_Gg_co2e_map"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $CH_4$ emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=750, high=high_color, name=units, limits=[0,2000])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
              plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_total_ch4_emissions_Gg.png", path=chart_dir,  units='cm', dpi=300)
g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(ch4_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="kg_co2e_ha"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $CH_4$ emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=3000, high=high_color, name=units, limits=[0,5500])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_ch4_emissions_kg_ha.png", path=chart_dir,  units='cm', dpi=300)
g



# %%
rice_n2o_sql = """
select *
from vwM_district_n2o_co2e_rice_results_ipcc_2019
where gwp_time_period = 100
"""
rice_n2o_df = load_map_data(db_client_input, rice_n2o_sql)
rice_n2o_df.head()

#%%
units = "$Ton\ CO_2e_{100}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(rice_n2o_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="fert_t_co2e_map"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total rice $N_2O$\nfertilizer induced emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=3, high=high_color, name=units, limits=[0,7])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
              plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_total_rice_n2o_emissions_Gg.png", path=chart_dir,  units='cm', dpi=300)
g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(rice_n2o_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="fert_kg_n2o_co2e_ha"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare rice $N_2O$\nfertilizer induced emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=2000, high=high_color, name=units, limits=[0,5500])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_rice_n2o_emissions_kg_ha.png", path=chart_dir,  units='cm', dpi=300)
g


# %%
n2o_sql = """
select *
from vwM_district_n2o_co2e_upland_crop_results_eagle_2020
where gwp_time_period = 100"""
n2o_df = load_map_data(db_client_input, n2o_sql)
n2o_df.head()

# %%
units = "$Gg\ CO_2e_{100}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(n2o_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_Gg_co2e_map"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Total non-rice $N_2O$\nfertilizer induced emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=65, high=high_color, name=units, limits=[0,150])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
              plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_total_n2o_emissions_Gg.png", path=chart_dir,  units='cm', dpi=300)
g

#%% 
units = "$Kg\ CO_2e_{100}\ Ha^{-1}$\n"
low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot(n2o_df)
    + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5,97.5))
     + scale_y_continuous(limits=(7.5,37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme_void()
    # + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     
    + geom_map(aes(fill="direct_kg_n2o_co2e_ha"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
            + labs(title='Per hectare non-rice $N_2O$\nfertilizer induced emissions')
     
   # + scale_fill_discrete(name='Crop')
    + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=250, high=high_color, name=units, limits=[0,600])
   
     + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
     )
)
g.save(filename="map_n2o_emissions_kg_ha.png", path=chart_dir,  units='cm', dpi=300)
g


# %%
farm_size_crop_prop_sql = """
select * from [vwG_crop_proportion_by_farm_size]
"""
farm_size_crop_prop_df = load_table_data(db_client_input, farm_size_crop_prop_sql)
farm_size_crop_prop_df.head()
farm_size_crop_prop_df.columns = [c if c in ["geog", "geometry"] else format_name(c).replace("Organic\nfertilizer", "Crop").replace(
    "organic\nfertilizer", "crop") for c in farm_size_crop_prop_df.columns]
farm_size_crop_prop_df.loc[farm_size_crop_prop_df.Landhold=="LARGE (10 AND ABOVE)",'Landhold'] = 'LARGE\n(10 AND\nABOVE)'
farm_size_crop_prop_df.loc[farm_size_crop_prop_df.Landhold=="MEDIUM (4.0 - 9.99)",'Landhold'] = 'MEDIUM\n(4.0 - 9.99)'
farm_size_crop_prop_df.loc[farm_size_crop_prop_df.Landhold=="SEMI-MEDIUM (2.0 - 3.99)",'Landhold'] = 'SEMI-MEDIUM\n(2.0 - 3.99)'
farm_size_crop_prop_df.loc[farm_size_crop_prop_df.Landhold=="SMALL (1.0 - 1.99)",'Landhold'] = 'SMALL\n(1.0 - 1.99)'
farm_size_crop_prop_df.loc[farm_size_crop_prop_df.Landhold=="MARGINAL (BELOW 1.0)",'Landhold'] = 'MARGINAL\n(BELOW 1.0)'

farm_size_crop_prop_df.loc[farm_size_crop_prop_df.Crop=="Rapeseed &Mustard",'Crop'] = 'Rapeseed\nand Mustard'



landhold_rank = ['LARGE\n(10 AND\nABOVE)',
                 'MEDIUM\n(4.0 - 9.99)',
                 'SEMI-MEDIUM\n(2.0 - 3.99)',
                 'SMALL\n(1.0 - 1.99)',
                 'MARGINAL\n(BELOW 1.0)']
farm_size_crop_prop_df["Landhold"] = farm_size_crop_prop_df["Landhold"].astype("category")
farm_size_crop_prop_df["Landhold"] = farm_size_crop_prop_df["Landhold"].cat.set_categories(landhold_rank, ordered=True)
# farm_size_crop_prop_df["Crop"] =     farm_size_crop_prop_df["Crop"].astype("category")
# farm_size_crop_prop_df["Crop"] =     farm_size_crop_prop_df["Crop"].cat.set_categories(crop_area_rank, ordered=True)
farm_size_crop_prop_df.head(10)

#%%

crop_area_rank = list(farm_size_crop_prop_df.sort_values(by=['Total crop area 100kha'], ascending=False)["Crop"])
top10_farm_size_crop_prop_df = farm_size_crop_prop_df[farm_size_crop_prop_df.Crop.isin(crop_area_rank[:50])]
top10_farm_size_crop_prop_df

# %%
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
l = (ggplot(top10_farm_size_crop_prop_df,aes(x="Landhold",y='Crop area ha', fill='Crop'))
     + geom_col(position="fill")
     + theme(figure_size=(8, 8),axis_text_x=element_text(rotation=0, hjust=.5),rect=element_rect(color=(0,0,0,0),fill=(0,0,0,0))) # panel_background=element_rect(fill="red"))
     + labs(title="Crop area by Size of Landhold", y="Proportion")
     + scale_y_continuous(labs="Proportion")
     + scale_fill_brewer(type="div", palette ="RdYlGn") #low=low_color, mid=mid_color, midpoint=700000, high=high_color, name=units)
     #+ scale_x_discrete(drop=True)
          + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
        ,axis_text_x=element_text(angle=90, va="top", ha="center", size=12)
        ,axis_text_y=element_text(size=12)
        ,axis_title_y=element_text(size=16)
        ,axis_title_x=element_text(size=16)
     )
     )
l.save(filename="crop_area_proportion_by_landhold.png", path=chart_dir, dpi=300)
l

# %%
def proper_case(x):
     return ' '.join([c[0].upper() + c[1:] for c in x.replace("_",' ').split(" ")])


map_div_sql = "select * from [vwM_crop_farm_size_diversity_index] where landhold = 'MARGINAL (BELOW 1.0)'"
map_div_df = load_map_data(db_client_input, map_div_sql)
map_div_df.columns = [c if c in ["geog", "geometry"] else proper_case(c) for c in map_div_df.columns]
map_div_df.loc[map_div_df.Landhold=="LARGE (10 AND ABOVE)",'Landhold'] = 'LARGE\n(10 AND\nABOVE)'
map_div_df.loc[map_div_df.Landhold=="MEDIUM (4.0 - 9.99)",'Landhold'] = 'MEDIUM\n(4.0 - 9.99)'
map_div_df.loc[map_div_df.Landhold=="SEMI-MEDIUM (2.0 - 3.99)",'Landhold'] = 'SEMI-MEDIUM\n(2.0 - 3.99)'
map_div_df.loc[map_div_df.Landhold=="SMALL (1.0 - 1.99)",'Landhold'] = 'SMALL\n(1.0 - 1.99)'
map_div_df.loc[map_div_df.Landhold=="MARGINAL (BELOW 1.0)",'Landhold'] = 'MARGINAL\n(BELOW 1.0)'

map_div_df["Landhold"] = map_div_df["Landhold"].astype("category")
map_div_df["Landhold"] = map_div_df["Landhold"].cat.set_categories(landhold_rank, ordered=True)

map_div_df.head()

#%%
units = "Sampson\nReciprocal\n"

low_color = "green"
mid_color= "yellow"
high_color = "red"
g = (ggplot()
     + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5, 97.5))
     + scale_y_continuous(limits=(7.5, 37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
     + theme(figure_size=(8, 8), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     + geom_map(map_div_df, aes(fill="Sampson Reciprocal"), color=None, show_legend=True)
    + geom_map(india_states, color="white", fill=None, size=.25, show_legend=False)
     + labs(title="Marginal Landhold Crop Diversity\nSimpson's Reciprocal Index")
    # + scale_fill_gradient(low=high_color, high=low_color , name=units)
     + scale_fill_gradient2(low=high_color, mid=mid_color, midpoint=4.75, high=low_color, name=units)

               + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
        ,axis_text_x=element_text(angle=90, va="top", ha="center", size=12)
        ,axis_text_y=element_text(size=12)
        ,axis_title_y=element_text(size=16)
        ,axis_title_x=element_text(size=16)
     )
)

g.save(filename="marginal_sampson_reciprocal_index.png", path=chart_dir,dpi=300)
g
# %%
map_fert_sql = "select * from vwM_actual_to_recommened_fertilizer_distance_by_farm_size where landhold = 'MARGINAL (BELOW 1.0)'"
map_fert_df = load_map_data(db_client_input, map_fert_sql)
map_fert_df.columns = [c if c in ["geog", "geometry"] else proper_case(c) for c in map_fert_df.columns]
map_fert_df.loc[map_fert_df.Landhold=="LARGE (10 AND ABOVE)",'Landhold'] = 'LARGE\n(10 AND\nABOVE)'
map_fert_df.loc[map_fert_df.Landhold=="MEDIUM (4.0 - 9.99)",'Landhold'] = 'MEDIUM\n(4.0 - 9.99)'
map_fert_df.loc[map_fert_df.Landhold=="SEMI-MEDIUM (2.0 - 3.99)",'Landhold'] = 'SEMI-MEDIUM\n(2.0 - 3.99)'
map_fert_df.loc[map_fert_df.Landhold=="SMALL (1.0 - 1.99)",'Landhold'] = 'SMALL\n(1.0 - 1.99)'
map_fert_df.loc[map_fert_df.Landhold=="MARGINAL (BELOW 1.0)",'Landhold'] = 'MARGINAL\n(BELOW 1.0)'

map_fert_df["Landhold"] = map_fert_df["Landhold"].astype("category")
map_fert_df["Landhold"] = map_fert_df["Landhold"].cat.set_categories(landhold_rank, ordered=True)

map_fert_df["Area Wt Avg Over Application Of N trunc"]= map_fert_df["Area Wt Avg Over Application Of N"].apply(lambda x: x if x < 400 else 400)
map_fert_df.head()

# %%
units = "$Kg N\ Ha^{-1}$\n"
g = (ggplot()
     + geom_map(india, fill='grey', color=None, show_legend=False)
     #+ scale_fill_brewer(type='div', palette=8)
     + scale_x_continuous(limits=(67.5, 97.5))
     + scale_y_continuous(limits=(7.5, 37.5))
     # + scale_size_continuous(range=(0.4, 1))
     + coord_cartesian()
       + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
        ,axis_text_x=element_text(angle=90, va="top", ha="center", size=12)
        ,axis_text_y=element_text(size=12)
        ,axis_title_y=element_text(size=16)
        ,axis_title_x=element_text(size=16)
       )
     + geom_map(map_fert_df, aes(fill="Area Wt Avg Over Application Of N trunc"), color=None, show_legend=True)
     + geom_map(india_states, fill=None, color="white", size=.25, show_legend=False)
     + labs(title="Marginal Landhold Over Nitrogen Application\nArea Weight Average of Nitrogen\nApplication over Recommendations")
     #+ scale_fill_gradient(low=low_color, high=high_color , name=units)
     + scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=100, high=high_color, name=units, limits=[0,400])
     )

g.save(filename="marginal_area_w_mean_over_n_application.png", path=chart_dir,  dpi=300)
g


# %%
graph_fert_sql = "select * from vwG_actual_to_recommened_fertilizer_over_n_by_farm_size"
graph_fert_df = load_table_data(db_client_input, graph_fert_sql)
graph_fert_df.columns = [c if c in ["geog", "geometry"] else proper_case(c) for c in graph_fert_df.columns]
graph_fert_df.loc[graph_fert_df.Landhold=="LARGE (10 AND ABOVE)",'Landhold'] = 'LARGE\n(10 AND\nABOVE)'
graph_fert_df.loc[graph_fert_df.Landhold=="MEDIUM (4.0 - 9.99)",'Landhold'] = 'MEDIUM\n(4.0 - 9.99)'
graph_fert_df.loc[graph_fert_df.Landhold=="SEMI-MEDIUM (2.0 - 3.99)",'Landhold'] = 'SEMI-MEDIUM\n(2.0 - 3.99)'
graph_fert_df.loc[graph_fert_df.Landhold=="SMALL (1.0 - 1.99)",'Landhold'] = 'SMALL\n(1.0 - 1.99)'
graph_fert_df.loc[graph_fert_df.Landhold=="MARGINAL (BELOW 1.0)",'Landhold'] = 'MARGINAL\n(BELOW 1.0)'

graph_fert_df["Landhold"] = graph_fert_df["Landhold"].astype("category")
graph_fert_df["Landhold"] = graph_fert_df["Landhold"].cat.set_categories(landhold_rank, ordered=True)

graph_fert_df
# %%
units = "Kg N\nHa$^-1$\n"
dodge_text = position_dodge(width=1)
graph_fert_df["prop"] = round(graph_fert_df['Proportion Area Over Fertilized']*100.0)                  # new


g = (ggplot(graph_fert_df,aes(x="Landhold",y='Area Wt Avg Over Application Of N', fill='Crop'))

     + geom_col(stat="identity", position='dodge', show_legend=False)
     + theme(figure_size=(8, 8),axis_text_x=element_text(rotation=0, hjust=.5), rect=element_rect(fill=(0,0,0,0), color=(0,0,0,0)))
     + labs(title="Over Application of Nitrogen\n(Area Weighted Average Per Hectare)")
     + geom_text(aes(y=-.5, label='Crop'),
                 position=dodge_text,
                 color='gray', size=8, angle=0, va='top')
     + geom_text(aes(label='prop'),                                    # new
                 position=dodge_text,
                 size=8, va='bottom', format_string='{}%')
     #+ scale_fill_gradient2(low=low_color, mid=mid_color, midpoint=100, high=high_color, name=units)
     + scale_x_discrete(drop=True)
     
       + theme(
         figure_size=(8,8),
         rect=element_rect(fill=(0, 0, 0, 0), color=(0, 0, 0, 0)),
         strip_background=element_rect(fill='white', color='white'),
         plot_title= element_text(ha='center', size=22),
         strip_text=element_text(size=20)
         , legend_title=element_text(size=18)
         , legend_text=element_text(size=14)
        ,axis_text_x=element_text(angle=90, va="top", ha="center", size=12)
        ,axis_text_y=element_text(size=12)
        ,axis_title_y=element_text(size=16)
        ,axis_title_x=element_text(size=16)
       )
     )

g.save(filename="per_hectare_over_application_n_by_landhold.png", path=chart_dir, width=15, height=15, units='cm', dpi=300)
g

# %%
