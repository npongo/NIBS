import PostDoc.db_clients.mssql_db_client as mssql
import patchworklib as pw
from plotnine import *
import numpy as np
import math as m
import pandas as pd
import geopandas as gpd
#from jenkspy import jenks_breaks
import jenkspy
version = 'v1_10'
conn_dict = {"server": ".\\npongo22", "database": f"india_ccafs_mot_results_{version}" }
# conn_dict["database"] = 'india_agriculture_census_ghg_results_v1'

db_client = mssql.SqlServerClient(conn_dict)
conn_dict_input = {"server": ".\\npongo22", "database": f"IndiaInputCensus" }
input_db_client = mssql.SqlServerClient(conn_dict_input)

conn_dict_ccafs = {"server": ".\\npongo22", "database": f"IndiaCCAFS_MOT" }
ccafs_db_client = mssql.SqlServerClient(conn_dict_ccafs)

def india_base_map():
    india_sql = "SELECT geog.STAsBinary() as geog FROM [dbo].[national_boundaries]"
    india = load_map_data(db_client, india_sql)
    #base_map = plot_map(None,india, None, 'base map', '')
    return india

def plot_box(data, x_cat, y, title):
    g = (ggplot()
         + geom_boxplot(data,aes(x=x_cat, y=y))
         + labs(title=title)
         )
    return g


def load_table_data(db_client, sql):
    """
    load an sql table into a dataframe
    :param db_client:
    :param sql:
    :return:
    """
    results, columns = db_client.exec_result_set_query(sql, return_defination=True)
    cols = [c[0] for c in columns]
    df = pd.DataFrame(results, columns=cols)
    return df


def load_map_data(db_client, sql, crs="EPSG:4326",geom_column='geog'):
    """
    load spatial dataset
    :param db_client:
    :param sql:
    :param crs:
    :param geom_column:
    :return:
    """
    df = load_table_data(db_client, sql)
    gs = gpd.GeoSeries.from_wkb(df[geom_column])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs=crs)
    return gdf

def upper_outliers(data, variable, by_variable, label_variable, coef=1.5, top_x=None):
    """
    upper outliers > 75% quantile + 1.5 * IQR
    :param data:
    :param variable:
    :param by_variable:
    :param label_variable:
    :param coef:
    :param top_x:
    :return:
    """
    outliers = None
    for i in np.unique(data[by_variable]):
        d_ul = data.loc[data[by_variable]==i,variable]
        upper_limit = outlier_upper_limit(d_ul, coef=coef)
        #print(upper_limit)
        d = data.loc[np.logical_and(data[variable]>upper_limit,data[by_variable]==i),[variable, by_variable, label_variable]]

        if outliers is None:
            if isinstance(top_x, int):
                d.sort_values(by=[variable], inplace=True, ascending=False)
                outliers = d.head(top_x)
            else:
                outliers = d
        else:
            if isinstance(top_x,int):
                d.sort_values(by=[variable], inplace=True, ascending=False)
                outliers = outliers.append(d.head(top_x))
            else:
                outliers = outliers.append(d)

    return outliers


def lower_outliers(data, variable, by_variable, label_variable, coef=1.5, top_x=None):
    """
    upper outliers < 25% quantile + 1.5 * IQR
    :param data:
    :param variable:
    :param by_variable:
    :param label_variable:
    :param coef:
    :param top_x:
    :return:
    """
    outliers = None
    for i in np.unique(data[by_variable]):
        d_ll = data.loc[data[by_variable]==i,variable]
        lower_limit = outlier_upper_limit(d_ll, coef=coef)
        #print(upper_limit)
        d = data.loc[np.logical_and(data[variable]<lower_limit,data[by_variable]==i),[variable, by_variable, label_variable]]

        if outliers is None:
            if isinstance(top_x, int):
                d.sort_values(by=[variable], inplace=True, ascending=False)
                outliers = d.head(top_x)
            else:
                outliers = d
        else:
            if isinstance(top_x,int):
                d.sort_values(by=[variable], inplace=True, ascending=False)
                outliers = outliers.append(d.head(top_x))
            else:
                outliers = outliers.append(d)

    return outliers


def outlier_upper_limit(data, coef=1.5):
    p75, p25 = np.percentile(data, [75, 25])
    iqr = p75 - p25
    # print(iqr)
    upper_limit = p75 + coef * iqr
    return upper_limit


def outlier_lower_limit(data, coef=1.5):
    """find_outlier <- function(x) {
  return(x < quantile(x, .25) - 1.5*IQR(x) | x > quantile(x, .75) + 1.5*IQR(x))
}"""
    p75, p25 = np.percentile(data, [75, 25])
    iqr = p75 - p25
    # print(iqr)
    lower_limit = p25 - coef * iqr
    return lower_limit

def outlier_upper_limit(data, coef=1.5):
    p75, p25 = np.percentile(data, [75, 25])
    iqr = p75 - p25
    # print(iqr)
    upper_limit = p75 + coef * iqr
    return upper_limit


def outlier_limits(data, coef=1.5):
    """find_outlier <- function(x) {
  return(x < quantile(x, .25) - 1.5*IQR(x) | x > quantile(x, .75) + 1.5*IQR(x))
}"""
    p75, p25 = np.percentile(data, [75, 25])
    iqr = p75 - p25
    # print(iqr)
    lower_limit = p25 - coef * iqr
    upper_limit = p75 + coef * iqr
    return lower_limit, upper_limit



def max_upper_limit(data, variable, by_variable, coef=1.5):
    outliers = list()
    for i in np.unique(data[by_variable]):
        d_ul = data.loc[data[by_variable]==i,variable]
        upper_limit = outlier_upper_limit(d_ul, coef=coef)
        outliers.append(upper_limit)

    return max(data) if len(outliers) == 0 else max(outliers)


def convert_catagorical(data):
    data_out = data.copy()
    if 'crop' in data_out.columns:
        crops = sorted(np.unique(data['crop']))
        data_out["crop"] = data_out["crop"].astype("category")
        data_out["crop"] = data_out["crop"].cat.set_categories(crops, ordered=True)
    if 'Size_ClassHA' in data_out.columns:
        farm_size = ['LARGE',
                     'MEDIUM',
                     'SEMI-MEDIUM',
                     'SMALL',
                     'MARGINAL']
        data_out.loc[data_out.Size_ClassHA=="LARGE (10 AND ABOVE)",'Size_ClassHA'] = 'LARGE'
        data_out.loc[data_out.Size_ClassHA=="MEDIUM (4.0 - 9.99)",'Size_ClassHA'] = 'MEDIUM'
        data_out.loc[data_out.Size_ClassHA=="SEMI-MEDIUM (2.0 - 3.99)",'Size_ClassHA'] = 'SEMI-MEDIUM'
        data_out.loc[data_out.Size_ClassHA=="SMALL (1.0 - 1.99)",'Size_ClassHA'] = 'SMALL'
        data_out.loc[data_out.Size_ClassHA=="MARGINAL (BELOW 1.0)",'Size_ClassHA'] = 'MARGINAL'
        data_out["Size_ClassHA"] = data_out["Size_ClassHA"].astype("category")
        data_out["Size_ClassHA"] = data_out["Size_ClassHA"].cat.set_categories(farm_size, ordered=True)
    if 'Irrigated' in data_out.columns:
        data_out.loc[data_out.Irrigated=="I",'Irrigated'] = 'Irrigated'
        data_out.loc[data_out.Irrigated=="UI",'Irrigated'] = 'Unirrigated'
        irrigation = ['Irrigated', 'Unirrigated']
        data_out["Irrigated"] = data_out["Irrigated"].astype("category")
        data_out["Irrigated"] = data_out["Irrigated"].cat.set_categories(irrigation, ordered=True)


    return data_out


def data_prep(data, percentile=.95, classes=5, null_out_of_range=True):
    data_out = data.copy()
    if percentile is not None:
        if not isinstance(percentile, list):
            percentile = [percentile]*9
        truncate = data_out.total_emissions_kg_co2e_ha.quantile(percentile[0])
        data_out.loc[data_out.total_emissions_kg_co2e_ha > truncate, 'total_emissions_kg_co2e_ha'] = None if null_out_of_range else truncate
        truncate = data_out.total_emissions_kg_co2e_production_kg.quantile(percentile[1])
        data_out.loc[
            data_out.total_emissions_kg_co2e_production_kg > truncate, 'total_emissions_kg_co2e_production_kg'] = None if null_out_of_range else truncate
        truncate = data_out.total_emissions_kg_co2e.quantile(percentile[2])
        data_out.loc[data_out.total_emissions_kg_co2e > truncate, 'total_emissions_kg_co2e'] = None if null_out_of_range else truncate

        truncate = data_out.ch4_kg_co2e_production_kg.quantile(percentile[3])
        data_out.loc[data_out.ch4_kg_co2e_production_kg > truncate, 'ch4_kg_co2e_production_kg'] = None if null_out_of_range else truncate
        truncate = data_out.ch4_kg_co2e_production_kg.quantile(percentile[4])
        data_out.loc[data_out.ch4_kg_co2e_production_kg > truncate, 'ch4_kg_co2e_production_kg'] = None if null_out_of_range else truncate
        truncate = data_out.ch4_kg_co2e.quantile(percentile[5])
        data_out.loc[data_out.ch4_kg_co2e > truncate, 'ch4_kg_co2e'] = None if null_out_of_range else truncate

        truncate = data_out.n_fert_kg_co2e_ha.quantile(percentile[6])
        data_out.loc[data_out.n_fert_kg_co2e_ha > truncate, 'n_fert_kg_co2e_ha'] = None if null_out_of_range else truncate
        truncate = data_out.n_fert_kg_co2e_production_kg.quantile(percentile[7])
        data_out.loc[data_out.n_fert_kg_co2e_production_kg > truncate, 'n_fert_kg_co2e_production_kg'] = None if null_out_of_range else truncate
        truncate = data_out.n_fert_kg_co2e.quantile(percentile[8])
        data_out.loc[data_out.n_fert_kg_co2e > truncate, 'n_fert_kg_co2e'] = None if null_out_of_range else truncate

    data_out.dropna(inplace=True)

    # if classes is an integer interpret as using natural breaks (jenks)
    if isinstance(classes, int):
        total_breaks = jenks_breaks.jenks_breaks(data_out['total_emissions_kg_co2e'], nb_class=classes)
        total_label_breaks = [f'{i:0.2f}' for i in total_breaks]
        total_joiners = ['-']*(len(total_breaks)-1)
        total_labels = [f'{b}{total_joiners[i]}{total_label_breaks[i+1]} Kg'
                        for i, b in enumerate(total_label_breaks[:-1])]
        data_out['total_emissions_kg_co2e_class'] = pd.cut(data_out['total_emissions_kg_co2e'],
                                                           bins=total_breaks,
                                                           labels=[str(i) for i in range(len(total_labels))],
                                                           include_lowest=True)

        ch4_breaks = jenks_breaks.jenks_breaks(data_out['ch4_kg_co2e'], nb_class=classes)
        ch4_label_breaks = [f'{i:0.2f}' for i in ch4_breaks]
        ch4_joiners = ['-']*(len(ch4_breaks)-1)
        ch4_labels = [f'{b}{ch4_joiners[i]}{ch4_label_breaks[i+1]} Kg'
                      for i, b in enumerate(ch4_label_breaks[:-1])]
        data_out['ch4_kg_co2e_class'] = pd.cut(data_out['ch4_kg_co2e'],
                                               bins=ch4_breaks,
                                               labels=[str(i) for i in range(len(ch4_labels))],
                                               include_lowest=True)

        n_fert_breaks = jenks_breaks.jenks_breaks(data_out['n_fert_kg_co2e'], nb_class=classes)
        n_fert_label_breaks = [f'{i:,.2f}' for i in n_fert_breaks]
        n_fert_joiners = ['-']*(len(n_fert_breaks)-1)
        n_fert_labels = [f'{b}{n_fert_joiners[i]}{n_fert_label_breaks[i+1]} Kg'
                         for i, b in enumerate(n_fert_label_breaks[:-1])]
        data_out['n_fert_kg_co2e_class'] = pd.cut(data_out['n_fert_kg_co2e'],
                                                  bins=n_fert_breaks,
                                                  labels=[str(i) for i in range(len(n_fert_labels))],
                                                  include_lowest=True)

        return data_out, total_labels, ch4_labels, n_fert_labels

    # if classes is a list interpret as using percentile breaks
    if isinstance(classes, list):
        total_breaks = np.percentile(data_out['total_emissions_kg_co2e'], classes)
        total_label_breaks = [f'{i:,.2f}' for i in total_breaks]
        total_joiners = ['-']*(len(total_breaks)-1)
        total_labels = [f'{b}{total_joiners[i]}{total_label_breaks[i+1]} Kg'
                        for i, b in enumerate(total_label_breaks[:-1])]
        data_out['total_emissions_kg_co2e_class'] = pd.cut(data_out['total_emissions_kg_co2e'],
                                                           bins=total_breaks,
                                                           labels=total_labels,
                                                           include_lowest=True)

        ch4_breaks = np.percentile(data_out['ch4_kg_co2e'], classes)
        ch4_label_breaks = [f'{i:,.2f}' for i in ch4_breaks]
        ch4_joiners = ['-']*(len(ch4_breaks)-1)
        ch4_labels = [f'{b}{ch4_joiners[i]}{ch4_label_breaks[i+1]} Kg'
                      for i, b in enumerate(ch4_label_breaks[:-1])]
        data_out['ch4_kg_co2e_class'] = pd.cut(data_out['ch4_kg_co2e'],
                                               bins=ch4_breaks,
                                               labels=ch4_labels,
                                               include_lowest=True
                                               )

        n_fert_breaks = np.percentile(data_out['n_fert_kg_co2e'], classes)
        n_fert_label_breaks = [f'{i:,.2f}' for i in n_fert_breaks]
        n_fert_joiners = ['-']*(len(n_fert_breaks)-1)
        n_fert_labels = [f'{b}{n_fert_joiners[i]}{n_fert_label_breaks[i+1]} Kg'
                         for i, b in enumerate(n_fert_label_breaks[:-1])]
        data_out['n_fert_kg_co2e_class'] = pd.cut(data_out['n_fert_kg_co2e'],
                                                  bins=n_fert_breaks,
                                                  labels=n_fert_labels,
                                                  include_lowest=True)


        return data_out, total_labels, ch4_labels, n_fert_labels

    return data_out, None, None, None


def plot_map(data, base_data, map_variable_name, title, units, low_color='blue', high_color='red', **kwargs):
    g = (ggplot()
         + geom_map(base_data, fill='grey', color=None, show_legend=False)
         #+ scale_fill_brewer(type='div', palette=8)
         + scale_x_continuous(limits=(67.5,97.5))
         + scale_y_continuous(limits=(7.5,37.5))
         # + scale_size_continuous(range=(0.4, 1))
         + coord_cartesian()
         + theme_void()
         + theme(figure_size=(8, 8), panel_background=element_rect(fill='white'))
         )
    if data is not None:
        g = (g+ geom_map(data, aes(fill=map_variable_name), color=None, show_legend=True)
             + labs(title=title)
             + scale_fill_gradient(low=low_color, high=high_color, name=units)
             )
    return g


def layout_plot_grid(plots, cols=None, rows=None, figsize=(1,1)):
    empty = pw.load_ggplot(ggplot(), figsize=figsize)
    if cols is None and rows is None:
        raise(Exception("Both cols and rows can not be None"))
    if cols is None:
        cols = m.ceil(float(len(plots))/rows)
    if rows is None:
        rows = m.ceil(float(len(plots))/cols)
    for i in range(rows):
        x = None
        for j in range(cols):
            idx = i * cols + j
            g = plots if idx < len(plots) else empty
            if x is None:
                x = g
            else:
                x = x|g
        if y is None:
            y = x
        else:
            y = y/x
    return y


def series_box_plots(df, by_variable, params, kargs):
    outplots = dict()
    for c in params.keys():
        print(c)
        p = params[c]
        xlab = by_variable[0].upper() + by_variable[1:]
        xlab = 'Farm Size' if 'Size_ClassHA' else xlab
        gbc = plot_box(df, by_variable, c, p['title']) + labs(y=p['units'], x=xlab) + scale_y_log10() + coord_flip()
        # print(gbc)
        outplots[c] = gbc
        
    return outplots

def series_map_plots(df, params, base_map, kargs):
    outplots = dict()
    df_local = df.copy()
    for c in params.keys():
        print(c)
        p = params[c]
        precision = (p or {}).get('precision',(kargs or {}).get('precision',4))
        df_local[c] = np.round(df_local[c],precision)
        # xlab = by_variable[0].upper() + by_variable[1:]
        # xlab = 'Farm Size' if 'Size_ClassHA' else xlab
        if len(np.unique(df_local[c])) > 25:
            ll, ul = outlier_limits(df_local[c])
            #print(ll,ul)
            df_local.loc[df_local[c] < ll, c] = ll
            df_local.loc[df_local[c] > ul, c] = ul
        gmp = plot_map(df_local, base_map, c, **p)

        outplots[c] = gmp
    return outplots


def create_plots_series(series_def, db_client, base_map, name_fn=None, kargs=None):
    output = dict()

    for s, s_def in series_def.items():
        print(s)
        output[s] = dict()
        output[s]['maps'] = dict()
        output[s]['boxplots'] = dict()
        #for t, t_def in s_def.items():
        sql = f"select * from {s}"
        df = load_map_data(db_client, sql)
        params = {c: area_wt_avg_params(i, c) for i, c in
                  enumerate([c for c in df.columns if c.startswith('area_wt_avg')])}
        gcol = None
        if s_def['grouper'] is not None:
            gcol = [c for c in df.columns if c.lower() == s_def['grouper'].lower()][0]
            df[gcol] = df[gcol].astype("category")
        if 'map' in s_def['plots']:
            print('map variable')
            gms = series_map_plots(df, params, base_map, kargs)  ##f() returns list of maps for all variables in  or dict?
            output[s]['maps'] = gms
        if 'boxplot' in s_def['plots']:
            print('boxplot')
            by_variable = gcol
            gbs = series_box_plots(df, by_variable, params, kargs)##f() returns list of blox plots for all variables in  or dict?
            output[s]['boxplots'] = gbs
    return output


colors = ['green', 'blue', 'red', 'orange', 'purple', 'violet']

def area_wt_avg_params(i, c):
    w = c.split('_')
    caps = [l[0].upper() + l[1:] for l in w]
    title = ' '.join(caps)
    title = title.replace('Kg Ha0', 'Kg Ha$^{-1}$').replace('Wt', 'Weight')
    low_color = 'white'
    high_color = colors[i]
    units = caps[-1]
    if c.endswith('perc_uap'):
        units = 'Percent UAP'
    if c.endswith('kg_co2e_ha'):
        units = 'CO$_2$e Kg Ha$^{-1}$'
    if c.endswith('kg_ha'):
        units = 'Kg Ha$^{-1}$'
    if c.endswith('n_kg_ha'):
        units = 'N Kg Ha$^{-1}$'
    return {'title': title, 'units': units, 'low_color': low_color, 'high_color': high_color}


def avg_wt_graphs():
    dfs = {"crop_ch4_emissions":{"vwR_map_crop_ch4":{"grouper":None, "plots":["map"]},"vwR_map_crop_ch4_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_crop_ch4_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_crop_ch4_irrigated":{"grouper":"irrigated", "plots":["boxplot"]}},"fertilizer_production_inputs_emissions":{"vwR_map_fertilizer_production_inputs":{"grouper":None, "plots":["map"]},"vwR_map_fertilizer_production_inputs_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_fertilizer_production_inputs_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_fertilizer_production_inputs_fertilizer":{"grouper":"fertilizer", "plots":["boxplot"]},"vwR_map_fertilizer_production_inputs_irrigated":{"grouper":"irrigated", "plots":["boxplot"]}},"n2o_emissions":{"vwR_map_n2o":{"grouper":None, "plots":["map"]},"vwR_map_n2o_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_n2o_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_n2o_irrigated":{"grouper":"irrigated", "plots":["boxplot"]},"vwR_map_n2o_soil_texture":{"grouper":"soil_texture", "plots":["boxplot"]}},"n2o_leaching_emissions":{"vwR_map_n2o_leaching":{"grouper":None, "plots":["map"]},"vwR_map_n2o_leaching_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_n2o_leaching_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_n2o_leaching_irrigated":{"grouper":"irrigated", "plots":["boxplot"]}},"nh3_emissions":{"vwR_map_nh3":{"grouper":None, "plots":["map"]},"vwR_map_nh3_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_nh3_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_nh3_fert_app_method":{"grouper":"fert_app_method", "plots":["boxplot"]},"vwR_map_nh3_fertilizer":{"grouper":"fertilizer", "plots":["boxplot"]},"vwR_map_nh3_irrigated":{"grouper":"irrigated", "plots":["boxplot"]},"vwR_map_nh3_soil_texture":{"grouper":"soil_texture", "plots":["boxplot"]}},"no_emissions":{"vwR_map_no":{"grouper":None, "plots":["map"]},"vwR_map_no_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_no_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_no_irrigated":{"grouper":"irrigated", "plots":["boxplot"]},"vwR_map_no_soil_texture":{"grouper":"soil_texture", "plots":["boxplot"]}},"rice_ch4_emissions":{"vwR_map_rice_ch4":{"grouper":None, "plots":["map"]},"vwR_map_rice_ch4_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_rice_ch4_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_rice_ch4_irrigated":{"grouper":"irrigated", "plots":["boxplot"]},"vwR_map_rice_ch4_preseason_water_status":{"grouper":"preseason_water_status", "plots":["boxplot"]},"vwR_map_rice_ch4_rice_climate":{"grouper":"rice_climate", "plots":["boxplot"]},"vwR_map_rice_ch4_water_regime":{"grouper":"water_regime", "plots":["boxplot"]}},"urea_emissions":{"vwR_map_urea":{"grouper":None, "plots":["map"]},"vwR_map_urea_crop":{"grouper":"crop", "plots":["boxplot"]},"vwR_map_urea_farm_size":{"grouper":"Size_ClassHA", "plots":["boxplot"]},"vwR_map_urea_fertilizer":{"grouper":"fertilizer", "plots":["boxplot"]},"vwR_map_urea_irrigated":{"grouper":"irrigated", "plots":["boxplot"]}}}
    india = india_base_map()
    for k, v in dfs.items():
        print(k)
        p = create_plots_series(v, db_client, india, name_fn=None, kargs=None)
        print(p)


def format_column(c):
    label = c.replace('_',' ') \
        .replace('no','NO') \
        .replace('n2o','N$_2$O') \
        .replace('ch4','CH$_4$') \
        .replace('caco3','CaCO$_3$') \
        .replace('nh3','NH$_3$') \
        .replace('kg n production kg','N Kg Crop Kg$^{-1}$') \
        .replace('kg co2e production kg','CO$_2$ Kg Crop Kg$^{-1}$') \
        .replace('kg co2e ha','CO$_2$ Kg Ha$^{-1}$') \
        .replace('kg n ha','N Kg Ha$^{-1}$') \
        .replace('kg co2e','CO$_2$ Kg') \
        .replace('kg n','N Kg') \
        .replace('n kg','N Kg')
    label = ' '.join([w[0].upper() + w[1:] for w in label.split(' ')])

    unit = 'N Kg Crop Kg$^{-1}$' if c.endswith('kg_n_production_kg') else None
    unit = 'Kg Crop Kg$^{-1}$' if c.endswith('kg_production_kg') else unit
    unit = 'CO$_2$ Kg Crop Kg$^{-1}$' if c.endswith('kg_co2e_production_kg') else unit
    unit = 'CO$_2$ Kg Ha$^{-1}$' if c.endswith('kg_co2e_ha') else unit
    unit = 'N Kg Ha$^{-1}$' if c.endswith('kg_n_ha') else unit
    unit = 'CO$_2$ Kg' if c.endswith('kg_co2e') else unit
    unit = 'N Kg' if c.endswith('kg_n') else unit
    unit = 'N Kg' if c.endswith('n_kg') else unit
    unit = 'N Kg Ha$^{-1}$' if c.endswith('n_kg_ha') else unit


    low_color, high_color = ('lime', 'red') if c.endswith('production_kg') else (None,None)
    low_color, high_color = ('green', 'red') if c.endswith('ha') else (low_color, high_color)
    low_color, high_color = ('#31a354', '#b30000') if c.endswith('kg') else (low_color, high_color)
    low_color, high_color = ('#31a354', '#b30000') if c.endswith('kg_n') else (low_color, high_color)
    low_color, high_color = ('green', 'red') if c.endswith('kg_co2e') else (low_color, high_color)

    # '#31a354','#c2e699','#fef0d9','#fc8d59','#b30000'
    return {'label': label, 'units': unit, 'low_color': low_color, 'high_color': high_color}


if __name__ == '__main__':
    # sql = "select * from [dbo].[MapDistrictCropRiceEmissionsByFarmSize]"
    # df = load_map_data(db_client, sql)
    # df = convert_catagorical(df)
    # df_outliers = upper_outliers(df,'total_fert_n_kg_ha','Size_ClassHA', 'label', coef=1.5, top_x=5)
    # df_outliers.size
    #
    # box = plot_box(df, 'Size_ClassHA', 'total_fert_n_kg_ha', 'title')
    # gbl = box + geom_text(df_outliers, aes(x='Size_ClassHA',y='total_fert_n_kg_ha', label='label'), size=8)
    # plot_map(df, 'total_fert_n_kg_ha', 'dkdkd' )

    avg_wt_graphs()

#%%

#%%

#%%

#%%

#%%

#%%
