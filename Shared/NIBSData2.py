
from plotnine import *
import numpy as np
import math as m
import pandas as pd
import geopandas as gpd
import requests 
import duckdb 
from os import path, makedirs
from PIL import Image
from scipy import stats
from scipy.stats import chi2_contingency
from Shared.NIBSUrls import *
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

#NIBSData(data_dir=r"D:\Repos\NIBS\Data" , chart_dir=r"D:\Repos\NIBS\Graphs")
class NIBSData():

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set these directories to the location of the data and chart directories on the local system
    # you can customize the colors for the maps as well if desired
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, data_dir:str=r"B:\Repos\NIBS\Data"
                 , chart_dir:str=r"B:\Repos\NIBS\Graphs"
                 ,low_color:str="green"
                 ,mid_color:str="yellow"
                 ,high_color:str="red"
                 ,colors:list[str]= ['green', 'yellowgreen', 'yellow', 'orange', 'red']
                 , dpi:int=72)->None:
         
        self._data_dir = data_dir
        self._chart_dir = chart_dir

        if not path.exists(self._chart_dir):
            makedirs(self._chart_dir)
        if not path.exists(self._data_dir):
            makedirs(self._data_dir)

        self._dpi = dpi 
        self._low_color =low_color
        self._mid_color= mid_color
        self._high_color = high_color
        self._colors = colors

        self.apy_crop_replacements = {
            'other oilseeds': 'Oilseeds',  # '<1.0 Ha',
            'Moong(Green Gram)': 'Green Gram',  # '<1.0 Ha',
            'Rapeseed &Mustard': 'Rapeseed & Mustard',  # '<1.0 Ha',
            'Cotton(lint)': 'Cotton',  # '<1.0 Ha',
        }

        self._farm_size_replacements = {
            'MARGINAL (BELOW 1.0)': '<2.0 Ha',  # '<1.0 Ha',
            'SMALL (1.0 - 1.99)': '<2.0 Ha',  # '1.0 - 1.99 Ha',
            'SEMI-MEDIUM (2.0 - 3.99)': '2.0 - 3.99 Ha',
            'MEDIUM (4.0 - 9.99)': '4.0 - 9.99 Ha',
            'LARGE (10 AND ABOVE)': '≥10 Ha'
        }
        
        duckdb.execute("SET default_collation = 'nocase';")
        duckdb.execute("INSTALL spatial")
        duckdb.execute("LOAD spatial")
        self._duckdb = duckdb
        

    @property
    def data_dir(self)->str:
        return self._data_dir
    @property
    def chart_dir(self) -> str:
        return self._chart_dir

    @property
    def colors(self)->list[str]:
        return self._colors
    
    @property
    def dpi (self)->int:
        return self._dpi 
    
    @property
    def india (self)->gpd.GeoDataFrame:
        return self.get_india()
    
    @property
    def india_states (self)->gpd.GeoDataFrame:
        return self.get_india_states()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # base data for maps
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # urls = ['https://www.dropbox.com/scl/fi/llqo3pu1pbdgen981qnlm/national_boundaries.parquet?rlkey=amdb79rqwbboqxwbumadd82p7&dl=1',
        #         'https://www.dropbox.com/scl/fi/9r8ulaqp6vs6fsf4m6nj0/vwM_india_states.parquet?rlkey=1s14epati5c90xcww70ytv1nr&dl=1']
        
        self.india_states = self.get_india_states()
        self.india = self.get_india()

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # helper functions
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def percentile_score(self, data):
        return [stats.percentileofscore(data, n) for n in data]


    def download_missing_data(self, sql:str)->None:
        """download missing data from the web

        Args:
            sql (str): sql query to download the data
        """
        if "read_parquet" not in sql:
            return 
        unions = sql.lower().split("union")
        
        for sql in unions:
            parquet_file_path = sql[sql.find("read_parquet"):]
            parquet_file_path = parquet_file_path[parquet_file_path.find("('")+2:parquet_file_path.find("')")]
            url_key = path.basename(parquet_file_path).replace('.parquet','')
            url = get_data_url(url_key)
            file_name = url[:url.find('?')].split('/')[-1]
            save_path = path.join(self._data_dir, file_name)
            if not path.exists(save_path):
                print(f"Starting to download {file_name} from {url}")
                self.download_file(url, save_path)

    def load_dataset(self, dataset:str, crs:str="EPSG:4326",geom_column:str='geog')->pd.DataFrame:
        """
        load an sql table into a dataframe
        :param dataset: name of the dataset to load
        :return: dataframe
        """
        sql = f"select * from read_parquet('{path.join(self._data_dir,f"{dataset}.parquet")}')"
        return self.query_dataset(sql, crs, geom_column)

    def query_dataset(self, sql:str, crs:str="EPSG:4326",geom_column:str='geog')->pd.DataFrame:
        """
        load an sql table into a dataframe
        :param sql: sql query to execute
        :return: dataframe
        """
        self.download_missing_data(sql)
        df = self._duckdb.execute(sql).fetch_df()
        if geom_column in df.columns:
            geom_sql = f" EXCLUDE {geom_column}, CAST({geom_column} AS VARCHAR) AS {geom_column} from"
            sql = sql.lower().replace("from", geom_sql)
            df = self._duckdb.execute(sql).fetch_df()
            gdf = gpd.GeoDataFrame(df,geometry= gpd.GeoSeries.from_wkt(df[geom_column]),crs=crs)
            return gdf
        else:
            return df 

    def load_table_data(self, sql:str)->pd.DataFrame:
        """
        load an sql table into a dataframe
        :param sql: sql query to execute
        :return: dataframe
        """
        self.download_missing_data(sql)
        duckdb.execute("SET default_collation = 'nocase';")
        return duckdb.execute(sql).fetch_df()
    

    def load_map_data(self, sql:str, crs:str="EPSG:4326",geom_column:str='geog')->gpd.GeoDataFrame:
        """loads a parquet file into a geopandas dataframe. The function is profided for consistency with other data sources, ie database access.
        Args:
            sql (str): duckdb sql to read a parquet file
            crs (_type_, optional): CRS identifer. Defaults to "EPSG:4326". NOT USED
            geom_column (str, optional): name of geometry columns in the dataset. Defaults to 'geog'. NOT USED

        Returns:
            gpd.GeoDataFrame
        """
        self.download_missing_data(sql)
        # duckdb.execute("INSTALL spatial")
        # duckdb.execute("LOAD spatial")
        # duckdb.execute("SET default_collation = 'nocase';")
        df = self._duckdb.execute(sql).fetch_df()
        gdf = gpd.GeoDataFrame(df,geometry= gpd.GeoSeries.from_wkt(df[geom_column]),crs=crs)
        return gdf


    def download_file(self, url:str, save_path:str)->None:
        """Download a file from a url and save it to a local path

        Args:
            url (str): url to download the file from
            save_path (str): path to save the downloaded file
        """
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded to {save_path}")


    def download_parquet_files(self, urls:list[str], data_dir:str=None)->None:
        """download all the parquet files from the url list provided

        Args:
            urls (list[str]): list of urls to downlaod
            data_dir (str, optional): path to the directory to save the downloaded file in
        """
        if data_dir is None:
            data_dir = self._data_dir
        for url in urls:
            # print(f"processing url: {url}")
            file_name = url[:url.find('?')].split('/')[-1]
            save_path = path.join(data_dir, file_name)
            if not path.exists(save_path):
                # print(f"Starting to download {file_name} from {url}")
                self.download_file(url, save_path)


    # Open an image from a computer 
    def open_image_local(self, path_to_image:str)->np.array:
        """Open an image from a computer

        Args:
            path_to_image (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = Image.open(path_to_image) # Open the image
        image_array = np.array(image) # Convert to a numpy array
        return image_array # Output


    def format_name(self, c):
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



    def get_india_states(self)->gpd.GeoDataFrame:
        """load the india states boundary line data for mapping

        Returns:
            gpd.GeoDataFrame: india states boundary line data
        """
        #download_parquet_files(self.urls)
        parquet_file_path = path.join(self._data_dir,'vwM_india_states.parquet')
        sql = f"""select * exclude geog,cast(geog as varchar) as geog
                from read_parquet('{parquet_file_path}')"""
        india_states = self.load_map_data(sql)       
        return india_states

    def get_india(self)->gpd.GeoDataFrame:
        """load the national polygon boundary data as map background to show no data districts

        Returns:
            gpd.GeoDataFrame: national polygon boundary data
        """
        #download_parquet_files(self.urls)
        parquet_file_path = path.join(self._data_dir,'national_boundaries.parquet')
        sql = f"""select * exclude geog,cast(geog as varchar) as geog
                from read_parquet('{parquet_file_path}')"""
        india = self.load_map_data(sql)       
        return india


    def percentile_map(self, data:gpd.GeoDataFrame, variable:str, title:str, units:str
                       , file_name:str
                       , chart_dir:str=None
                       , precentiles:list[float] = [.4, .7, .9, 1]
                       , round:int=0
                       , colors:list=None
                       , legend_position:tuple[float,float]=(.65,.18)
                       , figure_size:tuple[float,float]=(7,8)
                       , title_size:int=22
                       , india_poly:gpd.GeoDataFrame=None
                       , india_states_lines:gpd.GeoDataFrame=None
                       , dpi:int=None
                       ,format=None)->ggplot:
        """Map a continuouse variable colorized by percentile breaks in the mapped variable. Use to map per hectare and total emissoins for the
        different models and gases.

        Args:
            data (gpd.GeoDataFrame): input data
            variable (str): continuous variable to plot
            title (str): title of the plot
            units (str): units of the variable use in the legend title
            file_name (str): file name to save the plot
            chart_dir (str, optional): directory to save the plot to. Defaults to chart_dir.
            precentiles (list, optional): list of percentiles to break the variable by. Defaults to [.4, .7, .9, 1].
            round (int, optional): round the breaks to this number of decimal places. Defaults to 0.
            colors (list, optional): colors for the percentiles breaks. Defaults to colors.
            legend_position (tuple, optional): location of legend. Defaults to (.7,.02).
            figure_size (tuple, optional): size of the plot. Defaults to (7,8).
            title_size (int, optional): size of the title. Defaults to 22.
            india_poly (gpd.GeoDataFrame, optional): india polygon background. Defaults to india.
            india_states_lines (gpd.GeoDataFrame, optional): overlay of state boundaries. Defaults to india_states.
            dpi (int, optional): dpi of the plot. Defaults to 92.

        Returns:
            ggplot: _description_
        """
        breaks = None
        try:
            if chart_dir is None:
                chart_dir = self._chart_dir
            if colors is None:
                colors = self._colors.copy()
            if india_states_lines is None:
                india_states_lines = self.get_india_states()
            if india_poly is None:
                india_poly = self.get_india()
            if dpi is None:
                dpi = self._dpi
            if format is None:
                format = file_name.split('.')[-1]

            data[f'{variable}_p_score'] = data[variable].rank(pct=True)
            breaks =  data[variable].quantile(precentiles).round(round)

            if len(breaks.unique()) < len(breaks):
                raise Exception(f"Duplicate breaks found:>{breaks}")
            
            g = (ggplot(data)
                + geom_map(india_poly, fill='grey', color='black', show_legend=False)
                + scale_x_continuous(limits=(67.5,97.5))
                + scale_y_continuous(limits=(7.5,37.5))
                + coord_cartesian()
                + theme_void()
                + geom_map(aes(fill=f'{variable}_p_score'), color=None, show_legend=True)
                + geom_map(india_states_lines, color="white", fill=None, size=.25, show_legend=False)
                + labs(title=title)
                + scale_fill_gradientn(colors=colors, values= [0]+precentiles, name=units, labels= list(breaks)  ) 
                + theme(
                    figure_size=figure_size
                    , plot_title= element_text(ha='center', ma='center', size=title_size)
                    , strip_text=element_text(size=20)
                    , legend_title=element_text(size=18)
                    , legend_text=element_text(size=14)
                    , legend_direction='vertical' 
                    , legend_position=legend_position
                )
            )
            g.save(filename=file_name, format=format, path=chart_dir,  units='cm', dpi=dpi)
            return g
        except Exception as e:
            print(f"{e}\ncolors:{len(colors)} breaks:{len([0]+list(breaks))}, percentiles:{len(precentiles)}\ncolors:{colors} breaks:{breaks}, percentiles:{precentiles}")
            return None


    def catagorical_map(self, data:gpd.GeoDataFrame
                        , variable:str
                        , title:str
                        , legend_name:str
                        , file_name:str
                        , chart_dir:str=None
                        , legend_position:tuple[float,float]=(.65,.18)
                        , figure_size:tuple[float,float]=(7,8)
                        , title_size:int=22
                        , india_poly:gpd.GeoDataFrame=None
                        , india_states_lines:gpd.GeoDataFrame=None
                        , dpi:int=None)->ggplot:
        """_summary_

        Args:
            data (gpd.GeoDataFrame): _description_
            variable (str): _description_
            title (str): _description_
            legend_name (str): _description_
            file_name (str): _description_
            chart_dir (str, optional): _description_. Defaults to None.
            legend_position (tuple, optional): _description_. Defaults to (.7,.02).
            figure_size (tuple, optional): _description_. Defaults to (7,8).
            title_size (int, optional): _description_. Defaults to 22.
            india_poly (gpd.GeoDataFrame, optional): _description_. Defaults to self.india.
            india_states_lines (gpd.GeoDataFrame, optional): _description_. Defaults to self.india_states.
            dpi (int, optional): _description_. Defaults to 92.

        Returns:
            ggplot: _description_
        """
        
        if chart_dir is None:
            chart_dir = self._chart_dir
        if india_states_lines is None:
            india_states_lines = self.get_india_states()
        if india_poly is None:
            india_poly = self.get_india()
        if dpi is None:
            dpi = self._dpi

        g = (ggplot(data)
            + geom_map(india_poly, fill='grey', color="black", show_legend=False)
            + scale_x_continuous(limits=(67.5,97.5))
            + scale_y_continuous(limits=(7.5,37.5))
            + coord_cartesian()
            + theme_void()
            + geom_map(aes(fill=variable), color=None, show_legend=True)
            + geom_map(india_states_lines, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=title)
            + scale_fill_discrete(name=legend_name)
            + theme(
                figure_size=figure_size
                , plot_title= element_text(ha='center', ma='center', size=title_size)
                , strip_text=element_text(size=20)
                , legend_title=element_text(size=18)
                , legend_text=element_text(size=14)
                , legend_direction='vertical' 
                , legend_position=legend_position
            )
        )
        g.save(filename=file_name, path=chart_dir,  units='cm', dpi=dpi)
        return g
    



    def manual_catagorical_map(self, data:gpd.GeoDataFrame
                               , variable:str, title:str
                               , legend_name:str
                               , colors:dict[str:str]
                               , file_name:str
                               , chart_dir:str=None
                               , legend_position:tuple[float,float]=(.65,.18)
                               , figure_size:tuple[float,float]=(7,8)
                               , title_size:int=22
                               , india_poly:gpd.GeoDataFrame=None
                               , india_states_lines:gpd.GeoDataFrame=None
                               , dpi:int=None)->ggplot:
        """_summary_
        Args:
            data (gpd.GeoDataFrame): _description_
            variable (str): _description_
            title (str): _description_
            legend_name (str): _description_
            file_name (str): _description_
            chart_dir (str, optional): _description_. Defaults to None.
            legend_position (tuple, optional): _description_. Defaults to (.7,.02).
            figure_size (tuple, optional): _description_. Defaults to (7,8).
            title_size (int, optional): _description_. Defaults to 22.
            india_poly (gpd.GeoDataFrame, optional): _description_. Defaults to self.india.
            india_states_lines (gpd.GeoDataFrame, optional): _description_. Defaults to self.india_states.
            dpi (int, optional): _description_. Defaults to 92.

        Returns:
            ggplot: map with manual color scale
        """
        
        if chart_dir is None:
            chart_dir = self._chart_dir
        if india_states_lines is None:
            india_states_lines = self.get_india_states()
        if india_poly is None:
            india_poly = self.get_india()
        if dpi is None:
            dpi = self._dpi
        if colors is None: 
            colors = self._colors
            
        g = (ggplot(data)
            + geom_map(india_poly, fill='grey', color=None, show_legend=False)
            + scale_x_continuous(limits=(67.5,97.5))
            + scale_y_continuous(limits=(7.5,37.5))
            + coord_cartesian()
            + theme_void()
            + geom_map(aes(fill=variable), color=None, show_legend=True)
            + geom_map(india_states_lines, color="white", fill=None, size=.25, show_legend=False)
            + labs(title=title)
            + scale_fill_manual(values=colors,name=legend_name)
            + theme(
                figure_size=figure_size
                , plot_title= element_text(ha='center', ma='center', size=title_size)
                , strip_text=element_text(size=20)
                , legend_title=element_text(size=18)
                , legend_text=element_text(size=14)
                , legend_direction='vertical' 
                , legend_position=legend_position
            )
        )
        g.save(filename=file_name, path=chart_dir, dpi=dpi)
        return g

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # model comparison graphs 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #cramer's v correlation matrix plots

    def cramers_v(self, x:pd.Series, y:pd.Series)->float: 
        """compute the cramers v value for two categorical variables

        Args:
            x (pd.Series): categorical variable 1
            y (pd.Series): categroical variable 2

        Returns:
            float: cramers v value 
        """
        contingency_table = pd.crosstab(x, y)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))


    def cramers_v_plot(self, data:pd.DataFrame
                       , file_name:str
                       , rename_dict:dict
                       , title:str
                       , text_size:int=12
                       , text_color:str="white"
                       , legend_title:str="Cramér\'s V"
                       , figure_size:tuple[float,float]=(7,6)
                       , title_size:int=22
                       , chart_dir:str=None
                       , dpi:int=None)->None:
        """headmap plot of model aggreement using cramers v for categorical variables

        Args:     
            data (pd.DataFrame): input data
            file_name (str): file name to save the plot
            rename_dict (dict): rename variables to plot labels
            title (str): plot title
            text_size (int, optional): size of text in the plot tiles. Defaults to 12.
            text_color (str, optional): color of text in the plot tiles. Defaults to "white".
            legend_title (str, optional): title of the legend. Defaults to "Cramér\'s V".
            figure_size (tuple[float,float], optional): size of the plot. Defaults to (7,5).
            title_size (int, optional): size of the plot title. Defaults to 22. 
            chart_dir (str, optional): directory to save plot to. Defaults to chart_dir.
            dpi (int, optional): dpi of the plot. Defaults to 92.

        Returns:
            ggplot: headmap plot of model aggreement using cramers v for categorical variables.
        """
        if chart_dir is None:
            chart_dir = self._chart_dir
        if dpi is None:
            dpi = self._dpi
        categorical_columns = list(rename_dict.keys())
        cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 == col2:
                    cramers_v_matrix.loc[col1, col2] = 1.0
                else:
                    cramers_v_matrix.loc[col1, col2] = self.cramers_v(data[col1], data[col2])

        # Transform the Cramér's V matrix into long format for plotting
        cramers_v_long = cramers_v_matrix.stack().reset_index()
        cramers_v_long.columns = ['model1', 'model2', 'Cramers_V']

        # Rename categories for better readability

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

        #plot data as headmap
        heatmap_plot = (
            ggplot(cramers_v_long, aes(x='model1', y='model2', fill='Cramers_V')) +
            geom_tile() +
            geom_text(aes(label='round(Cramers_V, 2)'), size=text_size, color=text_color) +
            scale_fill_gradient(low='blue', high='red', name=legend_title) +
            scale_x_discrete(limits=model1_categories) +
            scale_y_discrete(limits=model2_categories) +
            labs(title=title, x='', y='') +
            theme(axis_text_x=element_text(rotation=0, size=10, ha='center', ma='center'),
                axis_text_y=element_text(rotation=90, size=10, va='center', ha='center', ma='center'),
                axis_ticks=element_blank(),
                panel_background=element_rect(fill='white'),
                plot_title= element_text(ha='center', ma='center', size=title_size),
                figure_size=figure_size)
        )

        # Save the heatmap
        heatmap_plot.save(path.join(chart_dir, file_name), dpi=dpi)

        return heatmap_plot


    #spearmans correlation matrix plots

    def spearman_corr(self, group):
        return group.drop(columns='geog_checksum').corr(method='spearman')


    def avg_spearman_plot(self, data:pd.DataFrame
                          , file_name:str
                          , rename_dict:dict
                          , title:str
                          , text_size:int=12
                          , text_color:str="white"
                          , figure_size:tuple[float,float]=(7,6)
                          , title_size:int=22
                          , chart_dir:str=None
                          , dpi:int=None)->None:
        """headmap plot of model aggreement using spearman rank correlation for continuous variables

        Args:
            data (pd.DataFrame): input data
            file_name (str): file name to save the plot
            rename_dict (dict): rename variables to plot labels
            title (str): plot title
            text_size (int, optional): size of text in the plot tiles. Defaults to 12.
            text_color (str, optional): color of text in the plot tiles. Defaults to "white".
            figure_size (tuple[float,float], optional): size of the plot. Defaults to (7,5).
            title_size (int, optional): size of the plot title. Defaults to 22.
            chart_dir (str, optional): directory to save plot to. Defaults to chart_dir.
            dpi (int, optional): dpi of the plot. Defaults to 92.

        Returns:
            ggplot: headmap plot of model aggreement using cramers v for categorical variables.
        """
        if chart_dir is None:
            chart_dir = self._chart_dir
        if dpi is None:
            dpi = self._dpi
        #calc the spearman correlation for each district    
        selected_columns = data.loc[:, ['geog_checksum']+ list(rename_dict.keys())]
        grouped_spearman_corr = selected_columns.groupby('geog_checksum').apply(self.spearman_corr)

        # Transform the correlation matrix into long format
        grouped_spearman_corr_long = grouped_spearman_corr.stack().reset_index()
        grouped_spearman_corr_long.columns = ['geog_checksum', 'model1', 'model2', 'Spearman_Correlation']
        grouped_spearman_corr_long.head(9)

        # Calculate the average district Spearman correlation by model1 and model2
        average_spearman_corr = grouped_spearman_corr_long.groupby(['model1', 'model2'])['Spearman_Correlation'].mean().reset_index()
        average_spearman_corr['model1'] = average_spearman_corr['model1'].replace(rename_dict)
        average_spearman_corr['model2'] = average_spearman_corr['model2'].replace(rename_dict)
        average_spearman_corr['model1'] = pd.Categorical(values=average_spearman_corr['model1'], categories=average_spearman_corr['model1'].unique(), ordered=True)  
        average_spearman_corr['model2'] = pd.Categorical(values=average_spearman_corr['model2'], categories=average_spearman_corr['model2'].unique(), ordered=True)        
        heatmap_data = average_spearman_corr.pivot(index="model1", columns="model2", values="Spearman_Correlation")

        # Reverse the order of the categories
        model1_categories = average_spearman_corr['model1'].unique()[::-1]
        model2_categories = average_spearman_corr['model2'].unique()#[::-1]
        print(model1_categories, model2_categories)

        #plot data as headmap
        heatmap_plot = (
            ggplot(average_spearman_corr, aes(x='model1', y='model2', fill='Spearman_Correlation')) +
            geom_tile() +
            geom_text(aes(label='round(Spearman_Correlation, 2)'), size=text_size, color=text_color) +
            scale_fill_gradient(low='blue', high='red',name="Corr") +
            scale_x_discrete(limits=model1_categories) +
            scale_y_discrete(limits=model2_categories) +
            labs(title=title, x='', y='') +
            theme(axis_text_x=element_text(rotation=0, size=10, ha='center', ma='center'),
                axis_text_y=element_text(rotation=90, size=10, va='center', ha='center', ma='center'),
                axis_ticks=element_blank(),
                panel_background=element_rect(fill='white'),
                plot_title= element_text(ha='center', ma='center', size=title_size),
                figure_size=figure_size)
        )

        # Save the heatmap
        heatmap_plot.save(path.join(chart_dir, file_name), dpi=dpi)
        return heatmap_plot


    def spearman_plot(self, data:pd.DataFrame
                      , file_name:str
                      , rename_dict:dict
                      , title:str
                      , text_size:int=12
                      , text_color:str="white"
                      , figure_size:tuple[float,float]=(7,6)
                      , title_size:int=22
                      , chart_dir:str=None
                      , dpi:int=None)->None:
        """headmap plot of model aggreement using spearman rank correlation for continuous variables

        Args:
            data (pd.DataFrame): input data
            file_name (str): file name to save the plot
            rename_dict (dict): rename variables to plot labels
            title (str): plot title
            text_size (int, optional): size of text in the plot tiles. Defaults to 12.
            text_color (str, optional): color of text in the plot tiles. Defaults to "white".
            figure_size (tuple[float,float], optional): size of the plot. Defaults to (7,5).
            title_size (int, optional): size of the plot title. Defaults to 22.
            chart_dir (str, optional): directory to save plot to. Defaults to chart_dir.
            dpi (int, optional): dpi of the plot. Defaults to 92.

        Returns:
            ggplot: headmap plot of model aggreement using cramers v for categorical variables.
        """
        if chart_dir is None:
            chart_dir = self._chart_dir
        if dpi is None:
            dpi = self._dpi

        # calculate spearman correlation
        corr_kg_co2e_ha_df = data.loc[:, list(rename_dict.keys())]
        corr_kg_co2e_ha_matrix = corr_kg_co2e_ha_df.corr(method='spearman')

        # Rename categories for better readability
        corr_kg_co2e_ha_matrix.columns = [rename_dict[x] for x in corr_kg_co2e_ha_matrix.columns]
        corr_kg_co2e_ha_matrix.index = [rename_dict[x] for x in corr_kg_co2e_ha_matrix.index]
        corr_kg_co2e_ha_matrix

        # Transform the correlation matrix into long format
        corr_kg_co2e_long = corr_kg_co2e_ha_matrix.stack().reset_index()
        corr_kg_co2e_long.columns = ['model1', 'model2', 'Spearman']

        # Convert to categorical with the new names
        corr_kg_co2e_long['model1'] = pd.Categorical(values=corr_kg_co2e_long['model1'], categories=corr_kg_co2e_long['model1'].unique(), ordered=True)
        corr_kg_co2e_long['model2'] = pd.Categorical(values=corr_kg_co2e_long['model2'], categories=corr_kg_co2e_long['model2'].unique(), ordered=True)

        # Reverse the order of the categories
        model1_categories = corr_kg_co2e_long['model1'].unique()[::-1]
        model2_categories = corr_kg_co2e_long['model2'].unique()

        # Change the data type of the 'Spearman' column to float64
        corr_kg_co2e_long['Spearman'] = corr_kg_co2e_long['Spearman'].astype('float64')

        #plot data as headmap
        heatmap_plot = (
            ggplot(corr_kg_co2e_long, aes(x='model1', y='model2', fill='Spearman')) +
            geom_tile() +
            geom_text(aes(label='round(Spearman, 2)'), size=text_size, color=text_color) +
            scale_fill_gradient(low='blue', high='red', name="Spearman Corr") +
            scale_x_discrete(limits=model1_categories) +
            scale_y_discrete(limits=model2_categories) +
            labs(title=title, x='', y='') +
            theme(axis_text_x=element_text(rotation=0, size=10, ha='center', ma='center'),
                axis_text_y=element_text(rotation=90, size=10, va='center', ha='center', ma='center'),
                axis_ticks=element_blank(),
                panel_background=element_rect(fill='white'),
                plot_title= element_text(ha='center', ma='center', size=title_size),
                figure_size=figure_size)
        )

        # Save the heatmap
        heatmap_plot.save(path.join(chart_dir, file_name), dpi=dpi)
        return heatmap_plot


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # data processing functions
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def farm_size_to_catagorical(self, data:pd.DataFrame, variables:list[str], farm_size_rename:dict[str,str]=None)->pd.DataFrame:
        """convert the farm size variable to ordered catagorical variable

        Args:
            data (pd.DataFrame): data with farm size variable

        Returns:
            pd.DataFrame: farm size data converted to catagorical data
        """
        if farm_size_rename is None:
            farm_size_rename = self._farm_size_replacements 

        for variable in variables:
            data[variable] = data[variable].astype(str).replace(farm_size_rename)
            data[variable] = pd.Categorical(data[variable].astype(str)
                                                        , categories=list(farm_size_rename.values())[1:]
                                                        , ordered=True)

        return data

    def apy_crop_6_class_to_catagorical(self, data:pd.DataFrame, variables:list[str], crop_rename:dict[str:str]=None)->pd.DataFrame:
        """convert the farm size variable to ordered catagorical variable

        Args:
            data (pd.DataFrame): data with farm size variable

        Returns:
            pd.DataFrame: farm size data converted to catagorical data
        """
        if crop_rename is None:
            crop_rename = self.apy_crop_replacements 

        for variable in variables:
            data[variable] = data[variable].replace(crop_rename)
            counts = data[data[variable] != 'Other Crops'][variable].value_counts()
            ordered_categories = list(counts.index) + ['Other Crops'] 
            data[variable] = pd.Categorical(data[variable], categories=ordered_categories, ordered=True)
        return data

