import pandas as pd
import folium
from folium.plugins import HeatMapWithTime

class MapCreator:
    def __init__(self, lat_lon_path):
        self.lat_lon_df = pd.read_excel(lat_lon_path)

    def load_data(self):
        df_23 = pd.read_excel('project_data/extra/education_2023.xlsx', sheet_name=2)
        df_22 = pd.read_excel('project_data/extra/education_2022.xlsx', sheet_name=2)
        df_21 = pd.read_excel('project_data/extra/education_2021.xlsx', sheet_name=2)
        df_20 = pd.read_excel('project_data/extra/education_2020.xlsx', sheet_name=2)
        df_19 = pd.read_excel('project_data/extra/education_2019.xlsx', sheet_name=2)

        dataframes = [df_19, df_20, df_21, df_22, df_23]
        years = [2019, 2020, 2021, 2022, 2023]

        return dataframes, years

    def create_combined_df(self, dataframes, years, column_name):
        processed_dfs = []

        for df, year in zip(dataframes, years):
            temp_df = df[['area_name', 'education_category', column_name]].copy()
            temp_df['year'] = year
            processed_dfs.append(temp_df)

        final_df = pd.concat(processed_dfs)
        final_df = final_df.sort_values(by=['area_name', 'education_category', 'year']).reset_index(drop=True)

        # Adding lat-long values
        lat_lon_df = self.lat_lon_df[['area_name', 'latitude', 'longitude']]
        final_df['area_name'] = final_df['area_name'].str.replace('nonmetropolitan area', '').str.strip()
        final_df = pd.merge(final_df, lat_lon_df, on='area_name', how='left')
        final_df[column_name] = final_df[column_name].fillna(0)

        return final_df

    def preprocess_for_maps(self, final_df, degree_name, column_name, normalize=False):
        data = final_df[final_df['education_category'] == degree_name]
        
        if normalize:
            max_val = data[column_name].max()
            if max_val > 0:
                data[column_name] = data[column_name] / max_val

        num_years = data['year'].unique()
        outer_list = []

        for year in num_years:
            list_data = data[data['year'] == year]
            inner_list = []

            for lat, long, val in zip(list_data['latitude'], list_data['longitude'], list_data[column_name]):
                inner_list.append([lat, long, val])

            outer_list.append(inner_list)

        return outer_list

    def heatmap_with_time(self, data, degree_name, column_name, normalize=False):
        data_list = self.preprocess_for_maps(data, degree_name, column_name, normalize)
        timestamps = data['year'].unique()

        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=7)
        hm = HeatMapWithTime(data_list, index=timestamps.tolist())
        hm.add_to(m)

        return m
