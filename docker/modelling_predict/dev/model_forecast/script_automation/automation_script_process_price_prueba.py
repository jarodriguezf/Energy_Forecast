import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import power_transform, StandardScaler
from sklearn.mixture import GaussianMixture

# - Funciones -

def lag_price_date(df_copy):
    df_copy['lag_1'] = df_copy['price_actual'].shift(1)
    df_copy['lag_2'] = df_copy['price_actual'].shift(2)
    df_copy['lag_3'] = df_copy['price_actual'].shift(3)
    # Imputar valores nulos media
    df_copy['lag_1'].fillna(df_copy['lag_1'].mean(), inplace=True)
    df_copy['lag_2'].fillna(df_copy['lag_2'].mean(), inplace=True)
    df_copy['lag_3'].fillna(df_copy['lag_3'].mean(), inplace=True)
    return df_copy

def diff_price_date(df_copy):
    df_copy['diff_1'] = df_copy['price_actual'].diff(1)
    df_copy['diff_2'] = df_copy['price_actual'].diff(2)
    # imputar valores nulos media
    df_copy['diff_1'].fillna(df_copy['diff_1'].mean(), inplace=True)
    df_copy['diff_2'].fillna(df_copy['diff_2'].mean(), inplace=True)
    return df_copy

def rolling_mean(df_copy):
    df_copy['rolling_mean_3'] = df_copy['price_actual'].rolling(window=3).mean()
    df_copy['rolling_mean_7'] = df_copy['price_actual'].rolling(window=7).mean()
    # Imputar valores nulos media
    df_copy['rolling_mean_3'].fillna(df_copy['rolling_mean_3'].mean(), inplace=True)
    df_copy['rolling_mean_7'].fillna(df_copy['rolling_mean_7'].mean(), inplace=True)
    return df_copy

# Extraemos de la variable tiempo la fecha en enteros.
def transform_time_hourly(df_copy):
    days=df_copy['time_hourly'].dt.day.values
    hours=df_copy['time_hourly'].dt.hour.values

    df_copy.drop('time_hourly', axis=1, inplace=True)
    df_copy['day']=days
    df_copy['hour']=hours
    return df_copy


# Clase perteneciente a generation_fossil_hard_coal, kmeans y transformacion por distancias al centroide
class distance_transform_generation_fossil_hard_coal:
    def __init__(self, df_copy):
        self.df_copy = df_copy
    
    def scaler(self):
        scaler = StandardScaler()
        self.df_copy['generation_fossil_hard_coal']=scaler.fit_transform(self.df_copy[['generation_fossil_hard_coal']])
        return self.df_copy

    def kmeans_transform(self):
        self.scaler() # Escalamos en funcion
        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(self.df_copy[['generation_fossil_hard_coal']])

        # Extraemos distancias
        distances = kmeans.transform(self.df_copy[['generation_fossil_hard_coal']])

        # Añadimos a la columna los nuevos valores
        self.df_copy['generation_fossil_hard_coal'] = distances
        
        return self.df_copy


def gmm_total_load_actual(df_copy):
    data = df_copy['total_load_actual'].values.reshape(-1, 1)
    
    # Ajustar el Modelo de Mezcla Gaussiana
    gm = GaussianMixture(n_components=2, random_state=0).fit(data)
    labels = gm.predict(data)

    df_copy['total_load_actual'] = labels
    return df_copy


# generation_hydro_water_reservoir, transformación logarítmica
def logarithm_generation_hydro_run_of_river_and_poundage(df_copy):
    array1d = df_copy['generation_hydro_run_of_river_and_poundage'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_hydro_run_of_river_and_poundage'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy


def gmm_generation_other_renewable(df_copy):
    data = df_copy['generation_other_renewable'].values.reshape(-1, 1)
    
    # Ajustar el Modelo de Mezcla Gaussiana
    gm = GaussianMixture(n_components=2, random_state=0).fit(data)
    labels = gm.predict(data)

    df_copy['generation_other_renewable'] = labels
    return df_copy


# generation_waste, transformación logarítmica
def logarithm_generation_waste(df_copy):
    array1d = df_copy['generation_waste'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_waste'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy


def gmm_generation_other(df_copy):
    data = df_copy['generation_other'].values.reshape(-1, 1)
    
    # Ajustar el Modelo de Mezcla Gaussiana
    gm = GaussianMixture(n_components=3, random_state=0).fit(data)
    labels = gm.predict(data)

    df_copy['generation_other'] = labels
    return df_copy


# generation_hydro_water_reservoir, transformación logarítmica
def logarithm_generation_hydro_water_reservoirl(df_copy):
    array1d = df_copy['generation_hydro_water_reservoir'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_hydro_water_reservoir'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy


# generation_wind_onshore, transformación logarítmica
def logarithm_generation_wind_onshore(df_copy):
    array1d = df_copy['generation_wind_onshore'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_wind_onshore'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy


def run_automation_process_price_data_prueba(df):
    try:
        df_copy = df.copy()

        df_copy = df_copy[['time_hourly','generation_fossil_gas','generation_fossil_hard_coal','total_load_actual',
                    'generation_nuclear','generation_hydro_run_of_river_and_poundage',
                    'generation_other_renewable','generation_waste','generation_fossil_oil',
                    'generation_other','generation_hydro_water_reservoir','generation_biomass',
                    'generation_solar','pressure','generation_wind_onshore','generation_hydro_pumped_storage_consumption',
                    'generation_fossil_brown_coal_lignite','temp_min','wind_speed','temp','temp_max',
                    'price_actual']]
        
        df_copy = lag_price_date(df_copy)
        df_copy = diff_price_date(df_copy)
        df_copy = rolling_mean(df_copy)
        df_copy=transform_time_hourly(df_copy)
        distance_transform = distance_transform_generation_fossil_hard_coal(df_copy)
        df_copy = distance_transform.kmeans_transform()
        df_copy = gmm_total_load_actual(df_copy)
        df_copy = logarithm_generation_hydro_run_of_river_and_poundage(df_copy)
        df_copy = gmm_generation_other_renewable(df_copy)
        df_copy=logarithm_generation_waste(df_copy)
        df_copy = gmm_generation_other(df_copy)
        df_copy = logarithm_generation_hydro_water_reservoirl(df_copy)
        df_copy = logarithm_generation_wind_onshore(df_copy)
        return df_copy

    except Exception as e:
        print('Error, al procesar los datos')
        print(e)