# Librerias
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, power_transform
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# -- Funciones --

# Eliminar variables con alta frecuencia de 0
def drop_var_energy_0(df_copy):
    df_copy.drop(['generation_fossil_coal_derived_gas',
                  'generation_fossil_oil_shale','generation_fossil_peat',
                  'generation_geothermal','generation_marine',
                  'generation_wind_offshore'], axis=1, inplace=True)
    return df_copy


# generation_fossil_brown_coal_lignite, estratificar en dos categorias 0 y 1.
def stratify_generation_fossil_brown_coal_lignite(df_copy):
    """ Si los valores de la variable son mayores que 200, el valor se modifica a 1.
        En caso contrario, se mantienen en 0.
    Args:
        df_copy (Dataframe): copia de dataframe sin modificar valores

    Returns:
        Dataframe: dataframe modificado
    """
    df_copy['generation_fossil_brown_coal_lignite']=df_copy['generation_fossil_brown_coal_lignite'].apply(lambda x: 1 if x >= 200 else 0)
    return df_copy


# generation_fossil_gas, eliminar valores atípicos
def outliers_generation_fossil_gas(df_copy):
    # Deteccion de atípicos 
    q1 = df_copy['generation_fossil_gas'].quantile(0.25)
    q3 = df_copy['generation_fossil_gas'].quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr

    print('Antes de eliminar atípicos:',len(df_copy['generation_fossil_gas']))
    
    # Filtramos en pandas extrayendo los valores entre los quantiles
    filtered_df = df_copy[(df_copy['generation_fossil_gas'] >= Lower_tail)&(df_copy['generation_fossil_gas'] <= Upper_tail)]

    print('Despues de eliminar atípicos:',len(filtered_df))
    return filtered_df


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
    

# generation_fossil_oil, eliminar valores atípicos
def outliers_generation_fossil_oil(df_copy):
    # Deteccion de atípicos 
    q1 = df_copy['generation_fossil_oil'].quantile(0.25)
    q3 = df_copy['generation_fossil_oil'].quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr

    print('Antes de eliminar atípicos:',len(df_copy['generation_fossil_oil']))
    
    # Filtramos en pandas extrayendo los valores entre los quantiles
    filtered_df = df_copy[(df_copy['generation_fossil_oil'] >= Lower_tail)&(df_copy['generation_fossil_oil'] <= Upper_tail)]

    print('Despues de eliminar atípicos:',len(filtered_df))
    return filtered_df


# generation_hydro_pumped_storage_consumption, estratificar en dos categorias 0 y 1.
def stratify_generation_hydro_pumped_storage_consumption(df_copy):
    """ Si los valores de la variable son mayores que 0, el valor se modifica a 1.
        En caso contrario, se mantienen en 0.
    Args:
        df_copy (Dataframe): copia de dataframe sin modificar valores

    Returns:
        Dataframe: dataframe modificado
    """
    df_copy['generation_hydro_pumped_storage_consumption']=df_copy['generation_hydro_pumped_storage_consumption'].apply(lambda x: 1 if x > 0 else 0)
    return df_copy


# generation_hydro_water_reservoir, transformación logarítmica
def logarithm_generation_hydro_water_reservoirl(df_copy):
    array1d = df_copy['generation_hydro_water_reservoir'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_hydro_water_reservoir'] = power_transform(array2d, method='box-cox', standardize=False)
    return df_copy


# generation_nuclear, estratificar en dos categorias 0, 1, 2, 3.
def stratify_generation_hydro_pumped_storage_consumption(df_copy):
    """ Modus operandi:.
        - el valor sera 0, si los MW son menores que 5000.
        - el valor sera 1, si los MW son mayores o iguales que 5000 y menores de 6000.
        - el valor sera 2, si los MW son mayores o iguales que 6000 y menores de 7000.
        - el valor sera 3, si los MW son mayores o iguales a 7000.
    Args:
        df_copy (Dataframe): copia de dataframe sin modificar valores

    Returns:
        Dataframe: dataframe modificado
    """
    bins = [0, 5000, 6000, 7000, float('inf')]
    labels = [0, 1, 2, 3]
    df_copy['generation_nuclear'] = pd.cut(df_copy['generation_nuclear'], bins=bins, labels=labels, right=False)
    return df_copy


def gmm_generation_other(df_copy):
    """Convertir valores de variable a cluster según multiples modos, mediante gaussian mixture.

    Args:
        df_copy (DataFrame): dataframe si modificar

    Returns:
        DataFrame: dataframe con valores modificado
    """
    # extraer columna
    data = df_copy['generation_other'].values
    data = data.reshape(-1,1)

    # entrenar
    gmm = GaussianMixture(n_components=3, n_init=10)
    gmm.fit(data)

    # predecir probabilidades
    values = gmm.predict_proba(data).round(3)
    values = np.argmax(values, axis=1)

    df_copy['generation_other'] = values
    return df_copy


def gmm_generation_other_renewable(df_copy):
    """Convertir valores de variable a cluster según multiples modos, mediante gaussian mixture.

    Args:
        df_copy (DataFrame): dataframe si modificar

    Returns:
        DataFrame: dataframe con valores modificado
    """
    # extraer columna
    data = df_copy['generation_other_renewable'].values
    data = data.reshape(-1,1)

    # entrenar
    gmm = GaussianMixture(n_components=2, n_init=10)
    gmm.fit(data)

    # predecir probabilidades
    values = gmm.predict_proba(data).round(3)
    values = np.argmax(values, axis=1)

    df_copy['generation_other_renewable'] = values
    return df_copy


# generation_solar, estratificar en dos categorias 0 y 1.
def stratify_generation_solar(df_copy):
    """ Si los valores de la variable son menores que 600, el valor se modifica a 0.
        En caso contrario, se mantienen en 1.
    Args:
        df_copy (Dataframe): copia de dataframe sin modificar valores

    Returns:
        Dataframe: dataframe modificado
    """
    df_copy['generation_solar']=df_copy['generation_solar'].apply(lambda x: 0 if x < 600 else 1)
    return df_copy


# generation_wind_onshore, transformación logarítmica
def logarithm_generation_wind_onshore(df_copy):
    array1d = df_copy['generation_wind_onshore'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_wind_onshore'] = power_transform(array2d, method='box-cox', standardize=False)
    return df_copy


# Funcion que extrae los grupos de generation_biomass
def transform_generation_biomass_gmm(filtered_df):
    data = filtered_df['generation_biomass'].values
    data = data.reshape(-1,1)

    # entrenar
    gmm = GaussianMixture(n_components=3, n_init=10)
    gmm.fit(data)

    # predecir probabilidades
    values = gmm.predict_proba(data).round(3)
    values = np.argmax(values, axis=1)

    filtered_df['generation_biomass'] = values
    return filtered_df


# Funcion que elimina atípicos de generation_biomass
def transform_generation_biomass_out(df_copy):
    q1 = df_copy['generation_biomass'].quantile(0.25)
    q3 = df_copy['generation_biomass'].quantile(0.75)
    iqr = q3-q1
    Lower_tail = q1 - 1.5 * iqr
    Upper_tail = q3 + 1.5 * iqr

    print('Antes de eliminar atípicos:',len(df_copy['generation_biomass']))
        
    # Filtramos en pandas extrayendo los valores entre los quantiles
    filtered_df = df_copy[(df_copy['generation_biomass'] >= Lower_tail)&(df_copy['generation_biomass'] <= Upper_tail)]

    print('Despues de eliminar atípicos:',len(filtered_df))

    return transform_generation_biomass_gmm(filtered_df)


# Eliminamos variables weather no necesarias
def drop_var_weather(df_copy):
    df_copy.drop(['pressure','weather_id','rain_1h','rain_3h','snow_3h'], axis = 1, inplace=True)
    return df_copy


# Reducimos dimensionalidad de las 3 variables relacionadas con la temperatura (evitar problemas de multicolinealidad)
def pca_temperatures_features(df_copy):
    temp_columns = df_copy[['temp','temp_min','temp_max']]

    pca = PCA(n_components=0.95)
    reduced_val=pca.fit_transform(temp_columns)

    df_copy.drop(['temp','temp_min','temp_max'], axis=1, inplace=True)

    df_copy['temp_pca'] = reduced_val
    return df_copy


def logarithm_humidity(df_copy):
    array1d = df_copy['humidity'].values
    array2d = array1d.reshape(-1,1)
    df_copy['humidity'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy


# Estratificar wind_speed 0 o 1
def stratify_wind_speed(df_copy):
    df_copy['wind_speed'] = df_copy['wind_speed'].apply(lambda x: 0 if x<10 else 1)
    return df_copy


# Estratificar stratify_wind_deg en funcion de rangos
def stratify_wind_deg(df_copy):
    bins = [0, 1, 50, 150, 250, float('inf')]
    labels = [0, 1, 2, 3, 4]
    df_copy['wind_deg'] = pd.cut(df_copy['wind_deg'], bins=bins, labels=labels, right=False)
    return df_copy


# Estratificar stratify_wind_deg en funcion de rangos
def stratify_clouds_all(df_copy):
    bins = [0, 1, float('inf')]
    labels = [0,1]
    df_copy['clouds_all'] = pd.cut(df_copy['clouds_all'], bins=bins, labels=labels, right=False)
    return df_copy


def transform_weather_main_out(df_copy):
    filtered_df = df_copy[(df_copy['weather_main']=='clear')|
                          (df_copy['weather_main']=='clouds')|
                          (df_copy['weather_main']=='rain')|
                          (df_copy['weather_main']=='mist')]
    
    return filtered_df


def run_script_processing_data(df):
    df_copy = df.copy()

    try:
        df_copy = drop_var_energy_0(df_copy)
        df_copy = stratify_generation_fossil_brown_coal_lignite(df_copy)
        df_copy=outliers_generation_fossil_gas(df_copy)
        distance_transform = distance_transform_generation_fossil_hard_coal(df_copy)
        df_copy = distance_transform.kmeans_transform()
        df_copy=outliers_generation_fossil_oil(df_copy)
        df_copy=stratify_generation_hydro_pumped_storage_consumption(df_copy)
        df_copy = logarithm_generation_hydro_water_reservoirl(df_copy)
        df_copy=stratify_generation_hydro_pumped_storage_consumption(df_copy)
        df_copy =  gmm_generation_other(df_copy)
        df_copy =  gmm_generation_other_renewable(df_copy)
        df_copy=stratify_generation_solar(df_copy)
        df_copy=logarithm_generation_wind_onshore(df_copy)
        df_copy=transform_generation_biomass_out(df_copy)
        df_copy=drop_var_weather(df_copy)
        df_copy=pca_temperatures_features(df_copy)
        df_copy = logarithm_humidity(df_copy)
        df_copy = stratify_wind_speed(df_copy)
        df_copy = stratify_wind_deg(df_copy)
        df_copy=stratify_clouds_all(df_copy)
        df_copy = transform_weather_main_out(df_copy)

        return df_copy
    except Exception:
        print('Error, al intentar procesar los datos')