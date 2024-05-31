import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.preprocessing import power_transform, StandardScaler
from sklearn.mixture import GaussianMixture


# - FUNCIONES -

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
    df_copy['generation_wind_onshore'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy



# generation_hydro_water_reservoir, transformación logarítmica
def logarithm_generation_hydro_water_reservoirl(df_copy):
    array1d = df_copy['generation_hydro_water_reservoir'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_hydro_water_reservoir'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy


# generation_hydro_water_reservoir, transformación logarítmica
def logarithm_generation_hydro_run_of_river_and_poundage(df_copy):
    array1d = df_copy['generation_hydro_run_of_river_and_poundage'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_hydro_run_of_river_and_poundage'] = power_transform(array2d, method='yeo-johnson', standardize=False)
    return df_copy



# Estratificamos variable generation_nuclear
def stratify_generation_nuclear(df_copy):
    bins = [0,6000,7000, float('inf')]
    labels = [0,1,2]
    df_copy['generation_nuclear'] = pd.cut(x=df_copy['generation_nuclear'].values,bins=bins, labels=labels)
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
    


# generation_waste, transformación logarítmica
def logarithm_generation_waste(df_copy):
    array1d = df_copy['generation_waste'].values
    array2d = array1d.reshape(-1,1)
    df_copy['generation_waste'] = power_transform(array2d, method='yeo-johnson', standardize=False)
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



def run_automation_process_load_data_prueba(df):
    try:
        df_copy = df.copy()

        df_copy = df_copy[['generation_hydro_pumped_storage_consumption','generation_solar',
                    'generation_fossil_gas','generation_wind_onshore',
                    'generation_fossil_oil','generation_hydro_water_reservoir',
                    'generation_hydro_run_of_river_and_poundage','generation_nuclear',
                    'generation_fossil_hard_coal','price_actual','generation_waste',
                    'generation_fossil_brown_coal_lignite','generation_biomass',
                    'generation_other','generation_other_renewable', 'total_load_actual']]
        
        df_copy=stratify_generation_hydro_pumped_storage_consumption(df_copy)
        df_copy = stratify_generation_solar(df_copy)   
        df_copy=logarithm_generation_wind_onshore(df_copy)   
        df_copy = logarithm_generation_hydro_water_reservoirl(df_copy)
        df_copy = logarithm_generation_hydro_run_of_river_and_poundage(df_copy)
        df_copy = stratify_generation_nuclear(df_copy)
        distance_transform = distance_transform_generation_fossil_hard_coal(df_copy)
        df_copy = distance_transform.kmeans_transform()
        df_copy = logarithm_generation_waste(df_copy)
        df_copy = stratify_generation_fossil_brown_coal_lignite(df_copy)
        df_copy =  gmm_generation_other(df_copy)
        df_copy =  gmm_generation_other_renewable(df_copy)
        return df_copy
    except Exception as e:
        print('Error, al procesar los datos',e)