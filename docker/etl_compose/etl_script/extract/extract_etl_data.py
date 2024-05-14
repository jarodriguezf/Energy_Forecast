import pandas as pd
from zipfile import ZipFile, BadZipFile
import logging


# carga de datos (origen)
path_data = 'data_raw/data_electric_pricing.zip'


def load_data_energy():
    # Abrimos el archivo comprimido (energy_dataset.csv) y almacenamos los csv necesarios en dataframes.
    with ZipFile(path_data) as myzip:
        with myzip.open('energy_dataset.csv') as myfile_energy:
            df_energy_data = pd.read_csv(myfile_energy, encoding='utf-8')

    return df_energy_data
    

def load_data_weather():
    # Abrimos el archivo comprimido (weather_features.csv) y almacenamos los csv necesarios en dataframes.
 
    with ZipFile(path_data) as myzip:
        with myzip.open('weather_features.csv') as myfile_weather:
            df_weather_data = pd.read_csv(myfile_weather, encoding='utf-8')

    return df_weather_data
    

# Funcion principal
def run_extract_task():
    try:
        df_energy_data = load_data_energy()
        df_weather_data = load_data_weather()

        logging.info('Datos energy_dataset extraidos correctamente.')
        logging.info('Datos weather_features extraidos correctamente.')

        return df_energy_data, df_weather_data
    except FileNotFoundError:
        logging.error(f'No se ha encontrado la ruta {path_data}')
    except BadZipFile:
        logging.error(f'El archivo data_electric_pricing.zip no es un archivo ZIP válido o está dañado.')