import pandas as pd
import logging
import re
from datetime import datetime


def format_dt_iso(df_copy):
    # Formatear fecha (dt_iso), actua como id único
    format = "%Y-%m-%d %H:%M"
    df_copy['dt_iso'] = df_copy['dt_iso'].apply(lambda x: re.split(r':00\+[0-9]{2}:00',x)[0]) # eliminamos milisegundos
    df_copy['dt_iso'] = df_copy['dt_iso'].apply(lambda x: datetime.strptime(x, format)) # Convertimos a tipo datetime
    logging.info('Parseo de fechas realizado correctamente.')
    return df_copy


def drop_weather_desc(df_copy):
    # Eliminamos variable weather_description
    df_copy.drop('weather_description', axis=1, inplace=True)
    logging.info('Eliminacion de variable weather_description realizada.')
    return df_copy


def convert_to_celsius(df_copy):
    # Convertimos el grado de las variables referentes a las temperaturas a Celsius (por defecto Kelvin)
    df_copy[['temp','temp_min','temp_max']] = df_copy[['temp','temp_min','temp_max']].apply(lambda x: x - 273.15)
    df_copy[['temp','temp_min','temp_max']] = df_copy[['temp','temp_min','temp_max']].apply(lambda x: round(x, 1))
    logging.info('Conversión de Kelvin a Celsius realizada.')
    return df_copy


def convert_to_kmh(df_copy):
    # Convertimos la velocidad del viento a km/h (por defecto m/s)
    df_copy['wind_speed'] = df_copy['wind_speed'].apply(lambda x: x * 3.6)
    logging.info('Conversión de m/s a km/h realizada.')
    return df_copy


def rename_dt_iso_column(df_copy):
    # Renombramos el nombre de variable 'dt_iso' a 'time_hourly'
    df_copy.rename(columns={'dt_iso':'time_hourly'}, inplace=True)
    logging.info('Renombrado de variables realizado.')
    return df_copy


def float_to_int(df_copy):
    # Casteamos variables float a enteros (necesarias dadas 0.0)
    columns = ['rain_1h','rain_3h','snow_3h']
    df_copy[columns] = df_copy[columns].astype(int)
    logging.info('Parsing a enteros realizado con exito.')
    return df_copy


def drop_duplicated(df_copy):
    df_copy.drop_duplicates(subset=['time_hourly'], inplace=True)
    logging.info('Filas fuplicadas eliminadas.')
    return df_copy


def run_transform_weather_task(df_weather_data):
    """
        Cargamos el dataframe en bruto y procesamos a través de las funciones correspondientes:
            - format_dt_iso: Parseo de string a datetime.
            - drop_weather_desc: Eliminamos variable con descripcion larga (no necesaria).
            - convert_to_celsius: Convertimos las temperaturas dadas en grados Kelvin a Celsius, conservando 1 decimal.
            - convert_to_kmh: Convertimos la velocidad del viento dado en m/s a km/h.
            - rename_dt_iso_column: Renombramos la variable dt_iso (actúa como id).
            - float_to_int: Casteamos variables a int (necesarias dado valores 0)
            - drop_duplicated: eliminamos filas con misma temporalidad
    Args:
        df_weather_data (DataFrame): Dataset en bruto (original)

    Returns:
        DataFrame: Copia del dataset original transformado y validado.
    """
    try:
        df_copy = df_weather_data.copy()
        df_copy=format_dt_iso(df_copy)
        df_copy=drop_weather_desc(df_copy)
        df_copy=convert_to_celsius(df_copy)
        df_copy=convert_to_kmh(df_copy)
        df_copy = rename_dt_iso_column(df_copy)
        df_copy=float_to_int(df_copy)
        df_copy=drop_duplicated(df_copy)
        return df_copy
    except Exception:
        logging.error('Fallo al intentar transformar el dataframe.')