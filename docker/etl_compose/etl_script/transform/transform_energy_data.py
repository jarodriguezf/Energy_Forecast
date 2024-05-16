import pandas as pd
from datetime import datetime
import re
import logging


def drop_empty_features(df_copy):
    # Eliminar atributos sin registros (todo es nulo)
    df_copy.dropna(axis=1, how='all', inplace=True)
    logging.info('Eliminación de variables sin datos realizada correctamente.')
    return df_copy


def drop_predict_TSO(df_copy):
    # Eliminar atributos de predicciones propias de los TSO (Transmission Service Operator)
    df_copy.drop(['forecast solar day ahead','forecast wind onshore day ahead',
                'total load forecast', 'price day ahead'], axis = 1, inplace=True)
    logging.info('Eliminacion de variables predictivas realizado correctamente.')
    return df_copy


def drop_duplicates(df_copy):
    # Eliminamos registros duplicados
    df_copy.drop_duplicates(inplace=True)
    logging.info('Eliminación de registros duplicados realizado correctamente.')
    return df_copy


def format_time_id(df_copy):
    # Formatear fecha (time), actua como id único
    format = "%Y-%m-%d %H:%M"
    df_copy['time'] = df_copy['time'].apply(lambda x: re.split(r':00\+[0-9]{2}:00',x)[0]) # eliminamos milisegundos
    df_copy['time'] = df_copy['time'].apply(lambda x: datetime.strptime(x, format)) # Convertimos a tipo datetime
    logging.info('Parseo de fechas realizado correctamente.')
    return df_copy


def rename_columns(df_copy):
    # Renombrar columnas
    renames_columns = ['time_hourly','generation_biomass','generation_fossil_brown_coal/lignite',
                    'generation_fossil_coal-derived_gas','generation_fossil_gas','generation_fossil_hard_coal',
                    'generation_fossil_oil','generation_fossil_oil_shale','generation_fossil_peat',
                    'generation_geothermal','generation_hydro_pumped_storage_consumption',
                    'generation_hydro_run-of-river_and_poundage', 'generation_hydro_water_reservoir',
                    'generation_marine', 'generation_nuclear','generation_other','generation_other_renewable',
                    'generation_solar','generation_waste','generation_wind_offshore','generation_wind_onshore',
                    'total_load_actual', 'price_actual']

    dict_colums = {old_name: new_name for old_name, new_name in zip(df_copy.columns, renames_columns)}
    df_copy.rename(columns=dict_colums, inplace=True)
    
    logging.info('Renombrado de variables realizado con exito.')
    return df_copy


def convert_to_int(df_copy):
    # Castear de float a enteros
    columns = ['generation_biomass','generation_fossil_brown_coal/lignite',
                'generation_fossil_coal-derived_gas','generation_fossil_gas','generation_fossil_hard_coal',
                'generation_fossil_oil','generation_fossil_oil_shale','generation_fossil_peat',
                'generation_geothermal','generation_hydro_pumped_storage_consumption',
                'generation_hydro_run-of-river_and_poundage', 'generation_hydro_water_reservoir',
                'generation_marine', 'generation_nuclear','generation_other','generation_other_renewable',
                'generation_solar','generation_waste','generation_wind_offshore','generation_wind_onshore',
                'total_load_actual']
    # Rellenamos NaN con 0
    df_copy[columns] = df_copy[columns].fillna(0)
    df_copy[columns] = df_copy[columns].astype(int)

    logging.info('Parsing a enteros realizado con exito.')
    return df_copy


def run_transform_energy_task(df_energy_data):
    """
        Cargamos el dataframe en bruto y procesamos a través de las funciones correspondientes:
            - drop_empty_features: Eliminar atributos con registros vacios.
            - drop_predict_TSO: Eliminar atributos predictivos de origen.
            - drop_duplicates: Eliminamos registros duplicados.
            - format_time_id: Parseo de string a datetime.
            - rename_columns: Renombramos los atributos.
            - convert_to_int: Convertimos floats a enteros.

    Args:
        df_energy_data (DataFrame): dataframe en bruto (original)

    Returns:
        DataFrame: dataframe transformado
    """
    
    try:
        df_copy = df_energy_data.copy()
        logging.warn('Copia de dataset realizada para ejecutar tareas de transformación.')

        df_copy=drop_empty_features(df_copy)
        df_copy=drop_predict_TSO(df_copy)
        df_copy=drop_duplicates(df_copy)
        df_copy=format_time_id(df_copy)
        df_copy=rename_columns(df_copy)
        df_copy=convert_to_int(df_copy)
        return df_copy
    except Exception:
        logging.error('Fallo al intentar transformar el dataframe.')

