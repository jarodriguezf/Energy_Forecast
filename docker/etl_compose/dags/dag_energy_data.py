from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
import logging
import datetime as dt
import pandas as pd
from etl_script.extract.extract_etl_data import run_extract_task
from etl_script.transform.transform_energy_data import run_transform_energy_task
from etl_script.transform.transform_weather_data import run_transform_weather_task
from etl_script.load.class_load_energy import TableManager, DataManager

default_args = {
    'owner':'Jorge Rodríguez',
    'start_date': dt.datetime(2024,5,13)
}


def return_extract_data():
    df_energy_data, df_weather_data = run_extract_task()
    df_energy_data.to_csv('data_clean_etl/return_extract_dataframes/df_energy_data.csv', index=False)
    df_weather_data.to_csv('data_clean_etl/return_extract_dataframes/df_weather_data.csv', index=False)
    logging.info('CSVs guardados correctamente en (data_clean_etl/return_extract_dataframes/)')



def transform_energy_data():
    df_energy_data = pd.read_csv('data_clean_etl/return_extract_dataframes/df_energy_data.csv')
    transformed_energy_data = run_transform_energy_task(df_energy_data)
    transformed_energy_data.to_csv('data_clean_etl/return_transform_dataframes/transformed_energy_data.csv',index=False)
    logging.info('CSV guardado correctamente en (data_clean_etl/return_transform_dataframes/)')


def transform_weather_data():
    df_weather_data=pd.read_csv('data_clean_etl/return_extract_dataframes/df_weather_data.csv')
    transformed_weather_data = run_transform_weather_task(df_weather_data)
    transformed_weather_data.to_csv('data_clean_etl/return_transform_dataframes/transformed_weather_data.csv',index=False)
    logging.info('CSV guardado correctamente en (data_clean_etl/return_transform_dataframes/)')


def load_energy_data():
    mysql_hook = MySqlHook(mysql_conn_id='mariadb_default')
    table_manager = TableManager(mysql_hook)
    data_manager = DataManager(mysql_hook)

    table_manager.create_table()
    logging.info('Tabla energy_data creada correctamente')
    data_manager.insert_data(csv_file_path='data_clean_etl/return_transform_dataframes/transformed_energy_data.csv')
    logging.info('Datos insertado en bd correctamente.')

with DAG(
    'energy_data_flow',
    default_args=default_args,
    schedule_interval=None

) as dag:
    
    start_ = DummyOperator(task_id='start')

    extract_data = PythonOperator(task_id = 'extract_data', python_callable=return_extract_data)

    transform_energy = PythonOperator(task_id='transform_energy', python_callable=transform_energy_data, provide_context=True)

    transform_weather = PythonOperator(task_id='transform_weather', python_callable=transform_weather_data, provide_context=True)

    load_energy_db = PythonOperator(task_id='load_energy', python_callable=load_energy_data)

    load_weather_db = DummyOperator(task_id = 'load_weather')

    end_ = DummyOperator(task_id = 'end')

start_ >> extract_data >> [transform_energy, transform_weather]
transform_energy >> load_energy_db >> end_
transform_weather >> load_weather_db >> end_