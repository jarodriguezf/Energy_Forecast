from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import logging
import datetime as dt
import pandas as pd
from etl_script.extract.extract_etl_data import run_extract_task
from etl_script.transform.transform_energy_data import run_transform_energy_task


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
    print(transformed_energy_data.head())


def transform_weather_data():
    df_weather_data=pd.read_csv('data_clean_etl/return_extract_dataframes/df_weather_data.csv')
    print(df_weather_data.head(3))


with DAG(
    'energy_data_flow',
    default_args=default_args,
    schedule_interval=None

) as dag:
    
    start_ = DummyOperator(task_id='start')

    extract_data = PythonOperator(task_id = 'extract_data', python_callable=return_extract_data)

    transform_energy = PythonOperator(task_id='transform_energy', python_callable=transform_energy_data, provide_context=True)

    transform_weather = PythonOperator(task_id='transform_weather', python_callable=transform_weather_data, provide_context=True)

    end_ = DummyOperator(task_id = 'end')

start_ >> extract_data >> [transform_energy, transform_weather] >> end_ 