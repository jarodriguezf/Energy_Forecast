from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import datetime as dt
import pandas as pd
from etl_script.extract.extract_etl_data import run_extract_task


default_args = {
    'owner':'Jorge Rodríguez',
    'start_date': dt.datetime(2024,5,13)
}


def print_values_extract():
    df_energy_data, df_weather_data = run_extract_task()
    print(df_energy_data.head())
    print(df_weather_data.head())


with DAG(

    'energy_data_flow',
    default_args=default_args,
    schedule_interval=None

) as dag:
    
    start_ = DummyOperator(task_id='start')

    extract_data = PythonOperator(task_id = 'extract', python_callable=print_values_extract)

    end_ = DummyOperator(task_id = 'end')

start_ >> extract_data >> end_