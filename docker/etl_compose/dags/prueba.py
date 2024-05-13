from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils import dates

default_args={
    'start_date':dates.days_ago(1)
}

def hello_world():
    print('Hello World!')

with DAG(
    'prueba_dag',
    default_args=default_args,
    schedule_interval =None
) as dag:
    t1 = DummyOperator(task_id='start')
    t2 = PythonOperator(task_id='print_hello', python_callable=hello_world)
    t3 = DummyOperator(task_id='end')

t1 >> t2 >> t3