FROM apache/airflow:2.2.1

COPY requirements.txt /requirements.txt

RUN pip install --user --upgrade pip

RUN pip install --no-cache-dir --user -r /requirements.txt