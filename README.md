<h1 align='center'>Energy Forecast</h1>

![energy_prediction_portada](https://github.com/jarodriguezf/Energy_Forecast/assets/112967594/bc13277f-53d9-4995-abaa-db44a37dce7f)

*El proposito de este proyecto es realizar un sistema de predictivo tanto de la demanda como del precio de la energía. Usando técnicas de ETL (extracción, transformación y carga de datos) así como el procesamiento y modelado de datos (modelo predictivo). EL proyecto ha sido realizado con fines didacticos, no estamos ante un proyecto profesional o con fines comerciales*.

##  Estructura del proyecto  📁
![estructura_proyecto](https://github.com/jarodriguezf/Energy_Forecast/assets/112967594/33ba9171-a2cd-4288-8f28-c802c0ff7f7a)

- data_raw/data_clean_etl: Contiene todo referente a los datos, tanto en bruto (orígen) como procesados (a través de la fase de ETL).
    
- docker: Contiene tanto la fase de ETL (creada y subida en contenedores de docker), como también, la fase del modelo predictivo.
    - etl_compose: contenedores con varios servícios propios para la extracción, transformación y carga de datos (desde el orígen hasta la base de datos).
      - dags: pipeline de procesamiento de datos.
      - etl_script: archivos usados en la pipeline de procesamiento para realizar las funciones propias de extracción, transformación y carga.
      - archivos propios del docker: usados para crear las imagenes y cargar los contenedores bajo una misma network.
    - modelling_predict: Contiene la baseline como primer calculo de similitud entre las ofertas y los curriculums.
      - data_parquet: Datos extraídos desde la base de datos y divididos en conjunto de entrenamiento y prueba.
      - dev: archivos de desarrollo:
        - db_to_csv: script de carga de datos desde BD a carpeta 'data_parquet'.
        - exploratory: análisis exploratorio de datos.
        - model_forecast: scripts de los modelos predictivos (demanda y precio).
        - process_data: notebooks de procesamiento de datos propios a la fase del modelado (ingeniería de características).
        - result_predict: Resultado final del precio y demanda energética (en función de las fechas dadas).
        - Imagenes del resultado predictivo (tanto de la demanda como el precio).

## Dataset 📄

Los datos han sido extraídos de [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

## Tecnologias usadas 💻

- Python (pandas, sklearn, numpy, tensorflow).
- Apache Airflow.
- Docker
- Git.
- Librerias de visualizacion de datos (matplotlib, seaborn, plotly).

## Funcionamiento del aplicación 🚀

- Podremos procesar datos desde un archivo comprimido, extrayendo así en la fase de ETL la información más relevante, cargándola al final en BD.
- Podremos ejecutar el script de la carpeta 'db_to_csv' para realizar una descarga de los datos de BD y tenerlos en csv (o parquets).
- Así mismo, en la carpeta 'process_data' podremos ejecutar el script para separar los datos extraídos en train y test, cargando la división en parquets.
- Unicamente ejecutando las celdas de 'result_predict' podremos predecir el precio y la demanda del conjunto de prueba dados.

## Conclusión 🎉

Después de completar este proyecto, he adquirido una comprensión mayor y una soltura ante problemas referente al campo del data engineer y data science. Realizando transformaciones con tecnologías como Apache Airflow he podido ver el potencial y comodidad para automatizar procesos de carga masiva de datos, así como la capacidad para emplear redes neuronales recurrentes en series temporales (predicción del precio). 

Espero que este proyecto resulte de interés y ayude tanto como me resulto a mi.

Gracias por ver!!!
