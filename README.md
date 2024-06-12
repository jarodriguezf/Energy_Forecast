<h1 align='center'>Energy Forecast</h1>

![energy_prediction_portada](https://github.com/jarodriguezf/Energy_Forecast/assets/112967594/bc13277f-53d9-4995-abaa-db44a37dce7f)

*El propósito de este proyecto es desarrollar un sistema predictivo tanto para la demanda como para el precio de la energía. Utilizando técnicas de ETL (extracción, transformación y carga de datos) y modelado de datos (modelos predictivos), este proyecto tiene fines didácticos y no es un proyecto profesional o con fines comerciales.*.

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

- Procesamiento de Datos: Se puede procesar datos desde un archivo comprimido, extrayendo la información más relevante en la fase de ETL y cargándola en la base de datos.
- Descarga de Datos: Ejecutar el script en 'db_to_csv' para descargar datos de la BD en formato CSV o Parquet.
- División de Datos: Ejecutar el script en 'process_data' para dividir los datos extraídos en conjuntos de entrenamiento y prueba, y guardar la división en formato Parquet.
- Predicción: Ejecutar las celdas en 'result_predict' para predecir el precio y la demanda del conjunto de prueba.

## Conclusión 🎉

Después de completar este proyecto, he adquirido una mayor comprensión y soltura en problemas relacionados con la ingeniería de datos y la ciencia de datos. Utilizando tecnologías como Apache Airflow, he podido automatizar procesos de carga masiva de datos y emplear redes neuronales recurrentes en series temporales para la predicción de precios.

Espero que este proyecto resulte de interés y sea de ayuda tanto como lo fue para mí.

¡Gracias por tu atención!
