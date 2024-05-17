import csv
from datetime import datetime

class TableManager_weather:
    def __init__(self,mysql_hook):
        self.mysql_hook = mysql_hook
    
    def create_table(self):
        conn = self.mysql_hook.get_conn()
        cursor = conn.cursor()

        drop_table_query = "DROP TABLE IF EXISTS weather_data;"
        cursor.execute(drop_table_query)
        create_table_query="""
                            CREATE TABLE IF NOT EXISTS weather_data (
                                id INT PRIMARY KEY AUTO_INCREMENT NOT NULL,
                                time_hourly DATETIME,
                                city_name VARCHAR(20),
                                temp  FLOAT(16),
                                temp_min FLOAT(16),
                                temp_max FLOAT(16),
                                pressure INT,
                                humidity INT,
                                wind_speed FLOAT(16),
                                wind_deg INT,
                                rain_1h INT,
                                rain_3h INT,
                                snow_3h INT,
                                clouds_all INT,
                                weather_id INT,
                                weather_main VARCHAR(20),
                                weather_icon VARCHAR(5)
                            )"""
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()


class DataManager_weather:
    def __init__(self, mysql_hook):
        self.mysql_hook = mysql_hook

    def insert_data(self, csv_file_path):
        conn = self.mysql_hook.get_conn()
        cursor = conn.cursor()
        
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                time_hourly = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                insert_query = """
                                    INSERT INTO weather_data 
                                    (time_hourly, city_name, temp, temp_min, temp_max, pressure, humidity, 
                                    wind_speed, wind_deg, rain_1h, rain_3h, snow_3h, clouds_all, weather_id, 
                                    weather_main, weather_icon) 
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """
                data = (time_hourly, row[1], float(row[2]), float(row[3]), float(row[4]), int(row[5]), int(row[6]), float(row[7]),
                    int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12]), int(row[13]), row[14], row[15])
                cursor.execute(insert_query, data)
        conn.commit()
        cursor.close()