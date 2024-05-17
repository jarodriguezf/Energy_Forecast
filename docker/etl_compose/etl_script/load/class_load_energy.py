import csv
from datetime import datetime

class TableManager:
    def __init__(self,mysql_hook):
        self.mysql_hook = mysql_hook
    
    def create_table(self):
        conn = self.mysql_hook.get_conn()
        cursor = conn.cursor()
        drop_table_query = "DROP TABLE IF EXISTS energy_data;"
        cursor.execute(drop_table_query)
        create_table_query="""
                            CREATE TABLE IF NOT EXISTS energy_data (
                                time_hourly DATETIME PRIMARY KEY NOT NULL,
                                generation_biomass INT,
                                generation_fossil_brown_coal_lignite  INT,
                                generation_fossil_coal_derived_gas INT,
                                generation_fossil_gas INT,
                                generation_fossil_hard_coal INT,
                                generation_fossil_oil INT,
                                generation_fossil_oil_shale INT,
                                generation_fossil_peat INT,
                                generation_geothermal INT,
                                generation_hydro_pumped_storage_consumption INT,
                                generation_hydro_run_of_river_and_poundage INT,
                                generation_hydro_water_reservoir INT,
                                generation_marine INT,
                                generation_nuclear INT,
                                generation_other INT,
                                generation_other_renewable INT,
                                generation_solar INT,
                                generation_waste INT,
                                generation_wind_offshore INT,
                                generation_wind_onshore INT,
                                total_load_actual INT,
                                price_actual FLOAT(16)
                            )"""
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()

class DataManager:
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
                                    INSERT INTO energy_data 
                                    (time_hourly, generation_biomass, generation_fossil_brown_coal_lignite, generation_fossil_coal_derived_gas,
                                    generation_fossil_gas, generation_fossil_hard_coal,generation_fossil_oil,
                                    generation_fossil_oil_shale,generation_fossil_peat,generation_geothermal,
                                    generation_hydro_pumped_storage_consumption,generation_hydro_run_of_river_and_poundage,
                                    generation_hydro_water_reservoir,generation_marine,generation_nuclear,generation_other,
                                    generation_other_renewable,generation_solar,generation_waste,generation_wind_offshore,
                                    generation_wind_onshore,total_load_actual,price_actual) 
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """
                data = (time_hourly, int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]),
                        int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12]), int(row[13]), int(row[14]),
                        int(row[15]), int(row[16]), int(row[17]), int(row[18]), int(row[19]), int(row[20]), int(row[21]),
                        float(row[22]))
                cursor.execute(insert_query, data)
        conn.commit()
        cursor.close()