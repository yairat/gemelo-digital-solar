import pandas as pd

print("1. Cargando datasets...")
df_gen = pd.read_csv('Plant_1_Generation_Data.csv')
df_sens = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

# Convertir todos los nombres de columnas a mayúsculas y quitar espacios invisibles
df_gen.columns = df_gen.columns.str.strip().str.upper()
df_sens.columns = df_sens.columns.str.strip().str.upper()

print("Columnas en sensores:", df_sens.columns.tolist())

print("2. Procesando fechas...")
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], dayfirst=True)
df_sens['DATE_TIME'] = pd.to_datetime(df_sens['DATE_TIME'])

# Ordenar por tiempo para la unión
df_gen = df_gen.sort_values('DATE_TIME')
df_sens = df_sens.sort_values('DATE_TIME')

print("3. Uniendo datos por tiempo...")
df = pd.merge_asof(df_gen, df_sens, on='DATE_TIME', direction='nearest')

print("Columnas en la tabla unida:", df.columns.tolist())

# Detectar automáticamente las columnas necesarias sin importar ligeras diferencias de nombre
col_irrad = [c for c in df.columns if 'IRRAD' in c][0]
col_temp = [c for c in df.columns if 'MODULE_TEMP' in c or 'MODULE' in c][0]
col_power = [c for c in df.columns if 'DC_POWER' in c][0]

print(f"Usando variables: Irradiancia='{col_irrad}', Temp='{col_temp}', Potencia='{col_power}'")

# 4. Gemelo Digital Teórico basado en Física Fotovoltaica (STC)
P_STC = 14000.0  # Potencia nominal pico en Watts
gamma = 0.004    # Coeficiente de pérdida por temperatura (0.4% por °C)

print("4. Calculando generación esperada (Gemelo Digital)...")
df['DC_POWER_EXPECTED'] = (
    P_STC * (df[col_irrad] / 1000.0) * (1 - gamma * (df[col_temp] - 25.0))
).clip(lower=0)

# 5. Métricas de evaluación
df_valid = df.dropna(subset=[col_power, 'DC_POWER_EXPECTED'])
mae = (df_valid[col_power] - df_valid['DC_POWER_EXPECTED']).abs().mean()
rmse = ((df_valid[col_power] - df_valid['DC_POWER_EXPECTED']) ** 2).mean() ** 0.5

print("\n¡Gemelo Digital Fotovoltaico ejecutado con éxito!")
print(f"Error Absoluto Medio (MAE): {mae:.2f} kW")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} kW")