import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. Cargar los datos
# Asegúrate de subir el .csv a Colab o a tu carpeta y pon aquí el nombre exacto
df = pd.read_csv('used_car_dataset.csv')

# Para evitar errores en tu primer intento, nos quedamos solo con las filas 
# que no tengan datos vacíos (NaN) en las columnas que vamos a usar
columnas_deseadas = ['Year', 'kmDriven', 'AskPrice']

df = df.dropna(subset=columnas_deseadas)
df = df[columnas_deseadas]
df['kmDriven'] = df['kmDriven'].str.replace(' km', '', regex=False)
df['kmDriven'] = df['kmDriven'].str.replace(',', '', regex=False)
df['kmDriven'] = df['kmDriven'].astype(float)

# Limpiar la columna del precio:
# Quitamos el símbolo "₹ ", quitamos las comas, y lo convertimos a número decimal
df['AskPrice'] = df['AskPrice'].str.replace('₹ ', '', regex=False)
df['AskPrice'] = df['AskPrice'].str.replace(',', '', regex=False)
df['AskPrice'] = df['AskPrice'].astype(float)
df['AskPrice'] = df['AskPrice'] * 0.011

print(df.head())
# 2. Separar características (X) y el objetivo a predecir (y)
# IMPORTANTE: Cambia 'Year', 'Mileage' y 'Price' por los nombres exactos de las columnas de tu CSV
X = df[['Year', 'kmDriven']] 
y = df['AskPrice']

# 3. Crear y entrenar el modelo (El Cerebro)
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Guardar el modelo entrenado en tu ordenador
# La extensión .pkl (pickle) es el estándar para guardar modelos en Python
joblib.dump(modelo, 'modelo_precio_coches.pkl')

print("¡Modelo entrenado y guardado con éxito!")