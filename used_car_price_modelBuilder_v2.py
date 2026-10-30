import pandas as pd
import logging 
from sklearn.linear_model import LinearRegression
import joblib

logging.basicConfig(
    level=logging.DEBUG,  # Muestra desde el nivel DEBUG en adelante
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Fuente de los datos originales (Kaggle): 
# El archivo CSV original se ignora en este repositorio mediante .gitignore
df = pd.read_csv('used_car_dataset.csv')

# Que opciones hay en el dataset para la columna de propietario
valores = df['Owner'].unique().tolist()
logging.debug(f'Valores unicos de Owner: {valores}\n')

columnas_deseadas = ['Year', 'kmDriven', 'AskPrice', 'Owner']

df = df.dropna(subset=columnas_deseadas)
df = df[columnas_deseadas]
df['kmDriven'] = df['kmDriven'].str.replace(' km', '', regex=False)
df['kmDriven'] = df['kmDriven'].str.replace(',', '', regex=False)
df['kmDriven'] = df['kmDriven'].astype(float)

df['AskPrice'] = df['AskPrice'].str.replace('₹ ', '', regex=False)
df['AskPrice'] = df['AskPrice'].str.replace(',', '', regex=False)
df['AskPrice'] = df['AskPrice'].astype(float)
df['AskPrice'] = df['AskPrice'] * 0.011 #Paso a dolares

# Columna Owner
mapeo_owners = {
    'first': 1,
    'second': 2,
}

df['Owner'] = df['Owner'].map(mapeo_owners)

logging.debug(f'\n {df.head()}')

"""X = df[['Year', 'kmDriven', 'Owner']] 
y = df['AskPrice']

modelo = LinearRegression()
modelo.fit(X, y)

joblib.dump(modelo, 'modelo_precio_coches.pkl')

print("¡Modelo entrenado y guardado con éxito!")"""