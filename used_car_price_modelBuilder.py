import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
# Fuente de los datos originales (Kaggle): 
# El archivo CSV original se ignora en este repositorio mediante .gitignore

df = pd.read_csv('used_car_dataset.csv')

columnas_deseadas = ['Year', 'kmDriven', 'AskPrice']

df = df.dropna(subset=columnas_deseadas)
df = df[columnas_deseadas]
df['kmDriven'] = df['kmDriven'].str.replace(' km', '', regex=False)
df['kmDriven'] = df['kmDriven'].str.replace(',', '', regex=False)
df['kmDriven'] = df['kmDriven'].astype(float)

df['AskPrice'] = df['AskPrice'].str.replace('₹ ', '', regex=False)
df['AskPrice'] = df['AskPrice'].str.replace(',', '', regex=False)
df['AskPrice'] = df['AskPrice'].astype(float)
df['AskPrice'] = df['AskPrice'] * 0.011

print(df.head())
X = df[['Year', 'kmDriven']] 
y = df['AskPrice']

modelo = LinearRegression()
modelo.fit(X, y)

joblib.dump(modelo, 'modelo_precio_coches.pkl')

print("¡Modelo entrenado y guardado con éxito!")
