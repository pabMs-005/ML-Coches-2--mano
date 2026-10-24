from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inicializar el "camarero" (La aplicación)
app = FastAPI()

# 2. Cargar el "cocinero" en la memoria de la API
# Se hace aquí arriba para que el modelo solo se cargue una vez al encender el servidor, ahorrando recursos.
modelo = joblib.load('modelo_precio_coches.pkl')

# 3. Definir el menú (Estructura de validación)
# Pydantic obliga a que la petición contenga exactamente estos datos. Si faltan, la API rechaza el cálculo automáticamente.
class CocheInput(BaseModel):
    Year: int
    kmDriven: float

# 4. Crear la ruta (El Endpoint)
# Le decimos a la API que cuando reciba un POST en la ruta "/predecir", ejecute el código de abajo.
@app.post("/predecir")
def predecir_precio(datos: CocheInput):
    # Transformamos el JSON entrante a un formato que tu modelo de Scikit-Learn entienda (un DataFrame de Pandas)
    df_entrada = pd.DataFrame([{"Year": datos.Year, "kmDriven": datos.kmDriven}])
    
    # El modelo hace el cálculo matemático exacto que aprendió en la Fase 1
    prediccion = modelo.predict(df_entrada)
    
    # La API devuelve el resultado al usuario en formato JSON
    return {"precio_estimado_euros": prediccion[0]}