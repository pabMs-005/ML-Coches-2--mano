from fastapi import FastAPI
from pydantic import BaseModel
import keras
import numpy as np

app = FastAPI()
modelo = keras.models.load_model('modelCar_price_v1.keras') #Debe ser generado con este nombre IMPORTANTE

class CocheInput(BaseModel):
    Year: int
    kmDriven: float
    Owner: int

@app.post("/predecir")
def predecir_precio(datos: CocheInput):
    datos_entrada = {
        'Year': np.array([[datos.Year]]),
        'kmDriven': np.array([[datos.kmDriven]]),
        'Owner': np.array([[datos.Owner]])
    }   

    prediccion_raw = modelo.predict(datos_entrada)
    
    precio_estimado = float(prediccion_raw[0][0])
    return {"precio_estimado_euros": precio_estimado}
