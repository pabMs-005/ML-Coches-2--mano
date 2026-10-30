from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
modelo = joblib.load('modelo_precio_coches.pkl') #Debe ser generado con este nombre IMPORTANTE

class CocheInput(BaseModel):
    Year: int
    kmDriven: float
    Owner: int

@app.post("/predecir")
def predecir_precio(datos: CocheInput):
    df_entrada = pd.DataFrame([{"Year": datos.Year, "kmDriven": datos.kmDriven, "Owner": datos.Owner}])
    
    prediccion = modelo.predict(df_entrada)
    
    return {"precio_estimado_euros": prediccion[0]}
