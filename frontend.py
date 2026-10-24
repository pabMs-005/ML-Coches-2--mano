import streamlit as st
import requests

st.title("Tasador Predictivo de Coches")
st.subheader("Entrenado con un conjunto de datos del mercado de coches de segunda mano exclusivamente de la India")

# Controles visuales para el usuario
year = st.number_input("Año de fabricación", min_value=1990, max_value=2024, value=2015)
km = st.number_input("Kilómetros recorridos", min_value=0.0, max_value=500000.0, value=50000.0)

# Acción al pulsar el botón
if st.button("Calcular Precio"):
    # Preparamos los datos con la misma estructura (JSON) que exige la clase de FastAPI
    datos = {"Year": int(year), "kmDriven": float(km)}
    
    # Llamamos a nuestro "camarero" (La API local)
    respuesta = requests.post("http://localhost:8000/predecir", json=datos)
    
    # Procesamos y mostramos el resultado
    if respuesta.status_code == 200:
        precio = respuesta.json()["precio_estimado_euros"]
        st.success(f"El precio de mercado estimado es: {precio:.2f} €")
    else:
        st.error("Error de conexión con la API")