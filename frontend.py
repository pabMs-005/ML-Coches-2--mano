import streamlit as st
import requests

st.title("Tasador Predictivo de Coches")
st.subheader("Entrenado con un conjunto de datos del mercado de coches de segunda mano exclusivamente de la India")

#year = st.number_input("Año de fabricación", min_value=1990, max_value=2024, value=2015)
year = st.slider(label="Año de fabricación", min_value=1990, max_value=2024, value=2015, help="Desliza la barra para seleccionar un año")
km = st.number_input("Kilómetros recorridos", min_value=0.0, max_value=500000.0, value=50000.0)

if st.button("Calcular Precio"):
    
    datos = {"Year": int(year), "kmDriven": float(km)}
    
    respuesta = requests.post("http://localhost:8000/predecir", json=datos)
    
    if respuesta.status_code == 200:
        precio = respuesta.json()["precio_estimado_euros"]
        st.success(f"El precio de mercado estimado es: {precio:.2f} €")
    else:
        st.error("Error de conexión con la API")

