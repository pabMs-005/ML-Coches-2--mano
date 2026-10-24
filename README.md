# Tasador Predictivo de Vehículos de Segunda Mano

Esta aplicación es un sistema predictivo end-to-end diseñado para estimar el valor de mercado de coches de segunda mano. Utiliza un modelo de Machine Learning basado en regresión lineal que procesa el año de fabricación y el kilometraje del vehículo para calcular una tasación en euros en tiempo real.

## Arquitectura del Sistema

El proyecto está dividido en tres componentes principales:

1. **Motor de Predicción (`used_car_price_modelBuilder.py`):** Pipeline de procesamiento de datos y entrenamiento del modelo de Regresión Lineal utilizando `scikit-learn` y `pandas`. Incluye la limpieza de cadenas de texto, conversión de divisas y generación del archivo binario del modelo (`.pkl`).
2. **API REST (`main.py`):** Servidor backend desarrollado con `FastAPI`. Actúa como intermediario entre el modelo matemático y las peticiones externas, validando los datos de entrada mediante `pydantic` y devolviendo la predicción en formato JSON.
3. **Interfaz de Usuario (`frontend.py`):** Aplicación web interactiva construida con `Streamlit` que permite a los usuarios introducir las características del vehículo mediante un formulario y visualizar la tasación al instante.

## Requisitos Previos

Asegúrate de tener Python instalado. Descarga las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt

```

## Despliegue en Local

Para ejecutar el proyecto, es necesario iniciar el servidor de la API y la interfaz web en terminales independientes.

**1. Iniciar la API (Backend):**
Abre una terminal en la raíz del proyecto y ejecuta el servidor con Uvicorn:

```bash
uvicorn main:app --reload

```

La API estará disponible en `http://localhost:8000`. Puedes probar los endpoints directamente en la documentación interactiva generada en `http://localhost:8000/docs`.

**2. Iniciar la Interfaz Web (Frontend):**
Abre una segunda terminal en la misma ruta y ejecuta:

```bash
streamlit run frontend.py

```

Se abrirá automáticamente una pestaña en el navegador con la aplicación funcional.

## Origen de los Datos

El modelo ha sido entrenado utilizando un dataset público de transacciones reales de vehículos, aplicando una conversión de divisas (Rupia India a Euro) en la fase de preprocesamiento. Los datos crudos originales pueden consultarse en [Pega tu enlace a Kaggle aquí].

---

Respecto a tu pregunta sobre el enlace de Kaggle: el **README** es el sitio oficial y más importante de un repositorio para documentar de dónde salen los datos. Sin embargo, a nivel de código limpio, la mejor práctica es añadir ese enlace como comentario en la cabecera de tu archivo `used_car_price_modelBuilder.py`.

Abre ese script y añade esto justo debajo de los `import`:

```python
# Fuente de los datos originales (Kaggle): https://www.kaggle.com/datasets/mohitkumar282/used-car-dataset
# El archivo CSV original se ignora en este repositorio mediante .gitignore

```
