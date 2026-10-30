import pandas as pd
import logging 

import plotly.express as plx
import keras
import ml_edu.experiment
import ml_edu.results

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
df['AskPrice'] = df['AskPrice'] * 0.011 #Paso a euros

# Columna Owner
mapeo_owners = {
    'first': 1,
    'second': 2,
}

df['Owner'] = df['Owner'].map(mapeo_owners)

logging.debug(f'\n {df.head()}')

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  hidden_layer = keras.layers.Dense(units=64, activation='relu')(concatenated_inputs)
  outputs = keras.layers.Dense(units=1)(hidden_layer)
  model = keras.Model(inputs=inputs, outputs=outputs)

 # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)

  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

logging.debug("SUCCESS: defining linear regression functions complete.")

# Configuración de los hiperparámetros y de los datos de entrada
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.01,
    number_epochs = 60,
    batch_size = 80,
    input_features = ['Year', 'kmDriven', 'Owner']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
car_price_model_v1 = create_model(settings_1, metrics)
experiment_1 = train_model('new_model_v1', car_price_model_v1, df, 'AskPrice', settings_1)

# Historial de errores (RMSE) de las épocas de entrenamiento
historial_error = experiment_1.metrics_history['rmse']


figura = plx.line(
    y=historial_error, 
    title="Evolución del Error cuadrático medio",
    labels={'y': 'Margen de Error (Euros)', 'index': 'Número de Época'}
)

# Grafico para observar resultados, representar una recta de regresion es inviable porque en vez de dos variables 
# Tenemos 4 (3 parametros y 1 que es predecida) asi que se representa la grafica del RMSE
figura.write_html("grafica_entrenamiento.html")

car_price_model_v1.save('modelCar_price_v1.keras')
logging.info("Modelo de Keras entrenado y guardado correctamente.")