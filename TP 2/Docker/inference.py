import joblib
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import tensorflow as tf
import sys

class NeuralNetwork:
    def __init__(self, epochs=300, batch_size=300, learning_rate=0.005, dropout_rate=0.3, alpha=0.005, n1=25, n2=23, n3=21):
        #inicializo algunos parámetros como épocas, batch_size, learning rate
        #(no son necesarios)
        #se puede agregar la cantidad de capas, la cantidad de neuronas por capa (pensando en hacer una clase que pueda ser usada para cualquier caso)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.model = None

    def build_model(self, input_shape, num_classes):
        # ejemplo con 3 capas ocultas de 10, 15 y 10 neuronas y activación sigmoidea a la salida (multiclase, recibe la cantidad de clases como input, además del input_shape)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.n1, activation='relu', input_shape=(input_shape,), kernel_regularizer=tf.keras.regularizers.l2(self.alpha)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.n2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.alpha)),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.n3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.alpha)),
            tf.keras.layers.Dropout(self.dropout_rate),
            # tf.keras.layers.Dense(20, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(self.alpha)),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')
        ])

        #compilo el modelo con el optimizador Adam, la función de pérdida [categorical_crossentropy, binary_crossentropy, dice, tversky, categorical_focal_crossentropy, binary_focal_crossentropy] y la métrica [accuracy, binary_accuracy, categorical_accuracy, recall, precision]
        #totalmente optimizable e incluso pueden ser parámetros de la función build_model

        metrics = [ tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalseNegatives(name='fn'),
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
              ]

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)#[tf.keras.metrics.OneHotIoU(num_classes=num_classes, target_class_ids=list(range(num_classes)))])

        self.model = model

    def fit(self, X_train, y_train, X_valid, y_valid):
        # simplemente el fit del modelo. Devuelvo la evolución de la función de pérdida, ya que es interesante ver como varía a medida que aumentan las épocas!
        history=self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=self.epochs, batch_size=self.batch_size)
        return history.history['loss'], history.history['val_loss']

    def evaluate(self, X_test, y_test):
        ### evalúo en test
        loss, tp, fp, tn, fn, accuracy, precision, recall, auc, prc = self.model.evaluate(X_test, y_test)
        print(f"test loss: {loss:.4f}")
        print(f"test tp: {tp}")
        print(f"test fp: {fp}")
        print(f"test tn: {tn}")
        print(f"test fn: {fn}")
        print(f"test accuracy: {accuracy:.4f}")
        print(f"test precision: {precision:.4f}")
        print(f"test recall: {recall:.4f}")
        print(f"test auc: {auc:.4f}")
        print(f"test prc: {prc:.4f}")

    def predict(self, X_new):
        ### predicciones
        predictions = self.model.predict(X_new)
        return predictions

    def get_params(self, deep=True):
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'n1': self.n1,
            'n2': self.n2,
            'n3': self.n3,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def save(self, filename):
        ## guarda el modelo
        self.model.save(filename)

    def load(self, filename):
        ## carga el modelo
        self.model = tf.keras.models.load_model(filename)

def lectDataset(path: str):
  '''
  Función que lee el dataset y lo divide en datos de entrada y de salida
  '''
  # Carga del dataset
  df = pd.read_csv(path)

  # Eliminación de datos donde la variable target esté vacía
  index_NaN = df[df['RainTomorrow'].isna()].index
  df.drop(index_NaN, inplace=True)
  df.reset_index(drop=True, inplace=True)

  # Dividimos el dataset en datos de entrada y de salida
  X = df.drop('RainTomorrow', axis=1)
  y = df['RainTomorrow']

  return X, y

def datasetCodif(X: pd.DataFrame):
  '''
  Función que codifica el dataset
  '''

  # Copiamos dataframe
  X_modified = X.copy()

  # Buscamos columnas tipo object
  categoric_cols = X_modified.select_dtypes(exclude='float64').columns.drop(['Date'])

  # Categorizamos columnas
  X_modified[categoric_cols] = X_modified[categoric_cols].astype('category')

  # Convertimos la columna 'Date' a tipo datetime
  X_modified['Date'] = pd.to_datetime(X_modified['Date'])

  # Descomponer la columna "Date"
  X_modified['Month'] = X_modified['Date'].dt.month

  # Eliminar la columna "Date"
  X_modified = X_modified.drop('Date', axis=1)


  ### CODIFICACIÓN DE LA VARIABLE "Location" --> Se Codifica en función de los valores de Latitud y Longitud

  # Creamos el diccionario con las coordenadas (latitud, longitud)
  location_coordinates = {
      'Williamtown': (-32.7925, 151.8397),
      'AliceSprings': (-23.6980, 133.8807),
      'Katherine': (-14.4657, 132.2635),
      'Launceston': (-41.4332, 147.1441),
      'MountGinini': (-35.5292, 148.7722),
      'Dartmoor': (-38.0667, 141.2167),
      'Watsonia': (-37.7161, 145.0831),
      'Portland': (-38.3496, 141.5881),
      'Townsville': (-19.2564, 146.8186),
      'Bendigo': (-36.7570, 144.2794),
  }

  # Creamos las columnas 'Latitude' y 'Longitude' en los datasets de entrenamiento y testeo
  X_modified['Latitude'] = X_modified['Location'].map(lambda x: location_coordinates[x][0] if x in location_coordinates else np.nan)
  X_modified['Longitude'] = X_modified['Location'].map(lambda x: location_coordinates[x][1] if x in location_coordinates else np.nan)

  # Eliminamos la columna "Location"
  X_modified.drop('Location', axis=1, inplace=True)


  ### CODIFICACIÓN DE LA VARIABLE "Month" --> Se Codifica usando la técnica seno-coseno

  # Creamos las columnas 'Month_sin' y 'Month_con' en los datasets de entrenamiento y testeo
  X_modified["Month_sin"] = np.where(pd.isna(X_modified["Month"]), np.NaN, np.sin(2 * np.pi * X_modified["Month"] / 12))
  X_modified["Month_cos"] = np.where(pd.isna(X_modified["Month"]), np.NaN, np.cos(2 * np.pi * X_modified["Month"] / 12))

  # Eliminamos la columna "Month"
  X_modified.drop("Month", axis=1, inplace=True)


  ### CODIFICACIÓN DE LAS VARIABLES  "WindGustDir", "WindDir9am", "WindDir3pm" --> Se Codifican usando la técnica seno-coseno

  # Definimos un diccionario con los ángulos correspondientes a cada dirección
  wind_direction_angles = {
      'N': 0,
      'NNE': 22.5,
      'NE': 45,
      'ENE': 67.5,
      'E': 90,
      'ESE': 112.5,
      'SE': 135,
      'SSE': 157.5,
      'S': 180,
      'SSW': 202.5,
      'SW': 225,
      'WSW': 247.5,
      'W': 270,
      'WNW': 292.5,
      'NW': 315,
      'NNW': 337.5,
  }

  wind_direction_encoded = {}
  for key, value in wind_direction_angles.items():
      wind_direction_encoded[key] = np.sin(np.deg2rad(value)), np.cos(np.deg2rad(value))

  # Creamos las columnas 'WindGustDir_sin' y 'WindGustDir_cos' en los datasets de entrenamiento y testeo
  X_modified['WindGustDir_sin'] = X_modified['WindGustDir'].map(lambda x: wind_direction_encoded[x][0] if x in wind_direction_encoded else np.nan)
  X_modified['WindGustDir_cos'] = X_modified['WindGustDir'].map(lambda x: wind_direction_encoded[x][1] if x in wind_direction_encoded else np.nan)

  # Creamos las columnas 'WindDir9am_sin' y 'WindDir9am_cos' en los datasets de entrenamiento y testeo
  X_modified['WindDir9am_sin'] = X_modified['WindDir9am'].map(lambda x: wind_direction_encoded[x][0] if x in wind_direction_encoded else np.nan)
  X_modified['WindDir9am_cos'] = X_modified['WindDir9am'].map(lambda x: wind_direction_encoded[x][1] if x in wind_direction_encoded else np.nan)

  # Creo las columnas 'WindDir3pm_sin' y 'WindDir3pm_cos' en los datasets de entrenamiento y testeo
  X_modified['WindDir3pm_sin'] = X_modified['WindDir3pm'].map(lambda x: wind_direction_encoded[x][0] if x in wind_direction_encoded else np.nan)
  X_modified['WindDir3pm_cos'] = X_modified['WindDir3pm'].map(lambda x: wind_direction_encoded[x][1] if x in wind_direction_encoded else np.nan)

  # Eliminamos las columnas "WindGustDir", "WindDir9am", "WindDir3pm"
  X_modified.drop("WindGustDir", axis=1, inplace=True)
  X_modified.drop("WindDir9am", axis=1, inplace=True)
  X_modified.drop("WindDir3pm", axis=1, inplace=True)


  ### CODIFICACIÓN DE LA VARIABLE "RainToday"" --> Se implementa codificación binaria
  X_modified['RainToday'] = X_modified['RainToday'].map({'Yes': 1, 'No': 0}, na_action='ignore')

  return X_modified

def imputeDataset(X_modified: pd.DataFrame):
  '''
  Función que imputa el dataset
  '''
  # Copiamos dataset
  X_imputed = X_modified.copy()

  # Creamos los imputadores
  imputador_media = joblib.load('imputador_media.joblib')
  imputador_mediana = joblib.load('imputador_mediana.joblib')
  imputador_knn = joblib.load('imputador_knn.joblib')

  # Lista de variables a imputar
  lista_media = ["Pressure9am", "Pressure3pm"]
  lista_mediana = ["Temp9am", "Temp3pm", "MaxTemp", "MinTemp", "Rainfall", "Evaporation"]

  # Imputamos con la media
  X_imputed[lista_media] = imputador_media.transform(X_imputed[lista_media])

  # Imputamos con la mediana
  X_imputed[lista_mediana] = imputador_mediana.transform(X_imputed[lista_mediana])

  # Imputación kNN
  X_imputed[X_modified.columns] = imputador_knn.transform(X_imputed[X_modified.columns])

  return X_imputed

def scaleDataset(X_imputed: pd.DataFrame):
  '''
  Función que escala el dataset
  '''
  # Copiamos dataset
  X_scaled = X_imputed.copy()

  # Instanciacion del escalador
  scaler = joblib.load('scaler.joblib')

  # Escalado de variables
  X_scaled = scaler.transform(X_scaled)

  return X_scaled

def predictions(X_scaled: pd.DataFrame):
  '''
  Función que realiza las predicciones
  '''
  # Instanciacion del modelo
  nn = NeuralNetwork()

  # Cargar el modelo
  nn.load('red_neuronal.keras')

  # Realizar predicciones
  y_pred = nn.predict(X_scaled)

  # Convertir a 'Yes' y 'No'
  y_pred_conv = np.where(y_pred[:] < 0.5, 'No', 'Yes')
  
  return y_pred_conv

if len(sys.argv) < 2:
    print("Error: Se debe proporcionar la ruta del archivo to_predict.csv como argumento.")
    sys.exit(1)
    
to_predict_path = sys.argv[2]

print("Iniciando inferencia...")
# Lectura del dataset
X, y = lectDataset(to_predict_path)

# Codificación del dataset
X_modified = datasetCodif(X)

# Imputado de datos faltantes
X_imputed = imputeDataset(X_modified)

# Escalado de dataset
X_scaled = scaleDataset(X_imputed)
print("Prediciendo...")
# Realizar predicciones
y_pred_conv = predictions(X_scaled)

# Crear dataframe de y_pred_conv
df_predictions = pd.DataFrame({'RainTomorrow': y_pred_conv.flatten()})
print("Predicciones realizadas con éxito.")
# Guardar dataframe a archivo CSV en la carpeta output
df_predictions.to_csv('output/predictions.csv', index=False)