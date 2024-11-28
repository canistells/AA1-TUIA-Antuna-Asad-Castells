# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo requirements.txt a la imagen
COPY requirements.txt .

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /app/output

# Copiar los archivos de modelado, como el joblib y el CSV de predicción
COPY red_neuronal.keras /app/red_neuronal.keras
COPY imputador_knn.joblib /app/imputador_knn.joblib
COPY imputador_media.joblib /app/imputador_media.joblib
COPY imputador_mediana.joblib /app/imputador_mediana.joblib
COPY scaler.joblib /app/scaler.joblib
COPY inference.py /app/inference.py

# Establecer el punto de entrada y el comando por defecto
ENTRYPOINT ["python", "inference.py"]
CMD ["to_predict.csv"]