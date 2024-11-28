@echo off

REM Run.bat - Ejecutar la imagen Docker con el archivo de predicción

SET TO_PREDICT_FILE=to_predict.csv
SET PREDICTIONS_FILE=output\predictions.csv

IF NOT EXIST "%TO_PREDICT_FILE%" (
    echo El archivo to_predict.csv no se encontro. Asegurate de que este en la misma carpeta que este script.
    EXIT /B 1
)

echo Ejecutando la imagen Docker con el archivo de prediccion...

REM Eliminar cualquier contenedor previo sin mostrar la salida
docker rm -f tp_aai_tuia >nul 2>&1

REM Ejecutar el contenedor y redirigir la salida estándar y de error a nul
docker run -v %CD%/output:/app/output -v %CD%/%TO_PREDICT_FILE%:/app/to_predict.csv -it --name tp_aai_tuia img_tp_aai_tuia:1.0 --to_predict /app/to_predict.csv >nul 2>&1

REM Verificar si predictions.csv existe dentro de la carpeta output
IF NOT EXIST "%PREDICTIONS_FILE%" (
    echo Error: no se encontro la imagen de Docker
    EXIT /B 1
)

echo.
echo Ejecucion terminada, el archivo predictions.csv se encuentra en la carpeta output.
endlocal