@echo off

SET TO_PREDICT_FILE=to_predict.csv
SET OUTPUT_DIR=%CD%/output
SET PREDICTIONS_FILE=output/predictions.csv

IF NOT EXIST "%TO_PREDICT_FILE%" (
    echo El archivo %TO_PREDICT_FILE% no se encontro. Asegurate de que este en la misma carpeta que este script.
    EXIT /B 1
)

echo Ejecutando la imagen Docker con el archivo de prediccion...

docker rm -f tp_aai_tuia >nul 2>&1

docker run ^
  -v "%CD%"/output:/app/output ^
  -v "%CD%"/%TO_PREDICT_FILE%:/app/to_predict.csv ^
  -it --name tp_aai_tuia img_tp_aai_tuia:1.0 --to_predict.csv /app/to_predict.csv

  

IF NOT EXIST "%PREDICTIONS_FILE%" (
    echo Error: no se genero el archivo predictions.csv en la carpeta output.
    EXIT /B 1
)

echo Ejecucion terminada, el archivo predictions.csv se encuentra en la carpeta output.