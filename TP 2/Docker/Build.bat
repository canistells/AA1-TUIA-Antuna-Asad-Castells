REM Build.bat - Construir la imagen Docker

REM Creación de carpeta que contendrá el comprimido con las imágenes y asignación de permisos
mkdir output

REM Construcción de la imagen a partir del Dockerfile
docker build --tag img_tp_aai_tuia:1.0 .

REM Confirmación del éxito
echo Imagen img_tp_aai_tuia:1.0 construida correctamente.