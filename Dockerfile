# Usa una imagen base oficial de Python
FROM python:3.10

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de tu proyecto al contenedor
COPY . /app

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto requerido por Cloud Run
EXPOSE 8080

# Ejecuta Streamlit escuchando en el puerto correcto
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false"]