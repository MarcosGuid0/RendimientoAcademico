import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI
import joblib

# Cargar entorno
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cargar modelo entrenado
with open('modelo.pkl', 'rb') as f:
        model = joblib.load(f)

# Función para generar explicación con OpenAI
def generar_explicacion(data_dict, prediccion):
    prompt = (
        f"Un estudiante tiene los siguientes hábitos: {data_dict}. "
        f"El modelo predice un puntaje de examen de {prediccion:.2f}. "
        f"Como asesor académico, da una recomendación personalizada para mejorar su rendimiento."
    )

    response = client.chat.completions.create(
        model="gpt-4",  # o gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content

# Título de la app
st.title("📚 Predicción del Rendimiento Académico")
st.markdown("Ingresa los hábitos del estudiante para predecir su rendimiento y obtener una recomendación.")

# Formulario de entrada
with st.form("input_form"):
    age = st.number_input("Edad", min_value=10, max_value=100, step=1)
    social_media_hours = st.number_input("Horas en redes sociales por día", min_value=0.0, step=0.5)
    netflix_hours = st.number_input("Horas viendo Netflix por día", min_value=0.0, step=0.5)
    mental_health_rating = st.slider("Estado de salud mental (1=muy pobre, 10=excelente)", min_value=1, max_value=10)
    study_hours = st.number_input("Horas de estudio por semana", min_value=0.0, step=0.5)
    sleep_hours = st.number_input("Horas de sueño por día", min_value=0.0, step=0.5)
    attendance = st.slider("Asistencia (%)", min_value=0, max_value=100)
    exercise_frequency = st.slider("Nivel de distracción (0=bajo, 10=alto)", min_value=0, max_value=10)
    gender = st.selectbox("Género", ["Male", "Female"])
    part_time_job = st.selectbox("Trabajo a medio tiempo", ["Yes", "No"])
    diet_quality = st.selectbox("Calidad de la dieta", ["Poor", "Fair", "Good"])
    parental_education_level = st.selectbox("Nivel educativo de los padres", ["High School", "Bachelor", "Master", "PhD"])
    internet_quality = st.selectbox("Calidad de internet", ["Poor", "Average", "Good"])
    extracurricular_participation = st.selectbox("Participación en actividades extracurriculares", ["Yes", "No"])
    
    submitted = st.form_submit_button("Predecir")

# Procesamiento
if submitted:
    column_names = [
        "age", "gender", "study_hours_per_day", "social_media_hours", "netflix_hours",
        "part_time_job", "attendance_percentage", "sleep_hours", "diet_quality",
        "exercise_frequency", "parental_education_level", "internet_quality",
        "mental_health_rating", "extracurricular_participation"
    ]
    
    data = [[
        age, gender, study_hours, social_media_hours, netflix_hours,
        part_time_job, attendance, sleep_hours, diet_quality,
        exercise_frequency, parental_education_level, internet_quality,
        mental_health_rating, extracurricular_participation
    ]]

    features = pd.DataFrame(data, columns=column_names)

    pred = model.predict(features)[0]
    
    # Crear diccionario con los valores ingresados
    data_dict = {
        "Edad": age,
        "Género": gender,
        "Horas de estudio": study_hours,
        "Horas en redes sociales": social_media_hours,
        "Horas viendo Netflix": netflix_hours,
        "Trabajo a medio tiempo": part_time_job,
        "Asistencia": attendance,
        "Horas de sueño": sleep_hours,
        "Calidad de la dieta": diet_quality,
        "Frecuencia de distraccion": exercise_frequency,
        "Nivel educativo de los padres": parental_education_level,
        "Calidad de internet": internet_quality,
        "Estado de salud mental": mental_health_rating,
        "Participación en actividades extracurriculares": extracurricular_participation
        }

    # Mostrar predicción
    st.success(f"📈 Predicción del rendimiento: **{pred:.2f}**")

    # Generar explicación con IA
    with st.spinner("Generando recomendación con IA..."):
        recomendacion = generar_explicacion(data_dict, pred)
        st.markdown("### 💡 Recomendación personalizada")
        st.info(recomendacion)
