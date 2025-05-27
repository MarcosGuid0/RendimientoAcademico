import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("data/student_habits_performance.csv")

# Eliminar columna innecesaria
df = df.drop(columns=['student_id'])

# Convertir la variable de salida (exam_score) en clases
def categorizar(score):
    if score <= 50:
        return 0  # Bajo
    elif score <= 75:
        return 1  # Medio
    else:
        return 2  # Alto

df['score_class'] = df['exam_score'].apply(categorizar)

# Separar características y etiquetas
X = df.drop(columns=['exam_score', 'score_class'])
y = df['score_class']

# Columnas categóricas
categorical_columns = X.select_dtypes(include='object').columns.tolist()

# Preprocesador
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
], remainder='passthrough')

# Pipeline con modelo de clasificación
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Predicciones
y_pred = pipeline.predict(X_test)

# Reporte de clasificación
print("📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Bajo", "Medio", "Alto"]))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Bajo", "Medio", "Alto"],
            yticklabels=["Bajo", "Medio", "Alto"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
