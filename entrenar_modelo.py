import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Cargar datos
df = pd.read_csv("data/student_habits_performance.csv")

# Eliminar columna innecesaria
df = df.drop(columns=['student_id'])

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# Identificar columnas categóricas
categorical_columns = X.select_dtypes(include='object').columns.tolist()

# Preprocesador para columnas categóricas
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
], remainder='passthrough')

# Crear pipeline con preprocesamiento y modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Evaluar modelo
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar métricas
print(f"✅ Entrenamiento completado.")
print(f"📊 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.3f}")

# Guardar modelo
output_path = "modelo.pkl"
joblib.dump(pipeline, output_path)
print(f"💾 Modelo guardado en: {os.path.abspath(output_path)}")
