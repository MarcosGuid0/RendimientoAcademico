import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/student_habits_performance.csv")

# Aquí no es necesario convertir categorías a números porque 'exam_score' es numérica
X = df.drop("exam_score", axis=1)  # Eliminar la columna de la etiqueta
y = df["exam_score"]  # Usar 'exam_score' como objetivo

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
