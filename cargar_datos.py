import pandas as pd

# Cargar el dataset
df = pd.read_csv("data/student_habits_performance.csv")

# Mostrar primeras filas
print(df.head())

# Mostrar estructura
print(df.info())
