import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el conjunto de datos Breast Cancer Wisconsin
dataset = pd.read_csv("breast-cancer-wisconsin.csv")
dataset = dataset[pd.to_numeric(dataset['bare_nucleoli'], errors='coerce').notnull()]
# Eliminar "id" column
dataset = dataset.drop('id', axis=1)

# Separar características (X) y etiquetas (y)
X = dataset.drop('class', axis=1)  # Las características son todas las columnas excepto 'class'
y = dataset['class']  # 'class' es la etiqueta que queremos predecir

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Crear un clasificador Naive Bayes
naive_bayes = GaussianNB()

# Entrenar el modelo con los datos de entrenamiento
naive_bayes.fit(X_train, y_train).predict(X_test)

# Realizar predicciones en los datos de prueba
y_pred = naive_bayes.predict(X_test)

# Calcular la precisión del modelo
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión y el reporte de clasificación
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))