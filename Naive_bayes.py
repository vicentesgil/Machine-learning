#!/usr/bin/env python

#Script modelo naive bayes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Breast Cancer Wisconsin
dataset = pd.read_csv("breast-cancer-wisconsin.csv")
dataset = dataset[pd.to_numeric(dataset['bare_nucleoli'], errors='coerce').notnull()]
# Eliminar "id" column
dataset = dataset.drop('id', axis=1)

# Separar características (X) y etiquetas (y)
X = dataset.drop('class', axis=1)  # Las características son todas las columnas excepto 'class'
y = dataset['class']  # 'class' es la etiqueta que queremos predecir

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

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


# Mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

classes = naive_bayes.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.ylabel('Clase real')
plt.xlabel('Clase predicha')
plt.title('Matriz de Confusión')
plt.colorbar()
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", cm[0,0]/(cm[0,0]+cm[0,1]))
print("Specificity:",cm[1,1]/(cm[0,1]+cm[1,1]))
print("Recall:",cm[0,0]/(cm[0,0]+cm[1,0]))


########## plot de roc_curve ########

from sklearn.metrics import roc_curve

y_score1 = naive_bayes.predict_proba(X_test)[:,1]
f_p_rate, t_p_rate, threshold1 = roc_curve(y_test, y_score1, pos_label = '2')

plt.subplots(1, figsize=(10,10))
plt.title('Roc curve')
plt.plot(f_p_rate,t_p_rate)
plt.plot([0,1], ls="--")
plt.plot([0,0],[1,0], c="7"), plt.plot([1,1], c="7")
plt.ylabel('true positive rate')
plt.xlabel('false positive rate')

plt.show()













