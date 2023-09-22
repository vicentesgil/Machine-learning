#!/usr/bin/env python


'''''''''
Script to train and validate an KNN strategy to classify and predict tumor outcome
Telemedicina y análisis de datos
'''''''''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("breast-cancer-wisconsin.csv")
dataset = dataset[pd.to_numeric(dataset['bare_nucleoli'], errors='coerce').notnull()]
# delete "id" column
dataset = dataset.iloc[:,1:]

# use only the features "size_uniformity" and "shape_uniformity" (columns with indexes 1 and 2)
# don't forget to also include column with index 9 ("class")
dataset = dataset.iloc[:,[1,2,3,9]]

# Create a dictionary to map class values to colors
class_color_map = {2: 'blue', 4: 'red'}

# Create a new column 'color' based on the 'class' column
dataset['color'] = dataset['class'].map(class_color_map)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x=dataset['size_uniformity']
y=dataset['shape_uniformity']
z=dataset['marginal_adhesion']
ax.scatter(x,y,z,c=dataset['color'])
ax.set_xlabel('size_uniformity')
ax.set_ylabel('shape_uniformity')
ax.set_zlabel('marginal_adhesion')

plt.show()

# Selecciona las características que deseas visualizar en 3D
features = ['size_uniformity', 'shape_uniformity', 'marginal_adhesion']
X = dataset[features]

# Selecciona las etiquetas de clase
y = dataset['class']


dataset = dataset.iloc[:,:-1]

# 1. split dataset into train and validation
from sklearn.model_selection import train_test_split, GridSearchCV


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)
print(X_train)


# 2. create a knn model using random k value (k=3) to then see if it can be improved
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# 3. train the model using the training sets
knn.fit(X_train, y_train)

# 4. predict the response for test dataset
y_pred = knn.predict(X_test)

# 5. evaluate the model
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

accuracy = metrics.accuracy_score(y_test, y_pred)
print("With K=%s, accuracy is: %s" % (k, accuracy))
print(classification_report(y_test, y_pred))

#### plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()


########## find best parameter for the KNN model #############
kvalues = np.arange(1,25)
train_accuracy = []
test_accuracy = []
# loop over different values of k
for i, k in enumerate(kvalues):
        knn = KNeighborsClassifier(n_neighbors=k)
        # fit with knn
        knn.fit(X_train,y_train)
        # train accuracy 
        train_accuracy.append(knn.score(X_train,y_train))
        # test accuracy
        test_accuracy.append(knn.score(X_test,y_test))

# plot the resulting accuracy results
plt.figure(figsize=[13,8])
plt.plot(kvalues, test_accuracy, label = 'Testing Accuracy')
plt.plot(kvalues, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('Train vs Test accuracy')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.xticks(kvalues)
plt.show()

# Print best accuracy result
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))


################ another method to find best parameter for the KNN model #####

error_rate=[] # list that will store the average error rate value of k
for i, k in enumerate(kvalues):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))
error_rate


# plotting the error rate vs k graph 
plt.figure(figsize=(12,6))
plt.plot(range(1,25),error_rate,marker="o",markerfacecolor="green",
         linestyle="dashed",color="red",markersize=15)
plt.title("Error rate vs k value",fontsize=20)
plt.xlabel("k- values",fontsize=20)
plt.ylabel("error rate",fontsize=20)
plt.xticks(kvalues)
plt.show()

################ repeat and train the model with best parameters #############
k = 4
knn = KNeighborsClassifier(n_neighbors=k)

# train the model using the training sets
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


print(classification_report(y_test, y_pred))

################ make predictions #################
random_x = [(2,5,8)]
random_y = knn.predict(random_x)
print("X=%s, Predicted=%s" % (random_x[0], random_y[0]))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", cm[0,0]/(cm[0,0]+cm[0,1]))
print("Specificity:",cm[1,1]/(cm[0,1]+cm[1,1]))
print("Recall:",cm[0,0]/(cm[0,0]+cm[1,0]))

