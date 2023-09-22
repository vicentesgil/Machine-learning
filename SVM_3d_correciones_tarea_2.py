#!/usr/bin/env python


'''''''''
Script to train and validate an SVM strategy to classify and predict tumor outcome
Telemedicina y análisis de datos
'''''''''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.svm import SVC
import plotly.graph_objects as go

# load and prepare dataset following the steps we studied during preprocessing
dataset = pd.read_csv("breast-cancer-wisconsin.csv")
# delete rows with not-numeric "bare_nucleoli" values (not necessary if we pick other columns)
# dataset = dataset[pd.to_numeric(dataset['bare_nucleoli'], errors='coerce').notnull()]
# delete "id" column
dataset = dataset.iloc[:,1:]

# use only the features "size_uniformity", "shape_uniformity" and "marginal_adhesion" (columns with indexes 1, 2 and 3)
# don't forget to also include column with index 9 ("class")
dataset = dataset.iloc[:,[1,2,3,9]]

########### plot features in a 2D graph ##############
# Create a dictionary to map class values to colors
class_color_map = {2: 'blue', 4: 'red'}

# Create a new column 'color' based on the 'class' column
dataset['color'] = dataset['class'].map(class_color_map)

# Plot results
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset['size_uniformity'], dataset['shape_uniformity'], dataset['marginal_adhesion'], c=dataset['color'])
plt.xlabel('Size Uniformity')
plt.ylabel('Shape Uniformity')
plt.ylabel('Marginal Adhesion')
plt.show()

# hint: the plot will give you an idea of what the svm hyperplane should look like
# so if you want to save computing time during the search for the best hyperparameters, 
# you can infer what kernel is most likely to be the best suit for our data
# (in this case it is highly improbable that a "rbf" or a "sigmoid" kernel would be the best fit)

#################### SVM MODEL ##################
# 0. remove the new column "color" that we just created
dataset = dataset.iloc[:,:-1]

# 1. split dataset into train and validation
from sklearn.model_selection import train_test_split, GridSearchCV

x,y = dataset.loc[:,dataset.columns != 'class'], dataset.loc[:,'class']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
print(X_train)

# 2. create a svm classifier using random values to then see if they can be improved
clf = SVC(C=0.01, kernel='rbf')

# 3. train the model using the training sets
clf.fit(X_train, y_train)

# 4. predict the response for test dataset
y_pred = clf.predict(X_test)

# 5. evaluate the model
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ########## find best parameters for the SVM model #############
# # search best hyperparameters using GridSearchCV
# # defining parameter range using a dictionary with different values of 'C', 'gamma' and 'kernel'
# param_grid = {'C': [0.1, 1, 10, 100], 
#               'gamma': [1, 0.1, 0.01, 0.001],
#               'kernel': ['rbf', 'linear', 'poly']} 
  
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# # fitting the model for grid search
# grid.fit(X_train, y_train)

# # print best parameter after tuning
# print(grid.best_params_)

# # predict with test values and see how accurate the new model is using the classification_report tool from sklearn
# grid_predictions = grid.predict(X_test)
# print(classification_report(y_test, grid_predictions))

################ repeat and train the model with best parameters #############
clf = SVC(C = 0.1, kernel = 'linear', gamma = 1)

# train the model using the training sets
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

################ make predictions #################
random_x = [(2,5,8)] 
random_y = clf.predict(random_x)
print("X=%s, Predicted=%s" % (random_x[0], random_y[0]))

############# 3D Plot
# The basics of the plot are the same as before plotting the hyperplane
# Create a dictionary to map class values to colors
class_color_map = {2: 'blue', 4: 'red'}

# Create a new column 'color' based on the 'class' column
dataset['color'] = dataset['class'].map(class_color_map)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d') 
ax.scatter(dataset['size_uniformity'], dataset['shape_uniformity'], dataset['marginal_adhesion'], c=dataset['color'])

# Legends for the axis
plt.xlabel('Size Uniformity')
plt.ylabel('Shape Uniformity')
plt.ylabel('Marginal Adhesion')


# Defining the hyperplane (this was the tricky part)
w = clf.coef_
w1 = w [:, 0]
w2 = w [:, 1]
w3 = w [:, 2]
b = clf.intercept_
xx, yy = np.meshgrid(range(min(dataset['size_uniformity'])-1,max(dataset['size_uniformity'] + 1)), range(min(dataset['shape_uniformity'])-1,max(dataset['shape_uniformity'] + 1)))
zz = (-w1 * xx - w2 * yy - b) * 1. /w3
# also works by not pre-defining w coordinates :  zz = (-w[0, 0] * xx - w[0, 1] * yy - b) / w[0, 2]

ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, color='green', alpha=0.5) # This line plots the hyperplane
# ax.plot_wireframe would plot a wireframe instead of a surface
# alpha modulates transparency


plt.show()

