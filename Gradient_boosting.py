#!/usr/bin/env python


'''''''''
Script to train and validate a Gradient Boosting model to classify and predict tumor outcome
Telemedicina y análisis de datos
'''''''''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# load and prepare dataset following the steps we studied during preprocessing
dataset = pd.read_csv("breast-cancer-wisconsin.csv")
# delete rows with not-numeric "bare_nucleoli" values
dataset = dataset[pd.to_numeric(dataset['bare_nucleoli'], errors='coerce').notnull()]
# delete "id" column
dataset = dataset.iloc[:,1:]


#################### GRADIENT BOOSTING MODEL ##################
# 1. split dataset into train and validation
from sklearn.model_selection import train_test_split, RandomizedSearchCV

x,y = dataset.loc[:,dataset.columns != 'class'], dataset.loc[:,'class']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

# 2. create a Gradient Booster model 
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier() # default parameters: subsample=1, learning_rate=0.1, n_estimators=100

# 3. train the model using the training sets
model.fit(X_train, y_train)

# 4. predict the response for test dataset
y_pred = model.predict(X_test)

# 5. evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy is: %s" % (accuracy))
print(classification_report(y_test, y_pred))

#### plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# calculate other metrics

acc = (TN + TP)/(TN + TP + FP +FN)
print("Accuracy is: %s" % (acc))
prec = TP/(TP+FP)
print("Precision is: %s" % (prec))
spec = TN / (TN + FP)
print("Specificity is: %s" % (spec))
recall = TP / (TP + FN)
print("Recall is: %s" % (recall))

##### accuracy is already quite high 
### however we can still try to tune the hyperparameters

param_dist = {'n_estimators': range(50,200),
              'subsample': np.arange(0.7,1.0,0.05),
              'learning_rate': np.arange(0.05,0.2, 0.01),
              'max_depth': range(1,20)}

# Use random search to find the best hyperparameters
best_model = RandomizedSearchCV(model, 
                                param_distributions = param_dist, 
                                n_iter=100, 
                                cv=7, error_score=0)

# Fit the random search object to the data
best_model.fit(X_train, y_train)

# Create a variable for the best model
best_rf = best_model.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', best_model.best_params_)


######### optimization of our model
# 2. optimize the model with our best estimators
# the best model is already defined by our variant "best_model"

# 3. train the model using the training sets
best_model.fit(X_train, y_train)

# 4. predict the response for test dataset
y_pred = best_model.predict(X_test)

# 5. evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy is: %s" % (accuracy))
print(classification_report(y_test, y_pred))

#### plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# calculate other metrics

acc = (TN + TP)/(TN + TP + FP +FN)
print("Accuracy is: %s" % (acc))
prec = TP/(TP+FP)
print("Precision is: %s" % (prec))
spec = TN / (TN + FP)
print("Specificity is: %s" % (spec))
recall = TP / (TP + FN)
print("Recall is: %s" % (recall))


################ make predictions #################
random_x = [(2,2,1,6,2,7,4,7,5)]
random_y = best_model.predict(random_x)
print("X=%s, Predicted=%s" % (random_x[0], random_y[0]))

