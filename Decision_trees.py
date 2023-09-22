#!/usr/bin/env python


'''''''''
Script to train and validate a Decision Tree strategy to classify and predict tumor outcome
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


#################### DECISION TREE MODEL ##################
# 1. split dataset into train and validation
from sklearn.model_selection import train_test_split

x,y = dataset.loc[:,dataset.columns != 'class'], dataset.loc[:,'class']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

# 2. create a Decision Tree model 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model = DecisionTreeClassifier() # default parameters: criterion="gini", splitter="best", max_depth=None

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


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", cm[0,0]/(cm[0,0]+cm[0,1]))
print("Specificity:",cm[1,1]/(cm[0,1]+cm[1,1]))
print("Recall:",cm[0,0]/(cm[0,0]+cm[1,0]))

# calculate other metrics

# ...

##### plot decision tree
features = X_train.columns.values.tolist()

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=features,  
                   filled=True)
fig.savefig("tree.png")


##### why are there so many differences when running the same script?
##### check if model is over-fitted 
### if model is overfitted, validation scores are lower than training scores
### which means that the model performs great on training data but not so much on the validation set
##con esta grafuca vemos que a partir de 8 el modelo esta totlamente overfitting

max_depth_range = list(range(3, 20))
train_accuracy = []
test_accuracy = []
# loop over different values of depth
for i, k in enumerate(max_depth_range):
        model = DecisionTreeClassifier(max_depth=k)
        model.fit(X_train,y_train)
        # train accuracy 
        train_accuracy.append(model.score(X_train,y_train))
        # test accuracy
        test_accuracy.append(model.score(X_test,y_test))

# plot the resulting accuracy results
plt.figure(figsize=[13,8])
plt.plot(max_depth_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(max_depth_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('Train vs Test accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.xticks(max_depth_range)
plt.show()


##### plot training and test scores using cross-validation
from sklearn.model_selection import validation_curve
train_score, test_score = validation_curve(DecisionTreeClassifier(), x, y,
                                           param_name="max_depth",
                                           param_range=max_depth_range,
                                           cv=5, scoring="accuracy")

# Calculating mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis=1)
std_train_score = np.std(train_score, axis=1)
 
# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis=1)
std_test_score = np.std(test_score, axis=1)
 
# Plot mean accuracy scores for training and testing scores
plt.plot(max_depth_range, mean_train_score,
         label="Training Score", color='b')
plt.plot(max_depth_range, mean_test_score,
         label="Cross Validation Score", color='g')
 
# Creating the plot
plt.title("Decision Tree validation curve")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc='best')
plt.show()


##### quick way to find best hyperparameters
parameters = {'max_depth':range(3,20)}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
clf.fit(X_train, y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 

##### find best tree size by plotting depth vs accuracy
max_depth_range = list(range(3, 20))
accuracy = []
for depth in max_depth_range:
    clf= DecisionTreeClassifier(max_depth = depth, random_state = 0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)

plt.figure()
plt.plot(max_depth_range, accuracy) # adds the line
plt.grid() # adds a grid to the plot
plt.ylabel('accuracy') # xlabel
plt.xlabel('tree size') # ylabel
plt.show()

##### another way to find best tree size plotting depth vs ROC AUC Scores
from sklearn.metrics import roc_auc_score
max_depth_range = list(range(3, 20))
ROC_AUC_scores = []
for depth in max_depth_range:
    treeModel = DecisionTreeClassifier(max_depth = depth)
    treeModel.fit(X_train, y_train)
    treePrediction = treeModel.predict(X_test)
    score = roc_auc_score(y_test, treePrediction)
    ROC_AUC_scores.append(score)

plt.figure()
plt.plot(max_depth_range,ROC_AUC_scores)
plt.grid()
plt.xlabel("tree size")
plt.ylabel("ROC AUC Scores")
plt.show()

######### optimization of our decision tree
# 2. create a Decision Tree model 
model = DecisionTreeClassifier(criterion="entropy", max_depth=4) # changing default parameters

# 3. train the model using the training sets
model.fit(X_train, y_train)

# 4. predict the response for test dataset
y_pred = model.predict(X_test)

# 5. evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy is: %s" % (accuracy))
print(classification_report(y_test, y_pred))

################ make predictions #################
random_x = [(2,2,1,6,2,7,4,7,5)]
random_y = model.predict(random_x)
print("X=%s, Predicted=%s" % (random_x[0], random_y[0]))

