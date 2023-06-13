# write your code here
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

import pandas as pd

(X, y), (_, _) = tf.keras.datasets.mnist.load_data()

X = X.reshape(60000, 28 * 28)

# print("Classes: " + str(np.unique(y_train.flatten())))
# print("Features' shape: " + str(x_train.shape))
# print("Target's shape: " + str(y_train.shape))
# print("min: " + str(x_train.min()) + ", max: " + str(x_train.max()))

X = X[:6000]
y = y[:6000]

norm = Normalizer()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

x_train = norm.fit_transform(x_train)
x_test = norm.transform(x_test)

# the function

acc = 0
name = ''

# Initialise the CV of Rfc
param_grid_knn = {
    'n_neighbors': [3, 4],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'brute']
}

CV_rfc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, scoring='accuracy', n_jobs=-1)

param_grid_rfc = {
    'n_estimators': [300, 500],
    'max_features': ['auto', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

cv_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=40), param_grid=param_grid_rfc, scoring='accuracy',
                      n_jobs=-1)


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    global acc, name
    clf = model.fit(features_train, target_train)
    score = clf.score(features_test, target_test)
    if score > acc:
        acc = score
        name = model.__class__.__name__
    print(f'Model: {model.__class__.__name__}\nAccuracy: {score}\n')


models = [KNeighborsClassifier(),
          DecisionTreeClassifier(random_state=40),
          LogisticRegression(random_state=40, solver="liblinear"),
          RandomForestClassifier(random_state=40)]

CV_rfc.fit(x_train, y_train)
print("K-nearest neighbours algorithm")
print("best estimator: " + str(CV_rfc.best_estimator_))
print("accuracy: " + str(CV_rfc.score(x_test, y_test)))

cv_rfc.fit(x_train, y_train)
print("Random forest algorithm")
print("best estimator: " + str(cv_rfc.best_estimator_))
print("accuracy: " + str(cv_rfc.score(x_test, y_test)))
