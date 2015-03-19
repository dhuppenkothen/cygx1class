__author__ = 'danielahuppenkothen'

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=2.5, rc={"axes.labelsize": 26})
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20)
plt.rc("text", usetex=True)
import matplotlib.cm as cmap

import numpy as np
import generaltools as gtb
## utilities for the analysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix



#### RUNNING THE ENTIRE SUPERVISED CLASSIFICATIOn

def supervised_class(fscaled_train, fscaled_val, l_train, l_val):

    ## K0Nearest Neighbour
    params = {'n_neighbors': [1, 3, 5, 10, 15, 20, 25, 30, 50]}#, 'max_features': }
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid=params, verbose=1, n_jobs=10)
    grid_knn.fit(fscaled_train, l_train)
    labels_knn = grid_knn.predict(fscaled_val)

    print("I have finished KNN classification. RFC is next.")

    ## Random Forest Classifier
    params = {'max_depth': [1,3,5,6,7,8,9,10, 11,12,13,14,15,16,17, 18, 19, 20, 21, 22, 23, 25,30,40]}
    grid_rfc = GridSearchCV(RandomForestClassifier(n_estimators=500), param_grid=params,
                           verbose=1, n_jobs=10)
    grid_rfc.fit(fscaled_train, l_train)
    labels_rfc = grid_rfc.predict(fscaled_val)
    labels_train_rfc = grid_rfc.predict(fscaled_train)

    print("I have finished RFC classification. Linear Regression is next.")

    ### Linear SVC Classification
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_lm = GridSearchCV(linear_model.LogisticRegression(penalty="l1", class_weight="auto"),
                        param_grid=params, verbose=1, n_jobs=10)
    grid_lm.fit(fscaled_train, l_train)
    labels_lm = grid_lm.predict(fscaled_val)


    print("K-Nearest Neighbour Classification Results:\n")
    print(grid_knn.best_params_)
    print(grid_knn.score(fscaled_train, l_train))
    print(grid_knn.score(fscaled_val, l_val))

    print("Random Forest Classifier Results:\n")
    print(grid_rfc.best_params_)
    print(grid_rfc.score(fscaled_train, l_train))
    print(grid_rfc.score(fscaled_val, l_val))

    print("Linear SVC Classifier Results:\n")
    print(grid_lm.best_params_)
    print(grid_lm.score(fscaled_train, l_train))
    print(grid_lm.score(fscaled_val, l_val))

    return grid_knn, grid_rfc, grid_lm
    #return grid_knn


def predict_labels(clf, features):
    return clf.predict(features)


def plot_confusion_matrix(labels_true, labels_predicted):

    sns.set_style("white")
    unique_labels = np.unique(labels_true)
    cm = confusion_matrix(labels_true, labels_predicted, labels=unique_labels)
    print(cm)
    print(unique_labels)
    plt.matshow(cm, cmap=cmap.Spectral_r )
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(range(len(unique_labels)), unique_labels, rotation=70)
    plt.yticks(range(len(unique_labels)), unique_labels)
    plt.show()

    return