
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""reg_lin.py: Function to help in project."""

__author__ = "Antoine"
__credits__ = ["Antoine"]
__version__ = "1.0"
__status__ = "Production"


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from numpy import arange
import streamlit as st

# @st.cache(allow_output_mutation=True)
def get_metrix( y , X , model):
    """
    Given a model and a dataset, return the fit model and the predictions (only classes)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None)
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print ("Train R2 = {}; Train RMSE {}"
      .format(round(r2_score(y_train, y_train_pred),3),
              round(mean_squared_error(y_train, y_train_pred, squared=False),3),
              round(accuracy_score(y_train, y_train_pred),3)))

    print ("Test R2  = {}; Test RMSE {}"
      .format(round(r2_score(y_test, y_test_pred),3),
              round(mean_squared_error(y_test, y_test_pred, squared=False),3),
              round(accuracy_score(y_train, y_train_pred),3)))
    return model, y_test_pred, y_test


# @st.cache(allow_output_mutation=True)
def get_metrix_v2( y , X , model):
    """
    Given a model and a dataset, return the fit model and the predictions (classes and probabilities)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None)
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    print ("Train R2 = {}; Train RMSE {}"
      .format(round(r2_score(y_train, y_train_pred),3), round(mean_squared_error(y_train, y_train_pred, squared=False),3),
              round(accuracy_score(y_train, y_train_pred),3)))

    print ("Test R2  = {}; Test RMSE {}"
      .format(round(r2_score(y_test, y_test_pred),3),
              round(mean_squared_error(y_test, y_test_pred, squared=False),3),
              round(accuracy_score(y_train, y_train_pred),3)))
    return model, y_test_proba, y_test_pred, y_test


def get_alpha( y , X , model):
    """Given a dataset and a model, get the best parameters for the alpha of the model"""
    parameters = {'alpha':range(1000)}
    clf = RandomizedSearchCV(model, parameters , n_iter=100)
    search = clf.fit(X,y)

    best_alpha = search.best_params_["alpha"]

    cpt = 1
    best_alpha2=best_alpha
    for i in range (0,(len(str(best_alpha)))-1,1):
        print(best_alpha2)
        parameters = {'alpha': arange((best_alpha2)*(1-(0.5/cpt)),(best_alpha2)*(1+(0.5/cpt)))}
        clf = RandomizedSearchCV(model, parameters ,n_iter=100)
        search = clf.fit(X,y)

        best_alpha2 = search.best_params_["alpha"]
        cpt += 1

    return best_alpha
   