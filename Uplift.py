# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:38:40 2017

@author: Celine
"""
#--------------------------------------------------------------------------
# Traitement des données
#--------------------------------------------------------------------------

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import train_test_split #(validation croisée)

data = pd.read_csv('C:/Users/Celine/Documents/Documents these/Projet Uplift/Kevin_Hillstrom.csv', sep=',',)
data.head()
data.describe()

data_traitement=data[data['segment'] !='No E-Mail']
data_control=data[data['segment']=='No E-Mail']


X = data[["recency","history"]]
y = data["conversion"]


Xt=data_traitement[["recency","history"]]
yt=data_traitement["conversion"]

Xc=data_control[["recency","history"]]
yc=data_control["conversion"]


#Création de l'ensemble de validation
Xt_train, Xt_test, yt_train, yt_test = model_selection.train_test_split(Xt, yt, test_size=0.2)
Xc_train, Xc_test, yc_train, yc_test = model_selection.train_test_split(Xc, yc, test_size=0.2)



#--------------------------------------------------------------------------
# LDA - Uplift naif
#--------------------------------------------------------------------------
lda_t = LinearDiscriminantAnalysis(n_components=None)
lda_c = LinearDiscriminantAnalysis(n_components=None)

# On entraine le modele
lda_t.fit(Xt_train, yt_train)
lda_c.fit(Xc_train, yc_train)

# On test le modele
yt_predict = lda_t.predict(Xt_test)
yc_predict = lda_c.predict(Xc_test)

cm = confusion_matrix(yt_test, yt_predict)
print(cm)



def lda(X):
    return lda_t.predict_proba(X) - lda_c.predict_proba(X)


#--------------------------------------------------------------------------
# QDA - Uplift naif
#--------------------------------------------------------------------------
qda_t = QuadraticDiscriminantAnalysis()
qda_c = QuadraticDiscriminantAnalysis()

# On entraine le modele
qda_t.fit(Xt_train, yt_train)
qda_c.fit(Xc_train, yc_train)

def qda(X):
    return qda_t.predict_proba(X) - qda_c.predict_proba(X)

#--------------------------------------------------------------------------
# Regression logistique - Uplift naif
#--------------------------------------------------------------------------
lr_t = linear_model.LogisticRegression()
lr_c = linear_model.LogisticRegression()

# On entraine le modele
lr_t.fit(Xt_train, yt_train)
lr_c.fit(Xc_train, yc_train)

def lr(X):
    return lr_t.predict_proba(X) - lr_c.predict_proba(X)



