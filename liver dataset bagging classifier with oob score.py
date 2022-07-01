# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:15:43 2022

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier

df=pd.read_csv(r'C:\Users\hp\Desktop\Python work\datasets\Indian Liver Patient Dataset (ILPD).csv')
print(df.head())

df=pd.get_dummies(df, columns=['gender'],drop_first=True)

df['alkphos'].fillna(value=df.alkphos.mean(), inplace=True)
print(df.isna().sum())
X= df.drop('is_patient',axis=1).values
y= df['is_patient'].values
print(X.shape)

#X= MinMaxScaler().fit_transform(X)
X= StandardScaler().fit_transform(X)
dt= DecisionTreeClassifier(min_samples_leaf=8,random_state=1)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30, random_state=42)


bc=BaggingClassifier(base_estimator=dt, n_estimators=50,oob_score=True, random_state=1)    

bc.fit(X_train,y_train)
y_pred=bc.predict(X_test)

print(accuracy_score(y_test, y_pred))
acc_oob= bc.oob_score_
print(acc_oob)