#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:56:25 2023

@author: artutaraperegmail.com
"""
import pandas as pd 

read_file = pd.read_excel("estudio1.xlsx")
  
read_file.to_csv ("estudioTFG.csv", index = None, header=True)
    
df = pd.DataFrame(pd.read_csv("estudioTFG.csv"))

print(df)

#Dividimos nuestro conjunto de datos para tener en la variable 'X' una matriz con todos los
#valores de las filas de nuestro estudio, y en la variable 'y' una lista con los resultados 
#sobre el diagnóstico:
    
X = df.iloc[:, :-1].values
y = df.iloc[:, 5].values

print(X)

print(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



#APRENDIZAJE SUPERVISADO

#Algoritmo de regresión logística

print("Regresión logística")

from sklearn.linear_model import LogisticRegression

regresion_logistica = LogisticRegression()

regresion_logistica.fit(X_train, y_train)

y_pred1 = regresion_logistica.predict(X_test)

print("Datos de prueba:\n{}".format(X_test))

print("Predicciones:\n{}".format(y_pred1))

print("Resultados reales:\n{}".format(y_test))

print("Precisión:" ,accuracy_score(y_test, y_pred1)) 

print("MSE:" ,mean_squared_error(y_test, y_pred1))


#Algoritmo SVM
 
print("SVM")

from sklearn import svm 

svc = svm.SVC(kernel='linear')

svc.fit(X_train, y_train)

y_pred2 = svc.predict(X_test)

print("Datos de prueba:\n{}".format(X_test))

print("Predicciones:\n{}".format(y_pred2))

print("Resultados reales:\n{}".format(y_test))

print("Precisión:" ,accuracy_score(y_test, y_pred2)) 

print("MSE:" ,mean_squared_error(y_test, y_pred2))


#Algoritmo Naive Bayes

print("Naive Bayes")

from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)

y_pred3 = naive_bayes.predict(X_test)

print("Datos de prueba:\n{}".format(X_test))

print("Predicciones:\n{}".format(y_pred3))

print("Resultados reales:\n{}".format(y_test))

print("Precisión:" ,accuracy_score(y_test, y_pred3)) 

print("MSE:" ,mean_squared_error(y_test, y_pred3))


#Algoritmo K-Nearest Neighbours

print("K-Nearest Neighbours")

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(X_train, y_train)

y_pred4 = clf.predict(X_test)

print("Datos de prueba:\n{}".format(X_test))

print("Predicciones:\n{}".format(y_pred4))

print("Resultados reales:\n{}".format(y_test))

print("Precisión:" ,accuracy_score(y_test, y_pred4)) 

print("MSE:" ,mean_squared_error(y_test, y_pred4))


#Algoritmo Random Forest

print("Random Forest")

from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(n_estimators=100, random_state=0)

rforest.fit(X_train, y_train)

y_pred5 = rforest.predict(X_test)

print("Datos de prueba:\n{}".format(X_test))

print("Predicciones:\n{}".format(y_pred5))

print("Resultados reales:\n{}".format(y_test))

print("Precisión:" ,accuracy_score(y_test, y_pred5)) 

print("MSE:" ,mean_squared_error(y_test, y_pred5))




#APRENDIZAJE NO SUPERVISADO

#Algoritmo K-Means Clustering

print("K-Means Clustering")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(X_train, y_train)

y_pred6 = kmeans.predict(X_test)

print("Datos de prueba:\n{}".format(X_test))

print("Predicciones:\n{}".format(y_pred6))

print("Resultados reales:\n{}".format(y_test))

print("Precisión:" ,accuracy_score(y_test, y_pred6)) 

print("MSE:" ,mean_squared_error(y_test, y_pred6))



