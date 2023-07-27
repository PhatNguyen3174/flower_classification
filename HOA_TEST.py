import pandas as pd 
import numpy as np
import pickle



dataset = pd.read_csv('IRIS - IRIS.csv')
X = dataset.drop("species", axis=1)
Y = dataset["species"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=100)

#Huấn luyện và kiểm thử
from sklearn.svm import SVC

svm = SVC(kernel="linear")
svm.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_train, svm.predict(X_train))


filename = 'SVMModel.sav'
pickle.dump(svm, open(filename, 'wb'))