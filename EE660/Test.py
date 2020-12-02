import pandas
import missingno as mno
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import BayesianGaussianMixture
import joblib
from sklearn.preprocessing import PolynomialFeatures

f_handle = pandas.read_csv('credit_train.csv')
N = f_handle.shape[0]
D = f_handle.shape[1]




                                        ### Split the data
# split the data into train and test first  ---> 10% test and 90% training
f_handle.drop(f_handle.tail(514).index, inplace=True)
X_pre = f_handle.iloc[:N//20, :]
X_test = f_handle.iloc[N//20:1.5*N//10, :]
X_train = f_handle.iloc[1.5*N//10:, :]
name_lst = f_handle.columns
for name in name_lst:
    st = X_pre[name].cumsum
    st.plot()





                                         ### PREPROCESSING
# inspect the training data
#print(X_train.head())
# get the feature names
name_lst = f_handle.columns
#print(name_lst)
# get rid of irrelevant information such as Loan ID and Customer ID
X_train = X_train.iloc[:, 2:]
# Construct y_train
y_train = X_train.iloc[:, 0]
X_train = X_train.iloc[:, 1:]
print(len(X_train))


# check for missing data number
print(X_train.isnull().sum())
#print(y_train.isnull().sum()) # there is no missing data in y label
mno.matrix(X_train, figsize=(20, 20))
plt.show()