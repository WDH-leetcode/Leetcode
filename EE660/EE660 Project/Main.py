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

X_test = pandas.read_csv('data\credit_test.csv')
X_test.drop(X_test.tail(353).index, inplace=True)
X_test = X_test.iloc[:, 2:18]
X_test = X_test.drop(['Purpose', 'Months since last delinquent'], axis=1)
cleanup_data = {'Term': {'Short Term': -1, 'Long Term': 1},
                'Years in current job': {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                                         '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10},
                'Home Ownership': {'Rent': 1, 'Own Home': 2, 'HaveMortgage': 3, 'Home Mortgage': 4}}
X_test.replace(cleanup_data, inplace=True)
missing_columns = ['Credit Score', 'Annual Income', 'Years in current job', 'Bankruptcies', 'Maximum Open Credit',
                   'Tax Liens']
X_train_para = {'Credit Score': ['Current Credit Balance', 'Number of Credit Problems', 'Monthly Debt', 'Term', 'Number of Open Accounts', 'Home Ownership', 'Years of Credit History', 'Current Loan Amount'], 'Annual Income': ['Current Credit Balance', 'Number of Credit Problems', 'Monthly Debt', 'Term', 'Number of Open Accounts', 'Home Ownership', 'Years of Credit History', 'Current Loan Amount'], 'Years in current job': ['Current Credit Balance', 'Number of Credit Problems', 'Monthly Debt', 'Term', 'Number of Open Accounts', 'Home Ownership', 'Years of Credit History', 'Current Loan Amount'], 'Bankruptcies': ['Current Credit Balance', 'Number of Credit Problems', 'Monthly Debt', 'Term', 'Number of Open Accounts', 'Home Ownership', 'Years of Credit History', 'Current Loan Amount'], 'Maximum Open Credit': ['Current Credit Balance', 'Number of Credit Problems', 'Monthly Debt', 'Term', 'Number of Open Accounts', 'Home Ownership', 'Years of Credit History', 'Current Loan Amount'], 'Tax Liens': ['Current Credit Balance', 'Number of Credit Problems', 'Monthly Debt', 'Term', 'Number of Open Accounts', 'Home Ownership', 'Years of Credit History', 'Current Loan Amount']}
col_min = {'Current Loan Amount': 10802.0, 'Credit Score': 585.0, 'Annual Income': 76627.0, 'Monthly Debt': 0.0, 'Current Credit Balance': 0.0, 'Maximum Open Credit': -756539.8799273572}
col_max = {'Current Loan Amount': 99999999.0, 'Credit Score': 7510.0, 'Annual Income': 165557393.0, 'Monthly Debt': 435843.28, 'Current Credit Balance': 32878968.0, 'Maximum Open Credit': 1539737892.0}
for feature in missing_columns:
    filesaved = '{}_data.sav'.format(feature)
    loaded_model = joblib.load(filesaved)
    X_test.loc[X_test[feature].isnull(), feature] = \
        loaded_model.predict(X_test[X_train_para[feature]])[X_test[feature].isnull()]

# normalize test according to training parameters
cols_to_norm = ['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Current Credit Balance',
                'Maximum Open Credit']
for col in cols_to_norm:
    X_test[col] = X_test[col].apply(lambda x: (x - col_min[col])/(col_max[col]-col_min[col]))
lr_clf = joblib.load('logistic regression_model.sav')
y_pred = lr_clf.predict(X_test)
y_pred = pandas.DataFrame(y_pred)
X_test_write = pandas.read_csv('data\credit_test.csv')
X_test_write['Loan Status'] = y_pred
X_test_write.to_csv('credit_test.csv', index=False)
print('results for credit test is added to the last column, 0 is paid off, 1 is charged off.')

                                        ### TEST created from credit_train
# random forest and logistic regression has the comparable accuracy, but logistic regression runs faster
# Preprocessing using parameters from x_train
X_test = pandas.read_csv('X_test_data.csv')
X_test = X_test.iloc[:, 3:]
X_test = X_test.drop(['Purpose', 'Months since last delinquent'], axis=1)
# Construct y_train
y_test = X_test.iloc[:, 0]
X_test = X_test.iloc[:, 1:]

# replace categorical data with numeric values
X_test.replace(cleanup_data, inplace=True)

# fill in missing data according to the parameters from training
missing_columns = ['Credit Score', 'Annual Income', 'Years in current job', 'Bankruptcies', 'Maximum Open Credit',
                   'Tax Liens']
for feature in missing_columns:
    filesaved = '{}_data.sav'.format(feature)
    loaded_model = joblib.load(filesaved)
    X_test.loc[X_test[feature].isnull(), feature] = \
        loaded_model.predict(X_test[X_train_para[feature]])[X_test[feature].isnull()]

# normalize test according to training parameters
cols_to_norm = ['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Current Credit Balance',
                'Maximum Open Credit']
for col in cols_to_norm:
    X_test[col] = X_test[col].apply(lambda x: (x - col_min[col])/(col_max[col]-col_min[col]))

# assign y_test to 0 and 1 for comparison
label_cleanup = {'Fully Paid': 1, 'Charged Off': 0}
y_test.replace(label_cleanup, inplace=True)

## load prediction model: logistic regression
y_pred = lr_clf.predict(X_test)
y_test_lst = y_test.values.tolist()
error = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test_lst[i]:
        error += 1
#print(error)
print('\nlogistic regression accuracy on test set {}%'.format((1 - error / len(y_pred)) * 100))
print(lr_clf.coef_)