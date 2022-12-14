import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from utils import *
import matplotlib.pyplot as plt

# ===== Supervised Learning ====
# SL System Results
df_bachelor = pd.read_csv("data_processed/bachelor_preprocessed.csv")
X,y = encode_data(df_bachelor)

# Split the training, validation, test data
np.random.seed(100)

#default is trainsize=0.7, valsize=0.1, testsize=0.2
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X,y)

# Combine training and validation into a new training set for final test performance
y_train_combined = pd.concat([y_train, y_val])
X_train_combined  = pd.concat([X_train, X_val])

#standardize values based on training statistics
X_test = standardize_data(X_train_combined, X_test)
X_train_combined = standardize_data(X_train_combined, X_train_combined)

X_test_S = X_test
X_train_combined_S = X_train_combined
y_test_S = y_test
y_train_combined_S = y_train_combined
X_val_S = X_val
y_val_S = y_val
X_train_S = X_train
y_train_S = y_train
X_S = X

#calculate e_M for generalization bound\
M=1
N_Source = len(y_test_S)
delta = 0.05
e_M_Source = np.sqrt((1/(2*N_Source))*np.log(2*M/delta))
print("Number of source test samples: ", N_Source)
print("e_M_Source = ", e_M_Source)


#Load model and print metrics
filename = 'SL_final_model_Ridge.pickle'
model_SL = pickle.load(open(filename, 'rb'))
y_pred = model_SL.predict(X_test)
y_pred_train = model_SL.predict(X_train_combined)
print("Ridge (L2) Regression")
print_metrics(y_pred, y_test, y_pred_train, y_train_combined, e_M_Source)
print(model_SL.coef_)
print("R^2: ", model_SL.score(X_test, y_test))
plot_scatter_pred_true(y_pred, y_test, "Ridge (L2) Regression")

# ===== Transfer Learning ====
# TL System Results
df_bachelorette = pd.read_csv("data_processed/bachelorette_preprocessed.csv")
X,y = encode_data(df_bachelorette)

# Split the training, validation, test data
np.random.seed(15)

#default is trainsize=0.7, valsize=0.1, testsize=0.2
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X,y)

# Combine training and validation into a new training set for final test performance
y_train_combined = pd.concat([y_train, y_val])
X_train_combined  = pd.concat([X_train, X_val])

#standardize values based on training statistics
X_test = standardize_data(X_train_combined, X_test)
X_train_combined = standardize_data(X_train_combined, X_train_combined)

X_test_T = X_test
X_train_combined_T = X_train_combined
y_test_T = y_test
y_train_combined_T = y_train_combined
X_val_T = X_val
y_val_T = y_val
X_train_T = X_train
y_train_T = y_train
X_T = X

N_Target = len(y_test_T)
print("Number of target test samples: ", N_Target)

#Load model and print metrics
filename = 'TL_final_model_TrAdaBoost.pickle'
model_TF = pickle.load(open(filename, 'rb'))
y_pred = model_TF.predict_estimator(X_test_T)
y_pred_train = model_TF.predict(X_train_combined_S)
test_mse = mean_squared_error(y_test_T, y_pred)
print("\n")
print("Transfer Learning: TrAdaBoostR2 with Ridge")
print_metrics(y_pred, y_test_T, y_pred_train, y_train_combined_S)
print(model_TF.predict_weights("target"))
print("R^2: ", model_TF.score(X_test_T, y_test_T))
plot_scatter_pred_true(y_pred, y_test_T, "TrAdaBoost2 with Ridge (L2) Regression")


