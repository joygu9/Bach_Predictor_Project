""" Some utility functions for the project"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def normalize_col(df_in, columnname, mean_val, std_val):
    """ normalize column data"""
    df_out = df_in.copy()
    df_out[columnname] = (df_out[columnname] - mean_val) / std_val
    return df_out


def encode_data(df_in):
    """do one-hot encoding for hometown"""
    y = df_in['ElimWeek']
    df_in = df_in.drop(columns=["CONTESTANT", "ElimWeek"])
    X = pd.get_dummies(df_in, columns=["Region"])

    return X, y


def standardize_data(X_train, df_in):
    """standardize Age, NumRoses3, FirstDate based on training statistics"""
    age_mean, age_std = X_train["Age"].mean(), X_train["Age"].std()
    rose_mean, rose_std = X_train["NumRoses3"].mean(), X_train["NumRoses3"].std()
    date_mean, date_std = X_train["FirstDate"].mean(), X_train["FirstDate"].std()
    X_norm = df_in.copy()
    X_norm = normalize_col(X_norm, "Age", age_mean, age_std)
    X_norm = normalize_col(X_norm, "NumRoses3", rose_mean, rose_std)
    X_norm = normalize_col(X_norm, "FirstDate", date_mean, date_std)
    return X_norm


def split_data(X, y, trainsize=0.7, valsize=0.1, testsize=0.2):
    """Split into training, validation, and test data"""
    # Training and remainder
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, train_size=trainsize)

    # Validation and Test - test is 2/3 of remaining, to give ~10% to validation and 20% to test
    new_testsize = testsize / (1 - trainsize)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=new_testsize)

    return X_train, y_train, X_val, y_val, X_test, y_test


def model_selection(Xtrain, Ytrain, Xval, Yval, reg_setting, fit_int=False):
    """ run model selection steps by finding best lambda then return the fitted regressor,
    use reg_setting = "no",'L1', 'L2'"""
    if reg_setting == 'no':
        regressor = LinearRegression(fit_intercept=fit_int)
        reg_fit = regressor.fit(Xtrain, Ytrain)
        y_pred = reg_fit.predict(Xval)
        print("Linear Regression")
        print("Validation MSE: %.3f" % mean_squared_error(Yval, y_pred))
        return reg_fit

    start = -10
    end = 10
    step = 0.5
    num_lambdas = int((end - start) / step) + 1
    lambdas = np.logspace(start, end, num_lambdas, base=2)

    best_lambda, regressor, best_mse = find_best_lambda(lambdas, reg_setting, Xtrain, Ytrain, Xval, Yval)
    print(reg_setting + ' Best log2 lambda: ' + str(np.log2(best_lambda)))
    print("Validation MSE:", best_mse)

    reg_fit = regressor.fit(Xtrain, Ytrain)
    return reg_fit


def find_best_lambda(lambdas, reg_setting, Xtrain, Ytrain, Xval, Yval, fit_int=False):
    """ return best lambda and the regressor and validation mse"""
    best_lambda = 0
    best_mse = 100000

    for lambda_i in lambdas:
        if reg_setting == 'L1':  # lasso
            regressor = Lasso(alpha=lambda_i, max_iter=1000000, fit_intercept=fit_int)
        elif reg_setting == 'L2':  # ridge
            regressor = Ridge(alpha=lambda_i, max_iter=1000000, fit_intercept=fit_int)
        else:
            print('not valid reg _setting')

        regressor.fit(Xtrain, Ytrain)
        y_pred = regressor.predict(Xval)
        val_mse = mean_squared_error(Yval, y_pred)

        if val_mse <= best_mse:
            best_lambda = lambda_i
            best_mse = val_mse

    if reg_setting == 'L1':  # lasso
        regressor = Lasso(alpha=best_lambda, max_iter=1000000, fit_intercept=fit_int)
    elif reg_setting == 'L2':  # ridge
        regressor = Ridge(alpha=best_lambda, max_iter=1000000, fit_intercept=fit_int)
    return best_lambda, regressor, best_mse


def model_sel_RF(Xtrain, Ytrain, Xval, Yval):
    """ model selection for Random Forest"""
    # parameters
    n_trees = [10, 50, 100, 150]
    max_depth = [None, 1, 2, 3, 4]

    best_ntree = 0
    best_maxd = 0
    best_mse = 10000
    for n_tree in n_trees:
        for max_d in max_depth:
            reg_RF = RandomForestRegressor(n_estimators=n_tree, max_depth=max_d, random_state=100)
            reg_RF.fit(Xtrain, Ytrain)
            y_pred = reg_RF.predict(Xval)
            val_mse = mean_squared_error(Yval, y_pred)

            if val_mse < best_mse:
                best_mse = val_mse
                best_ntree = n_tree
                best_maxd = max_d
    print("Random Forest")
    print("num trees = ", best_ntree)
    print("max depth = ", best_maxd)
    print("Validation MSE:  %.3f " % best_mse)
    return reg_RF


def print_metrics(y_pred_test, y_true_test, y_pred_train, y_true_train, e_M=None):
    print("Training MSE: %0.3f" % mean_squared_error(y_true_train, y_pred_train))
    print("Test MSE: %.3f " % mean_squared_error(y_true_test, y_pred_test))
    print("Test Error: %.3f" % mean_absolute_error(y_true_test, y_pred_test))
    if e_M is not None:
        print("Generalization Error bound: <= %.3f" % (mean_absolute_error(y_true_test, y_pred_test) + e_M))
    print("\n")


def plot_scatter_pred_true(y_pred, y_true, titlestr):
    plt.clf()
    plt.scatter(y_pred, y_true)
    plt.ylabel("True ElimWeek")
    plt.xlabel("Predicted ElimWeek")
    plt.title(titlestr)
    plt.grid()
    plt.show()
