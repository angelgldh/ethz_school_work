# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

# %%
def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    
    X_linear = X
    X_squared = np.square(X)
    X_exp = np.exp(X)
    X_cos = np.cos(X)
    X_constant = np.ones(shape = (X.shape[0],1))

    X_transformed = np.concatenate([X_linear, X_squared, X_exp, X_cos,X_constant], axis=1)

    assert X_transformed.shape == (700, 21)
    return X_transformed


# %%
def fit(X, y, method):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # method = "lm"

    if method == "lm":
        model = LinearRegression()
        
    elif method == "Ridge":
        model = Ridge(alpha=0.1)
        
    elif method == "Lasso":
        model = Lasso(alpha=0.1)
        
    elif method == "RidgeCV":
        model = RidgeCV(cv=10, alphas= np.linspace(0,10, 100), fit_intercept=False )
        
    elif method == "LassoCV":
        model = LassoCV(cv=10, alphas= np.linspace(0,10, 100), fit_intercept=False )
        
    model.fit(X_transformed,y)
    w = model.coef_

    print(f"{method}. (training) RMSE: {calculate_RMSE(w, X_transformed, y)}")

    assert w.shape == (21,)
    return w


# %%
def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    RMSE = 0
    n, p = X.shape
    
    y_pred = X @ w
    RMSE = np.sqrt((1/n)*np.sum((y - y_pred)**2) )

    assert np.isscalar(RMSE)
    return RMSE

# %%
# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y, method = "RidgeCV")
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")


