# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# %%
def ridge_cost_gradient(X, y, w, alpha):
    n = len(y)
    y_pred = X @ w
    cost = (1 / (2 * n)) * np.sum((y_pred - y) ** 2) + (alpha / 2) * np.sum(w[1:] ** 2)
    # To calculate the gradient, one needs to take the derivative of the cost with respect to the weight vector
    gradient = (1 / n) * (X.T @ (y_pred - y)) + alpha * np.hstack(([0], w[1:])) * (1 / n)
    return cost, gradient    

# %%
def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    n, p = X.shape
    w = np.zeros((p,))
    method = "analytical"

    if method =="gradient_descent":
        # Option 1: gradient descent 
        # To fit Ridge regression: gradient descent will be applied
        learning_rate = 1e-6
        max_iter = 2000
        for ii in range(max_iter):
            cost, gradient = ridge_cost_gradient(X, y, w, alpha = lam)
            w -= learning_rate*gradient
            # if (ii % 100 == 0):
            #    print(f"Iteration {ii} completed. Cost: {cost}")
    elif method == "analytical":
        # Option 2: closed-fomr solution
        A = np.linalg.inv(np.dot(X.T, X) + lam * np.identity(n=p))
        B = X.T @ y
        w =  np.dot(A,B)

    assert w.shape == (13,)
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
def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    kf = KFold(n_splits = n_folds, shuffle=True, random_state=42)

    # Evaluate the obtained RMSE per every value of the proposed regularization parameters
    for ii in range(len(lambdas)):
        lam = lambdas[ii]

        RMSE_at_this_lambda = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            w = fit(X_train, y_train, lam)
            RMSE_at_this_lambda.append(calculate_RMSE(w, X_test, y_test))
            
        RMSE_mat[:, ii] = RMSE_at_this_lambda


    assert RMSE_mat.shape == (n_folds, len(lambdas))

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# %%
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")



