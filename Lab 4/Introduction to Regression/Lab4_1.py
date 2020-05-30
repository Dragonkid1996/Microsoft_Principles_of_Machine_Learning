import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn import preprocessing
from sklearn import linear_model
import scipy.stats as ss
import seaborn as sns
import math

nr.seed(34567)
x = np.arange(start = 0.0, stop = 10.0, step = 0.1)
y = np.add(x, nr.normal(scale = 1.0, size = x.shape[0]))

sns.regplot(x, y, fit_reg = False)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Data for regression")

#%%
#SPLIT THE DATASET
nr.seed(9988)
indx = range(len(x))
indx = ms.train_test_split(indx, test_size = 50)
x_train = np.ravel(x[indx[0]])
y_train = np.ravel(y[indx[0]])
x_test = np.ravel(x[indx[1]])
y_test = np.ravel(y[indx[1]])

#SCALE NUMERIC FEATURES - TRAINING DATA NOT TEST DATA
scaler = preprocessing.StandardScaler().fit(x_train.reshape(-1,1))
x_train = scaler.transform(x_train.reshape(-1,1))
y_train = scaler.transform(y_train.reshape(-1,1))

#TRAIN THE REGRESSION MODEL
lin_mod = linear_model.LinearRegression()
lin_mod.fit(x_train.reshape(-1,1), y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

def plot_regression(x, y_score, y):
    #PLOT THE RESULT
    sns.regplot(x, y, fit_reg = False)
    plt.plot(x, y_score, c = 'red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fit of model to test data")
    
y_score = lin_mod.predict(x_test.reshape(-1, 1))
plot_regression(x_test, y_score, y_test)

#%%
def print_metrics(y_true, y_predicted, n_parameters):
    #FIRST COMPUTE R^2 AND THE ADJUSTED R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (y_true.shape[0] - 1)/(y_true.shape[0] - n_parameters - 1) * (1 - r2)
    
    #PRINT THE USUAL METRICS AND THE R^2 VALUES
    print("Mean Square Error      = " + str(sklm.mean_squared_error(y_true, y_predicted)))
    print("Root Mean Square Error = " + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print("Mean Absolute Error    = " + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print("Median Absolute Error  = " + str(sklm.median_absolute_error(y_true, y_predicted)))
    print("R^2                    = " + str(r2))
    print("Adjusted R^2           = " + str(r2_adj))
    
print_metrics(y_test, y_score, 2)

#%%
def hist_resids(y_test, y_score):
    #FIRST COMPUTE VECTOR OF RESIDUALS
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    #NOW MAKE THE RESIDUAL PLOTS
    sns.distplot(resids)
    plt.title("Histogram of residuals")
    plt.xlabel("Residual value")
    plt.ylabel("Count")

hist_resids(y_test, y_score)

#%%
def resid_qq(y_test, y_score): #QUANTILE-QUANTILE PLOT
    #FIRST COMPUTE VECTOR OF RESIDUALS
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    #NOW MAKE THE RESIDUAL PLOTS
    ss.probplot(resids.flatten(), plot = plt)
    plt.title("Residuals vs. predicted values")
    plt.xlabel("Quantiles of standard Normal distribution")
    plt.ylabel("Quantiles of residuals")
    
resid_qq(y_test, y_score)

#%%
def resid_plot(y_test, y_score):
    #FIRST COMPUTE VECTOR OF RESIDUALS
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    #NOW MAKE THE RESIDUAL PLOTS
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title("Residuals vs. predicted values")
    plt.xlabel("Predicted values")
    plt.ylabel("Residual")

resid_plot(y_test, y_score)