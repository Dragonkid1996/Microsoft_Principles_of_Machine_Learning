import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math

auto_prices = pd.read_csv("Auto_Data_Preped.csv")
auto_prices.columns
auto_prices.head()

#CREATE DUMMY VARIABLES FROM CATEGORICAL VARIABLES
print(auto_prices['body_style'].unique())
Features = auto_prices['body_style']
enc = preprocessing.LabelEncoder()
enc.fit(Features)
Features = enc.transform(Features)
print(Features)

ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(Features.reshape(-1,1))
Features = encoded.transform(Features.reshape(-1,1)).toarray()
print(Features[:10,:])

#%%
def encode_string(cat_feature):
    #FIRST ENCODE THE STRINGS TO NUMERIC CATEGORIES
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    #NOW APPLT ONE HOT ENCODING
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()

categorical_columns = ['fuel_type', 'aspiration', 'drive_wheels', 'num_of_cylinders']

for col in categorical_columns:
    temp = encode_string(auto_prices[col])
    Features = np.concatenate([Features, temp], axis = 1)

print(Features.shape)
print(Features[:2, :])

#ADD THE NUMERIC FEATURES
Features = np.concatenate([Features, np.array(auto_prices[['curb_weight', 'horsepower', 'city_mpg']])], axis = 1)
Features[:2, :]

#%%
#SPLIT THE DATASET
nr.seed(9988)
labels = np.array(auto_prices['log_price'])
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 40)
x_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
x_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

#%%
#RESCALE NUMERIC FEATURES
scaler = preprocessing.StandardScaler().fit(x_train[:,14:])
x_train[:,14:] = scaler.transform(x_train[:,14:])
x_test[:,14:] = scaler.transform(x_test[:,14:])
print(x_train.shape)
x_train[:5,:]

#%%
#CONSTRUCT THE LINEAR REGRESSION MODEL
lin_mod = linear_model.LinearRegression(fit_intercept = False)
lin_mod.fit(x_train, y_train)

print(lin_mod.intercept_)
print(lin_mod.coef_)

#%%
#EVALUATE THE MODEL
def print_metrics(y_true, y_predicted, n_parameters):
    #FIRST COMPUTE R^2 AND THE ADJUSTED R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    #PRINT THE USUAL METRICS AND THE R^2 VALUES
    print("Mean Square Error      = " + str(sklm.mean_squared_error(y_true, y_predicted)))
    print("Root Mean Square Error = " + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print("Mean Absolute Error    = " + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print("Median Absolute Error  = " + str(sklm.median_absolute_error(y_true, y_predicted)))
    print("R^2                    = " + str(r2))
    print("Adjusted R^2           = " + str(r2_adj))

y_score = lin_mod.predict(x_test)
print_metrics(y_test, y_score, 28)

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
def resid_qq(y_test, y_score):
    #FIRST COMPUTE VECTOR OF RESIDUALS
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    #NOW MAKE THE RESIDUAL PLOTS
    ss.probplot(resids.flatten(), plot = plt)
    plt.title("Residuals vs. predicted values")
    plt.xlabel("Predicted values")
    plt.ylabel("Residual")
    
resid_qq(y_test, y_score)

#%%
def resid_plot(y_test, y_score):
    #FIRST COMPUTE VECTOR OF RESIDUALS
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    #NOW MAKE THE RESIDUAL PLOTS
    sns.regplot(y_score, resids, fit_reg = False)
    plt.title("Residuals vs. predicted values")
    plt.xlabel("Predicted values")
    plt.ylabel("Residual")

resid_plot(y_test, y_score)

#%%
y_score_untransform = np.exp(y_score)
y_test_untransform = np.exp(y_test)
resid_plot(y_test_untransform, y_score_untransform)