import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

#EXAMPLE LOGISTIC FUNCTION
xseq = np.arange(-7, 7, 0.1)

logistic = [math.exp(v)/(1 + math.exp(v)) for v in xseq]

plt.plot(xseq, logistic, color = 'red')
plt.plot([-7,7], [0.5,0.5], color = 'blue')
plt.plot([0,0], [0,1], color = 'blue')
plt.title('Logistic function for two-class classification')
plt.ylabel('log likelihood')
plt.xlabel('Value of output from linear regression')

credit = pd.read_csv('German_Credit_Preped.csv')
print(credit.shape)
print(credit.head())

credit_counts = credit[['credit_history', 'bad_credit']].groupby('bad_credit').count()
print(credit_counts)

#%%
#PREPARE THE DATA FOR MODEL
labels = np.array(credit['bad_credit'])

def encode_string(cat_features):
    #FIRST ENCODE THE STRINGS TO NUMERIC CATEGORIES
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    #NOW APPLY ONE HOT ENCODING
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns = ['credit_history', 'purpose', 'gender_status',
                       'time_in_residence', 'property']

Features = encode_string(credit['checking_account_status'])
for col in categorical_columns:
    temp = encode_string(credit[col])
    Features = np.concatenate([Features, temp], axis = 1)
    
print(Features.shape)
print(Features[:2, :])

Features = np.concatenate([Features, np.array(credit[['loan_duration_mo', 'loan_amount',
                                                      'payment_pcnt_income', 'age_yrs']])], axis = 1)
print(Features.shape)
print(Features[:2, :])

#%%
#RANDOMLY SAMPLE CASES TO CREATE INDEPENDENT TRAINING AND TEST DATA
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

scaler = preprocessing.StandardScaler().fit(X_train[:,34:])
X_train[:,34:] = scaler.transform(X_train[:,34:])
X_test[:,34:] - scaler.transform(X_test[:,34:])
X_train[:2,]

#%%
#CONSTRUCT THE REGRESSION MODEL
logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(X_train, y_train)

print(logistic_mod.intercept_)
print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

#%%
#SCORE AND EVALUATE THE MODEL
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

#%%
def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


    
print_metrics(y_test, scores)

#%%
#PLOT ROC CURVE WITH AREA UNDER CURVE(AUC)
def plot_auc(labels, probs):
    #COMPUTE FALSE POSITIVE RATE, TRUE POSITIVE RATE AND THRESHOLD ALONG WITH AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    
    #PLOT THE RESULT
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

plot_auc(y_test, probabilities)

#%%
probs_positive = np.concatenate((np.ones((probabilities.shape[0], 1)),
                                 np.zeros((probabilities.shape[0], 1))),
                                 axis = 1)
scores_positive = score_model(probs_positive, 0.5)
print_metrics(y_test, scores_positive)
plot_auc(y_test, probs_positive)

#%%
#COMPUTE A WEIGHTED MODEL
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.45, 1:0.55})
logistic_mod.fit(X_train, y_train)

probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])

scores = score_model(probabilities, 0.5)
print_metrics(y_test, scores)
plot_auc(y_test, probabilities)

#%%
#FIND A BETTER THRESHOLD
def test_threshold(probs, labels, threshold):
    scores = score_model(probs, threshold)
    print('')
    print('For threshold = ' + str(threshold))
    print_metrics(labels, scores)

thresholds = [0.45, 0.40, 0.35, 0.3, 0.25]
for t in thresholds:
    test_threshold(probabilities, y_test, t)