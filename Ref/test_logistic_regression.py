###
# import packages
###
import numpy as np
import torch
import joblib

from sklearn.metrics import roc_auc_score, average_precision_score

###
# Data Load
###
X_train = np.load('./X_train_logistic.npy')
X_test = np.load('./X_test_logistic.npy')
y_train = np.load('./y_train.npy')
y_test = np.load('./y_test.npy')
# print('Data load Complete!')


# ###
# # Logistic regression pretrained model parameter load
# ###
model = joblib.load('model_parameter_logistic_regression.joblib') 


###
# Evaluate using AUROC and AUPRC
###
train_roc = round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 4)
test_roc = round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
train_prc = round(average_precision_score(y_train, model.predict_proba(X_train)[:, 1]), 4)
test_prc = round(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]), 4)


###
# Save the result of evaluation at './20214577_logistic_regression.txt'
###
with open('20214577_logistic_regression.txt', 'w') as f:
    f.write(f'20214577\n{train_roc}\n{train_prc}\n{test_roc}\n{test_prc}\n')