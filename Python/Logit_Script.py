# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:01:18 2020

@author: Alexander
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scorecardpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("E:\GitHub\Credit-Scorecard-Project\Python\hmeq_clean.csv")

df.LOAN = np.log(df.LOAN)
df.MORTDUE = np.log(df.MORTDUE)
df.VALUE = np.log(df.VALUE)

bins = sc.woebin(df, 'BAD')

for k, bin_ in bins.items():
    print(k)
    print(bin_[['woe', 'bin_iv','total_iv']])
    
# split into train and test set
train, test = sc.split_df(df, 'BAD').values()

# Convert values into woe
train_woe = sc.woebin_ply(train, bins)
test_woe = sc.woebin_ply(test, bins)

train_woe = sm.add_constant(train_woe)
test_woe = sm.add_constant(test_woe)

y_train = train_woe.loc[:,'BAD']
X_train = train_woe.loc[:,train_woe.columns != 'BAD']
y_test = test_woe.loc[:,'BAD']
X_test = test_woe.loc[:,train_woe.columns != 'BAD']

# Fit logit model
lr = sm.GLM(y_train, X_train, family=sm.families.Binomial())
fit = lr.fit()

fit.summary()

# Get probabilities
train_pred = fit.predict(X_train)
test_pred = fit.predict(X_test)


# Plot diagnositcs
train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")

class ModelDetails():

    def __init__(self, intercept, coefs):
        """Because the scorecardpy package can only take a model class of
        LogisticRegression from the scikit-learn package this class is needed
        to hold the values from the statsmodels package.
        """

        self.intercept_ = [intercept]
        self.coef_ = [coefs.tolist()]
        
model = ModelDetails(fit.params[0], fit.params[1:])

card = sc.scorecard(bins, model, X_train.columns[1:])

train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)

sc.perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)