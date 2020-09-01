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

# Read data
df = pd.read_csv("E:\GitHub\Credit-Scorecard-Project\Python\hmeq_clean.csv")

# df.drop(['VALUE'], axis=1, inplace=True)

# Apply Transformations
df.LOAN = np.log(df.LOAN)
# df.MORTDUE = np.log(df.MORTDUE)
# df.VALUE = np.log(df.VALUE)
# Variable contained zeros so added 1 year to every observation
df.YOJ = np.log(df.YOJ + 1)

# Drop REASON and MORTDUE
df.drop(['REASON', 'MORTDUE'], axis=1, inplace=True)

# Create WOE bins
bins = sc.woebin(df, 'BAD', method='chimerge')

# Job was not binning correctly, this fixed that
break_list = {'JOB': df.JOB.unique().tolist()}
job_bins = sc.woebin(df, 'BAD', method='chimerge', x=['JOB'],
                     breaks_list=break_list)
bins['JOB'] = job_bins['JOB']

# Plot WOE bins
# fig, axs = plt.subplots(ncols=2)
# sc.woebin_plot(bins, figsize=[8,5])

# Print results of binning
# for k, bin_ in bins.items():
#     print(bins[k].iloc[:,0:-2].round(2).to_latex(index=False))
    
# split into train and test set
train, test = sc.split_df(df, 'BAD').values()

# Convert values into woe
train_woe = sc.woebin_ply(train, bins)
test_woe = sc.woebin_ply(test, bins)

# Add constant
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
test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['ks'])
test_perf = sc.perf_eva(y_test, test_pred, title = "test", plot_type=['roc'])

class ModelDetails():

    def __init__(self, intercept, coefs):
        """Because the scorecardpy package can only take a model class of
        LogisticRegression from the scikit-learn package this class is needed
        to hold the values from the statsmodels package.
        """

        self.intercept_ = [intercept]
        self.coef_ = [coefs.tolist()]
        
model = ModelDetails(fit.params[0], fit.params[1:])

# Create scorecard
card = sc.scorecard(bins, model, X_train.columns[1:], points0=800, pdo=50)

# Create scores
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)

# Plot scorecard
sc.perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)


df['SCORE'] = 0
df.SCORE.update(train_score.score)

## Score Train Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = (
    plt.subplots(3,
                sharex=True,
                gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                figsize=[15,8])
)
# Add a graph in each part
subset = df.loc[df.SCORE != 0]
goods = subset.loc[subset.BAD == 0, 'SCORE']
bads = subset.loc[subset.BAD == 1, 'SCORE']
sns.distplot(
    goods, hist=False, label='Good', color='green', ax=ax_dist)
sns.distplot(
    bads, hist=False, label='Bad', color='red', ax=ax_dist
).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
ax_dist.set_xlabel("SCORE: TRAIN",fontsize=20)

# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

divergence_train = (
    pow((goods.mean() - bads.mean()), 2) 
    / (0.5 * (goods.var() + bads.var()))
)

df['SCORE'] = 0
df.SCORE.update(test_score.score)

## Score Test Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = (
    plt.subplots(3,
                sharex=True,
                gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                figsize=[15,8])
)
# Add a graph in each part
subset = df.loc[df.SCORE != 0]
goods = subset.loc[subset.BAD == 0, 'SCORE']
bads = subset.loc[subset.BAD == 1, 'SCORE']
sns.distplot(
    goods, hist=False, label='Good', color='green', ax=ax_dist)
sns.distplot(
    bads, hist=False, label='Bad', color='red', ax=ax_dist
).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
ax_dist.set_xlabel("SCORE: TEST",fontsize=20)

# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

divergence_test = (
    pow((goods.mean() - bads.mean()), 2) 
    / (0.5 * (goods.var() + bads.var()))
)

