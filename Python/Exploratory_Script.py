# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 18:55:45 2020

@author: Alexander
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("E:/GitHub/Credit-Scorecard-Project/Python/hmeq_clean.csv")

class ModelDetails():

    def __init__(self, intercept, coefs):
        """Because the scorecardpy package can only take a model class of
        LogisticRegression from the scikit-learn package this class is needed
        to hold the values from the statsmodels package.
        """

        self.intercept_ = [intercept]
        self.coef_ = [coefs.tolist()]
        
t = df.describe()       

corr_table = df.corr()
corr_table = corr_table.mask(np.tril(np.ones(corr_table.shape)).astype(np.bool))


plt.rcParams.update({'font.size': 16})
sns.set_style('darkgrid')

# Loan Plot
fig, ax = plt.subplots(figsize=[13.5,9])
loan_good = df.loc[df.BAD == 0, 'LOAN']
loan_bad = df.loc[df.BAD == 1, 'LOAN']
sns.distplot(loan_good, hist=False, label='Good', color='green')
sns.distplot(loan_bad, hist=False, label='Bad', color='red')
ax.set(ylim=(0, 0.00006))
ax.set(xlim=(0, 70000))
plt.savefig('figs/loan_dist.pdf')

# MORTDUE Plot
fig, ax = plt.subplots(figsize=[9,6])
goods = df.loc[df.BAD == 0, 'MORTDUE']
bads = df.loc[df.BAD == 1, 'MORTDUE']
sns.distplot(goods, hist=False, label='Good', color='green')
sns.distplot(bads, hist=False, label='Bad', color='red')
ax.set(ylim=(0, 0.000015))
ax.set(xlim=(0, 250000))
plt.savefig('figs/mortdue_dist.pdf')

# VALUE Plot
fig, ax = plt.subplots(figsize=[9,6])
goods = df.loc[df.BAD == 0, 'VALUE']
bads = df.loc[df.BAD == 1, 'VALUE']
sns.distplot(goods, hist=False, label='Good', color='green')
sns.distplot(bads, hist=False, label='Bad', color='red')
ax.set(ylim=(0, 0.000015))
ax.set(xlim=(0, 350000))
plt.savefig('figs/value_dist.pdf')

# REASON plot
reason_perc = df.groupby('REASON')['BAD'].value_counts(normalize=True)
reason_perc = reason_perc * 100
reason_perc = reason_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[9,6])
sns.barplot(x='REASON', y='Perc', hue='BAD', data=reason_perc, palette=['green', 'red'], ax=ax)
ax.set(ylim=(0,100))
for p in ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.1
    txt_y = p.get_height() + 1
    ax.text(txt_x,txt_y,txt)

plt.savefig('figs/reason_cat.pdf')

# JOB plot
reason_perc = df.groupby('JOB')['BAD'].value_counts(normalize=True)
reason_perc = reason_perc * 100
reason_perc = reason_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[15,6])
sns.barplot(x='JOB', y='Perc', hue='BAD', data=reason_perc, palette=['green', 'red'], ax=ax)
ax.set(ylim=(0,100))

for p in ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.05
    txt_y = p.get_height() + 1
    ax.text(txt_x,txt_y,txt)
    
plt.savefig('figs/job_cat.pdf')

# JOB plot
reason_perc = df.groupby('JOB')['BAD'].value_counts(normalize=True)
reason_perc = reason_perc * 100
reason_perc = reason_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[15,6])
sns.boxplot(x='BAD', y='YOJ', data=df, palette=['green', 'red'], ax=ax)
ax.set(ylim=(0,100))
    
plt.savefig('figs/job_cat.pdf')
    