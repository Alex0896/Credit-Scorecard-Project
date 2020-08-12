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
        
t = df.describe()       

corr_table = df.corr()
corr_table = corr_table.mask(np.tril(np.ones(corr_table.shape)).astype(np.bool))



sns.set(style="whitegrid")

## Loan Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
loan_good = df.loc[df.BAD == 0, 'LOAN']
loan_bad = df.loc[df.BAD == 1, 'LOAN']
sns.distplot(loan_good, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(loan_bad, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(loan_good, ax=ax_box_good, color='green')
sns.boxplot(loan_bad, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/loan_dist.pdf')

## Mortdue Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
goods = df.loc[df.BAD == 0, 'MORTDUE']
bads = df.loc[df.BAD == 1, 'MORTDUE']
sns.distplot(goods, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(bads, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/mortdue_dist.pdf')

## Value Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
goods = df.loc[df.BAD == 0, 'VALUE']
bads = df.loc[df.BAD == 1, 'VALUE']
sns.distplot(goods, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(bads, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/value_dist.pdf')

# REASON plot
reason_perc = df.groupby('REASON')['BAD'].value_counts(normalize=True)
reason_perc = reason_perc * 100
reason_perc = reason_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[13.5,9])
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

## YOJ Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
goods = df.loc[df.BAD == 0, 'YOJ']
bads = df.loc[df.BAD == 1, 'YOJ']
sns.distplot(goods, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(bads, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/yoj_dist.pdf')

# YOJ plot
fig, ax = plt.subplots(figsize=[13.5,9])
sns.boxplot(x='BAD', y='YOJ', data=df, palette=['green', 'red'], ax=ax)
plt.savefig('figs/yoj_box.pdf')

# DEROG plot
derog_perc = df.groupby('DEROG')['BAD'].value_counts(normalize=True)
derog_perc = derog_perc * 100
derog_perc = derog_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[13.5,9])
sns.barplot(x='DEROG', y='Perc', hue='BAD', data=derog_perc, palette=['green', 'red'], ax=ax)

for p in ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.08
    txt_y = p.get_height() + 1
    ax.text(txt_x,txt_y,txt)

plt.savefig('figs/derog_cat.pdf')



# DELINQ plot
derog_perc = df.groupby('DELINQ')['BAD'].value_counts(normalize=True)
derog_perc = derog_perc * 100
derog_perc = derog_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[13.5,9])
sns.barplot(x='DELINQ', y='Perc', hue='BAD', data=derog_perc, palette=['green', 'red'], ax=ax)

for p in ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.08
    txt_y = p.get_height() + 1
    ax.text(txt_x,txt_y,txt)

plt.savefig('figs/delinq_cat.pdf')

## CLAGE Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
goods = df.loc[df.BAD == 0, 'CLAGE']
bads = df.loc[df.BAD == 1, 'CLAGE']
sns.distplot(goods, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(bads, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/clage_dist.pdf')

# NINQ plot
ninq_perc = df.groupby('NINQ')['BAD'].value_counts(normalize=True)
ninq_perc = ninq_perc * 100
ninq_perc = ninq_perc.rename('Perc').reset_index()

fig, ax = plt.subplots(figsize=[13.5,9])
sns.barplot(x='NINQ', y='Perc', hue='BAD', data=ninq_perc, palette=['green', 'red'], ax=ax)

for p in ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.05
    txt_y = p.get_height() + 1
    ax.text(txt_x,txt_y,txt)

plt.savefig('figs/ninq_cat.pdf')

## CLNO Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
goods = df.loc[df.BAD == 0, 'CLNO']
bads = df.loc[df.BAD == 1, 'CLNO']
sns.distplot(goods, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(bads, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/clno_dist.pdf')

## DEBTINC Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
goods = df.loc[df.BAD == 0, 'DEBTINC']
bads = df.loc[df.BAD == 1, 'DEBTINC']
sns.distplot(goods, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(bads, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(goods, ax=ax_box_good, color='green')
sns.boxplot(bads, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/debtinc_dist.pdf')

## Log LOAN Plot
fig, (ax_dist, ax_box_good, ax_box_bad) = plt.subplots(3,
                                                       sharex=True,
                                                       gridspec_kw={"height_ratios": (.80, .10, .10)}, 
                                                       figsize=[15,9]
                                                       )
# Add a graph in each part
loan_good = np.log(df.loc[df.BAD == 0, 'LOAN'])
loan_bad = np.log(df.loc[df.BAD == 1, 'LOAN'])
sns.distplot(loan_good, hist=False, label='Good', color='green', ax=ax_dist).set_xlabel("X Label",fontsize=20)
sns.distplot(loan_bad, hist=False, label='Bad', color='red', ax=ax_dist).tick_params(labelsize=16)
sns.boxplot(loan_good, ax=ax_box_good, color='green')
sns.boxplot(loan_bad, ax=ax_box_bad, color='red').tick_params(labelsize=16)
 
# Remove x axis name for the boxplot
ax_box_good.set(xlabel='')
ax_box_bad.set(xlabel='')

plt.savefig('figs/log_loan_dist.pdf')