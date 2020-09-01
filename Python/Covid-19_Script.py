# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:09:36 2020

@author: Alexander
"""
import numpy as np
import pandas as pd
import scorecardpy as sc
import statsmodels.api as sm

# # Subsetting
# dat = pd.read_csv("E:\GitHub\Credit-Scorecard-Project\Python\latestdata.csv")
# df = dat.loc[:, ['ID', 'age', 'sex', 'symptoms', 'chronic_disease', 'outcome']]
# df = df.dropna(subset=['outcome', 'age', 'sex', 'symptoms'])
# df.to_csv(
# 'E:\GitHub\Credit-Scorecard-Project\Python\covid_subset.csv', index=False
# )

df = pd.read_csv("E:\GitHub\Credit-Scorecard-Project\Python\covid_subset.csv")

df['symptoms'] = df['symptoms'].str.replace(':', ',')
df['symptoms'] = df['symptoms'].str.replace(';', ',')
df['symptoms'] = df['symptoms'].str.replace(', ', ',')
df['symptoms'] = df['symptoms'].str.lower()

df['symptoms'] = df['symptoms'].str.replace(
    'acute respiratory disease syndrome', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'acute respiratory distress syndrome', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'acute respiratory disease', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'acute respiratory distress', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'acute respiratory failure', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'respiratory stress', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'respiratory symptoms', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace(
    'severe acute respiratory infection', 'respiratory problems')

df['symptoms'] = df['symptoms'].str.replace(
    'severe pneumonia', 'pneumonia')
df['symptoms'] = df['symptoms'].str.replace(
    'dry cough', 'cough')

sym = df['symptoms'].str.get_dummies(sep=',')
t = sym.sum()

df['chronic_disease'] = df['chronic_disease'].str.replace(':', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(';', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(', ', ',')
df['chronic_disease'] = df['chronic_disease'].str.lower()

df['chronic_disease'] = df['chronic_disease'].str.replace(
    ' for more than 20 years', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(
    ' surgery four years ago', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(
    ' accident ', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(
    ' for five years', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(
    'hypertensive', 'hypertension')
df['chronic_disease'] = df['chronic_disease'].str.replace(
    'hypertenstion', 'hypertension')

cd = df['chronic_disease'].str.get_dummies(sep=',')
v = cd.sum()

sym = sym.loc[:, sym.sum() >= 40]
cd = cd.loc[:, cd.sum() >= 40]

df = pd.concat([df, sym], axis=1)
df = pd.concat([df, cd], axis=1)

df['outcome'] = df['outcome'].str.lower()
df.outcome = np.where(
    df.outcome.isin(['death', 'dead', 'died', 'deceased']),
    1,
    0
)

df.sex = np.where(df.sex == 'male',
         1,
         0
)

df.drop(['ID', 'symptoms', 'chronic_disease'], axis=1, inplace=True)
df.age = df.age.str.replace('50-59', '55')
df.age = df.age.str.replace('60-69', '65')
df.age = df.age.str.replace('15-88', '45')
df.age = df.age.str.replace('80-89', '85')
df.age = df.age.str.replace('20-29', '25')
df.age = df.age.str.replace('40-49', '45')
df.age = df.age.str.replace('70-79', '75')
df.age = df.age.str.replace('90-99', '95')
df.age = df.age.str.replace('28-35', '31')
df.age = df.age.str.replace('80-', '80')

df.age = df.age.astype('float')
df.dropna(inplace=True)

# Reorder columns
cols = df.columns.tolist()
cols = cols[2:3] + cols[0:2] + cols[3:]

df = df[cols]

df.drop(['diabetes' ], axis=1, inplace=True)

bins = sc.woebin(df, 'outcome', method='chimerge')

cols = df.iloc[:, 2:].columns
break_list = {}
for col in cols:
    break_list[col] = [1.0]
    
bins.update(sc.woebin(df, 'outcome', method='chimerge', x=cols.tolist(), 
                      breaks_list=break_list))

# split into train and test set
train, test = sc.split_df(df, 'outcome').values()

# Convert values into woe
train_woe = sc.woebin_ply(train, bins)
test_woe = sc.woebin_ply(test, bins)

train_woe = sm.add_constant(train_woe)
test_woe = sm.add_constant(test_woe)

y_train = train_woe.loc[:,'outcome']
X_train = train_woe.loc[:,train_woe.columns != 'outcome']
y_test = test_woe.loc[:,'outcome']
X_test = test_woe.loc[:,train_woe.columns != 'outcome']

# Fit logit model
lr = sm.GLM(y_train, X_train, family=sm.families.Binomial())
fit = lr.fit()

fit.summary()

pred = np.array(fit.predict(X_test) > 0.5, dtype=float)
table = np.histogram2d(y_test, pred, bins=2)[0]
table = np.round(( table / table.sum() ) * 100, 2) 
