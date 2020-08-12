# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 19:16:46 2020

@author: Alexander
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("E:/GitHub/Credit-Scorecard-Project/Python/hmeq.csv")

# Get number of nulls in each row
na_in_rows = df.isna().sum(axis=1)

df = df.loc[~(na_in_rows >= 4)]

# Remove the top 5% quantile from each variable
cols = df.select_dtypes([np.number]).columns
cols = cols[1:]
for col in cols:
    df = df.loc[
        (df[col] < df[col].quantile(0.99))
        | (df[col].isna())
    ]
    
t = df.describe()

## Mortgage impute
df = sm.add_constant(df, has_constant='add')
mort_missing_with_value = df.loc[(df.MORTDUE.isna()) & ~(df.VALUE.isna())] # 336
mort_with_value = df.loc[~(df.MORTDUE.isna()) & ~(df.VALUE.isna())] # 5201
mort_ols = sm.OLS(mort_with_value.MORTDUE, mort_with_value[['const', 'VALUE']])
mort_fit = mort_ols.fit()
predictions = mort_fit.predict(mort_missing_with_value[['const', 'VALUE']])

# Get mean before updating
mort_mean = df.MORTDUE.mean()

df.MORTDUE.update(predictions)
df.MORTDUE.fillna(mort_mean, inplace=True) # Fill any remaining with original mean

## Value impute
value_missing_with_mort = df.loc[(df.VALUE.isna()) & ~(df.MORTDUE.isna())] # 84
value_with_mort = df.loc[~(df.VALUE.isna()) & ~(df.MORTDUE.isna())] # 5537
value_ols = sm.OLS(value_with_mort.VALUE, value_with_mort[['const', 'MORTDUE']])
value_fit = value_ols.fit()
predictions = value_fit.predict(value_missing_with_mort[['const', 'MORTDUE']])

# Get mean before updating
value_mean = df.VALUE.mean()

df.VALUE.update(predictions)
df.VALUE.fillna(value_mean, inplace=True) # Fill any remaining with original mean

df.iloc[:, :-1] = df.iloc[:, :-1].fillna(df.median())
# df = df.fillna(df.median())

## DEBTINC impute
df_ = df.drop(['BAD', 'REASON', 'JOB'], axis=1)

debt_missing_with_others = df_.loc[df_.DEBTINC.isna()] # 991
debt_with_others = df_.loc[~df_.DEBTINC.isna()] # 4041
debt_ols = sm.OLS(debt_with_others.DEBTINC, debt_with_others.iloc[:,:-1])
debt_fit = debt_ols.fit()
predictions = debt_fit.predict(debt_missing_with_others.iloc[:,:-1])

df.DEBTINC.update(predictions)

df.drop('const', axis=1, inplace=True)

# Impute categorical variables
np.random.seed(100) # Set seed for reproducability
# Reason
reason_weights = df.REASON.value_counts() # DebtCon 3801, HomeImp 1682
reason_categories = reason_weights.index.tolist()
reason_probs = (reason_weights / reason_weights.sum()).tolist()


reason_missing = df.loc[df.REASON.isna(), 'REASON']
reason_missing = reason_missing.apply(
    lambda x: np.random.choice(reason_categories, p=reason_probs)
)
df.REASON.update(reason_missing)
df.REASON.value_counts() # DebtCon 3902, HomeImp 1719

# Job
df.JOB.value_counts()
# Other 2267, ProfExe 1251, Office 938, Mgr 740, Self 189, Sales 109
job_weights = df.JOB.value_counts()
job_categories = job_weights.index.tolist()
job_probs = (job_weights / job_weights.sum()).tolist()


job_missing = df.loc[df.JOB.isna(), 'JOB']
job_missing = job_missing.apply(
    lambda x: np.random.choice(job_categories, p=job_probs)
)
df.JOB.update(job_missing)
df.JOB.value_counts()
# Other 2313, ProfExe 1283, Office 958, Mgr 757, Self 197, Sales 113

df.isna().sum() # 0 missing values remaining

df.to_csv('hmeq_clean.csv', index=False)
