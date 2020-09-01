# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 19:16:46 2020

@author: Alexander
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Read data
df = pd.read_csv("E:/GitHub/Credit-Scorecard-Project/Python/hmeq.csv")

# Remove the top 1% quantile from each variable
cols = df.select_dtypes([np.number]).columns
cols = cols[1:]
for col in cols:
    df = df.loc[
        (df[col] < df[col].quantile(0.99))
        | (df[col].isna())
    ]

# Get summary details    
t = df.describe()

## Mortgage impute
df = sm.add_constant(df, has_constant='add')

mort_missing_with_value = df.loc[
    (df.MORTDUE.isna()) 
    & ~(df.VALUE.isna())
]
mort_with_value = df.loc[
    ~(df.MORTDUE.isna()) 
    & ~(df.VALUE.isna())
]

mort_ols = sm.OLS(mort_with_value.MORTDUE,
                  mort_with_value[['const', 'VALUE']])
mort_fit = mort_ols.fit()

predictions = mort_fit.predict(
    mort_missing_with_value[['const', 'VALUE']]
)

# Get mean before updating
mort_mean = df.MORTDUE.mean()

df.MORTDUE.update(predictions)
# Fill any remaining with original mean
df.MORTDUE.fillna(mort_mean, inplace=True)

## Value impute
value_missing_with_mort = df.loc[
    (df.VALUE.isna()) 
    & ~(df.MORTDUE.isna())
]
value_with_mort = df.loc[
    ~(df.VALUE.isna()) 
    & ~(df.MORTDUE.isna())
]

value_ols = sm.OLS(value_with_mort.VALUE, 
                   value_with_mort[['const', 'MORTDUE']])
value_fit = value_ols.fit()

predictions = value_fit.predict(
    value_missing_with_mort[['const', 'MORTDUE']]
)

# Get mean before updating
value_mean = df.VALUE.mean()

df.VALUE.update(predictions)
# Fill any remaining with original mean
df.VALUE.fillna(value_mean, inplace=True) 

# Drop constant column
df.drop('const', axis=1, inplace=True)

df.iloc[:, :-1] = df.iloc[:, :-1].fillna(df.median())

# Drop DEBTINC
df = df.drop('DEBTINC', axis=1)

# Drop any remaining missing values
df.dropna(inplace=True)

# 0 missing values remaining
df.isna().sum()

# Save cleaned data
df.to_csv(
    r"E:\GitHub\Credit-Scorecard-Project\Python\hmeq_clean.csv", 
    index=False
)
