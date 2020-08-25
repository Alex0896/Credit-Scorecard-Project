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
# df.to_csv('covid_subset.csv', index=False)

df = pd.read_csv("E:\GitHub\Credit-Scorecard-Project\Python\covid_subset.csv")

df['symptoms'] = df['symptoms'].str.replace(':', ',')
df['symptoms'] = df['symptoms'].str.replace(';', ',')
df['symptoms'] = df['symptoms'].str.replace(', ', ',')
df['symptoms'] = df['symptoms'].str.lower()

df['symptoms'] = df['symptoms'].str.replace('acute respiratory disease syndrome', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('acute respiratory distress syndrome', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('acute respiratory disease', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('acute respiratory distress', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('acute respiratory failure', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('respiratory stress', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('respiratory symptoms', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('severe acute respiratory infection', 'respiratory problems')
df['symptoms'] = df['symptoms'].str.replace('severe pneumonia', 'pneumonia')
df['symptoms'] = df['symptoms'].str.replace('shortness of breath', 'difficulty breathing')
df['symptoms'] = df['symptoms'].str.replace('dry cough', 'cough')
df['symptoms'] = df['symptoms'].str.replace('fatigure', 'fatigue')
df['symptoms'] = df['symptoms'].str.replace('muscular soreness', 'fatigue')
df['symptoms'] = df['symptoms'].str.replace('transient fatigue', 'fatigue')
df['symptoms'] = df['symptoms'].str.replace('running nose', 'runny nose')

sym = df['symptoms'].str.get_dummies(sep=',')
t = sym.sum()

df['chronic_disease'] = df['chronic_disease'].str.replace(':', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(';', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(', ', ',')
df['chronic_disease'] = df['chronic_disease'].str.lower()

df['chronic_disease'] = df['chronic_disease'].str.replace(' for more than 20 years', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(' surgery four years ago', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(' accident ', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace(' for five years', ',')
df['chronic_disease'] = df['chronic_disease'].str.replace('hypertensive', 'hypertension')
df['chronic_disease'] = df['chronic_disease'].str.replace('hypertenstion', 'hypertension')
df['chronic_disease'] = df['chronic_disease'].str.replace('hypothyroidism', 'hyperthyroidism')
df['chronic_disease'] = df['chronic_disease'].str.replace('chronic bronchitis', 'chronic obstructive pulmonary disease')

cd = df['chronic_disease'].str.get_dummies(sep=',')
cd.sum()

sym = sym.loc[:, sym.sum() >= 10]
cd = cd.loc[:, cd.sum() >= 10]

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

df = df.loc[df.iloc[:, 2:].sum(axis=1) > 0]

# split into train and test set
# train, test = sc.split_df(df, 'outcome').values()

# df = sm.add_constant(df, has_constant='add')

y = df.loc[:, 'outcome']
X = df.loc[:, df.columns != 'outcome']

X = X.drop(['runny nose', 'septic shock', 'chronic kidney disease', 'diabetes'], axis=1)

# Fit logit model
lr = sm.GLM(y, X, family=sm.families.Binomial())
fit = lr.fit(maxiter=100)

fit.summary()

# Get probabilities
pred = fit.predict(X)


# Plot diagnositcs
test_perf = sc.perf_eva(y, pred, title = "test", plot_type=['ks'])
test_perf = sc.perf_eva(y, pred, title = "test", plot_type=['roc'])

# class ModelDetails():

#     def __init__(self, intercept, coefs):
#         """Because the scorecardpy package can only take a model class of
#         LogisticRegression from the scikit-learn package this class is needed
#         to hold the values from the statsmodels package.
#         """

#         self.intercept_ = [intercept]
#         self.coef_ = [coefs.tolist()]
        
# model = ModelDetails(fit.params[0], fit.params[1:])

# card = sc.scorecard(bins, model, X_train.columns[1:])

# score = sc.scorecard_ply(df, card, print_step=0)

# sc.perf_psi(
#   score = {'train':score},
#   label = {'train':y}
# )

pred = np.array(fit.predict(X) > 0.5, dtype=float)
table = np.histogram2d(y, pred, bins=2)[0]