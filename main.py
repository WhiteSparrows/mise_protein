#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data_Cortex_Nuclear.csv')
df_save = df.copy()
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.keys())
for col in df.columns : 
    print(col, df[col].isnull().sum() / df[col].shape[0])

plt.figure(figsize=(10,10))
sns.heatmap(df.isna(),cbar=False)
plt.show()
print("These are the different class", df['class'].unique())
df = df.dropna(axis=0)
print(df.shape)
for col in df.columns : 
    print(col, df[col].isnull().sum() / df[col].shape[0])



