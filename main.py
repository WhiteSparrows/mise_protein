#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from xgboost import XGBClassifier

df = pd.read_csv('Data_Cortex_Nuclear.csv')
print(df.head())
print(df.shape)
print(df.info())
for col in df.columns : 
    print(col, df[col].isnull().sum() / df[col].shape[0])



