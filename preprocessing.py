#!/usr/bin/env python
# coding=utf-8

def encoding(df):
    code = {'Control':1,
            'Ts65Dn':0,
            'Memantine':1,
            'Saline':0,
            'C/S':0,
            'S/C':1,
            'c-CS-m':0,
            'c-SC-m':1,
            'c-CS-s':2,
            'c-SC-s':3,
            't-CS-m':4,
            't-SC-m':5,
            't-CS-s':6,
            't-SC-s':7,
           }
    for col in df.select_dtypes('object'):
        df.loc[:,col]=df[col].map(code)

    return df

def imputation(df):

    #df = df.dropna(axis=0)
    df = df.fillna(df.mean())

    return df
