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
print('There is' , df.shape[0] , 'rows')
print('There is' , df.shape[1] , 'columns')




"""
Data viz
"""
#heatmap of na
plt.figure(figsize=(10,10))
ax = plt.axes()
sns.heatmap(df.isna(),cbar=False)
ax.set_title('Heat map of NA')

#  Bar plot
#histograms

numerical_data = df.select_dtypes("number")
fig, ax = plt.subplots(ncols=9, nrows = 10, figsize = (20,10))
ax = ax.flatten()
index = 0
for col in numerical_data.columns:
    sns.kdeplot(x = df[col], ax= ax[index], fill= True, color = "orange")
    index += 1
fig.tight_layout()


plt.figure(figsize=(10,10))
ax = plt.axes()
sns.heatmap(data=df.corr(),  annot = True) #arg for color is cmap with for example cmap = 'seismic'
ax.set_title('Heat map')

fig, ax = plt.subplots(ncols = 5, nrows = 4, figsize = (20, 10))
ax = ax.flatten()
index = 0
for col in numerical_data.columns:
    sns.scatterplot(x = "TARGET_5Yrs", y = col, color = "#1aff66", ax = ax[index], data = df)
    index += 1
plt.tight_layout()



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df['TARGET_5Yrs']], axis = 1)
#print(finalDf.head())
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 20)
ax.set_ylabel('Principal Component 2', fontsize = 20)
ax.set_title('2 components PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['TARGET_5Yrs'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
