#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from preprocessing import imputation, encoding

df = pd.read_csv('Data_Cortex_Nuclear.csv')
df_save = df.copy()

print(df.head())
df_vals = df.drop(['class'],axis=1).values
labels = df['class'].values # labels


# Check balancedness of the set
LABELS_unique = df['class'].unique()
count_classes = df['class'].value_counts(normalize=True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Class Distribution")
plt.xticks(range(len(LABELS_unique)), LABELS_unique)
plt.xlabel("Class")
plt.ylabel("Frequency")
print(df['class'].value_counts(normalize=True))


#encoding of the class
label_encoder = LabelEncoder()
for col in ['Genotype', 'Treatment', 'Behavior', 'class']:
    df[col] = label_encoder.fit_transform(df[col])
X = df_vals
#X = StandardScaler().fit_transform(df_vals)

print(df.head())




"""
Data viz
"""
#heatmap of na
plt.figure(figsize=(10,10))
ax = plt.axes()
sns.heatmap(df.isna(),cbar=False)
ax.set_title('Heat map of NA')

#  Bar plot

plt.figure(figsize=(10,10))
ax = plt.axes()
sns.heatmap(data=df.corr(),  annot = True) #arg for color is cmap with for example cmap = 'seismic'
ax.set_title('Heat map')


"""
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df['class']], axis = 1)
#print(finalDf.head())
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 20)
ax.set_ylabel('Principal Component 2', fontsize = 20)
ax.set_title('2 components PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

"""

plt.show()
