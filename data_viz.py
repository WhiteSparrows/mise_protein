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
from sklearn.pipeline import make_pipeline

df = pd.read_csv('Data_Cortex_Nuclear.csv')
df_save = df.copy()

print(df.head())
labels = df['class'].values # labels


# Check balancedness of the set
LABELS_unique = df['class'].unique()
count_classes = df['class'].value_counts(normalize=True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Class Distribution")
plt.xticks(range(len(LABELS_unique)), LABELS_unique)
plt.xlabel("Class")
plt.ylabel("Frequency")
print('Repartition of classes')
print(df['class'].value_counts(normalize=True))

for col in df.select_dtypes("object"):
    plt.figure()
    df[col].value_counts().plot.pie()


#heatmap of na
plt.figure(figsize=(10,10))
ax = plt.axes()
sns.heatmap(df.isna(),cbar=False)
ax.set_title('Heat map of NA')

#####" PREPROCESSING ##########""
#encoding of the class
label_encoder = LabelEncoder()
for col in ['Genotype', 'Treatment', 'Behavior', 'class']:
    df[col] = label_encoder.fit_transform(df[col])
df = df.drop('MouseID', axis=1)
#df = df.dropna(axis=0)
df = df.fillna(df.mean())
#X = StandardScaler().fit_transform(df_vals)

X = df.drop('class', axis=1)
y = df['class'].astype(int)


print(df.head())

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3)


"""
Data viz
"""

#  Bar plot

plt.figure(figsize=(10,10))
ax = plt.axes()
sns.heatmap(data=df.corr(),  annot = False,cmap="YlGnBu", vmin=-1, vmax=+1) #arg for color is cmap with for example cmap = 'seismic'
ax.set_title('Correlation heat map')

#PCA
preprocessor = make_pipeline(StandardScaler())

PCAPipeline = make_pipeline(preprocessor, PCA(n_components=2,random_state=0))



PCA_df = pd.DataFrame(PCAPipeline.fit_transform(X))
PCA_df = pd.concat([PCA_df, df['class']], axis=1)


plt.figure(figsize=(8,8))
sns.scatterplot(PCA_df[0],PCA_df[1],hue=PCA_df['class'],palette=sns.color_palette("Paired", 8))
plt.title('PCA')
plt.show()
