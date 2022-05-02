#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV


def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=5,random_state=50,shuffle=True)
    confusion_mat = np.zeros((8,8))
    acc = 0
    for i, (training_ids,test_ids) in enumerate(kf.split(dataset)):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        acc += accuracy_score(test_labels, predicted_labels)
        y_pred_proba = classifier.predict_proba(test_set)[::,1]
    acc/=5
    print(confusion_mat) 
    print(f"The accuracy is : {acc}")
    return {'acc': acc, 'cm': confusion_mat}




df = pd.read_csv('Data_Cortex_Nuclear.csv')
df_save = df.copy()
labels = df['class'].values # labels
#####" PREPROCESSING ##########""
#encoding of the class
label_encoder = LabelEncoder()
for col in ['Genotype', 'Treatment', 'Behavior', 'class']:
    df[col] = label_encoder.fit_transform(df[col])
df = df.fillna(df.mean())
X = df.drop(['class', 'MouseID'],axis=1).values
#X = StandardScaler().fit_transform(X)

y = df['class'].astype(int)


print(df.head())

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3)

svc = SVC(random_state=0, probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test,y_pred))
svc_eval = score_classifier(X, svc, labels)


xgb = XGBClassifier(colsample_bytree=0.3,
                    learning_rate=0.03,
                    max_depth=3,
                   n_estimators=700)
#Best parameters: {'colsample_bytree': 0.3, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 700}
lr = LogisticRegression() # TODO: try different solver
rf = RandomForestClassifier()
naive_bayes = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
dt = tree.DecisionTreeClassifier(random_state=0)
Ada = AdaBoostClassifier()


dict_models = {'SVC': svc,
               'Logistic Regression': lr,
               'Decision Tree':dt,
               'Random Forest': rf,
              'XGBoost': xgb,
               'AdaBoost':Ada,
               'Naive Bayes': naive_bayes,
               'KNN':knn}

acc_scores = []
names = []
for name, model in dict_models.items():
    print('---------------------------------')
    names.append(name)
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    evalu = score_classifier(X, model, labels)
    acc_scores.append(evalu['acc'])


plt.plot(names, acc_scores, color='green')
plt.xlabel('Models')
plt.ylabel('Accuracy score')
plt.show()
