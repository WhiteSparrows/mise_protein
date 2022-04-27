#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=5,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    prec = 0
    f1 = 0
    acc = 0
    auc = 0
    fpr = []
    tpr = []
    tprs = []
    aucs = []
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    for i, (training_ids,test_ids) in enumerate(kf.split(dataset)):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        viz = RocCurveDisplay.from_estimator(
            classifier,
            test_set,
            test_labels,
            name=f"ROC fold {i} for model {classifier}",
            alpha=0.3,
            lw=1
        )
        plt.close()
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic example",
        )
        ax.legend(loc="lower right")
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        prec += precision_score(test_labels, predicted_labels)
        f1 += f1_score(test_labels, predicted_labels)
        acc += accuracy_score(test_labels, predicted_labels)
        y_pred_proba = classifier.predict_proba(test_set)[::,1]
        fpr, tpr, _ = metrics.roc_curve(test_labels, y_pred_proba)
        auc += metrics.roc_auc_score(test_labels, y_pred_proba)
    recall/=5
    prec/=5
    f1/=5
    acc/=5
    auc/=5
    plt.close()
    print(confusion_mat)
    print(f"The accuracy is : {acc}")
    print(f"The recall score is : {recall}")
    return {'acc': acc, 'prec': prec, 'rec': recall, 'f1': f1,  
            'fpr': mean_fpr, 'tpr': mean_tpr, 'auc': auc, 'cm': confusion_mat}




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


# TODO: plot all classes, NA distribution, pca ?



