from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import numpy as np

def train_dt(xTrain, yTrain, xTest, yTest):
    # --- FULL TRAINING DATA ---
    # create model object
    dt = DecisionTreeClassifier(max_depth=6)

    # fit model
    dt.fit(xTrain, yTrain)

    # evaluate on testing data
    yTrue = yTest
    yHatProb = dt.predict_proba(xTest)
    yHatClass = dt.predict(xTest)
    fullTestAuc = roc_auc_score(yTrue['label'], yHatProb[:, 1])
    fullTestAcc = accuracy_score(yTrue['label'], yHatClass)

    # --- REMOVE 5% TRAINING DATA ---
    # create model object
    dt1 = DecisionTreeClassifier(max_depth=6)

    # Randomly remove 5% of data
    remove_n = int(0.05 * len(xTrain))
    drop_indices = np.random.choice(xTrain.index, remove_n, replace=False)
    xTrain5 = xTrain.drop(drop_indices)
    yTrain5 = yTrain.drop(drop_indices)

    # fit model
    dt1.fit(xTrain5, yTrain5)

    # evaluate on testing data
    yHatProb5 = dt1.predict_proba(xTest)
    yHatClass5 = dt1.predict(xTest)
    TestAuc5 = roc_auc_score(yTrue['label'], yHatProb5[:, 1])
    TestAcc5 = accuracy_score(yTrue['label'], yHatClass5)

    # --- REMOVE 10% TRAINING DATA ---
    # create model object
    dt2 = DecisionTreeClassifier(max_depth=6)

    # Randomly remove 10% of data
    remove_n = int(0.1 * len(xTrain))
    drop_indices = np.random.choice(xTrain.index, remove_n, replace=False)
    xTrain10 = xTrain.drop(drop_indices)
    yTrain10 = yTrain.drop(drop_indices)

    # fit model
    dt2.fit(xTrain10, yTrain10)

    # evaluate on testing data
    yHatProb10 = dt2.predict_proba(xTest)
    yHatClass10 = dt2.predict(xTest)
    TestAuc10 = roc_auc_score(yTrue['label'], yHatProb10[:, 1])
    TestAcc10 = accuracy_score(yTrue['label'], yHatClass10)

    # --- REMOVE 20% TRAINING DATA ---
    # create model object
    dt3 = DecisionTreeClassifier(max_depth=6)

    # Randomly remove 20% of data
    remove_n = int(0.2 * len(xTrain))
    drop_indices = np.random.choice(xTrain.index, remove_n, replace=False)
    xTrain20 = xTrain.drop(drop_indices)
    yTrain20 = yTrain.drop(drop_indices)

    # fit model
    dt3.fit(xTrain20, yTrain20)

    # evaluate on testing data
    yHatProb20 = dt3.predict_proba(xTest)
    yHatClass20 = dt3.predict(xTest)
    TestAuc20 = roc_auc_score(yTrue['label'], yHatProb20[:, 1])
    TestAcc20 = accuracy_score(yTrue['label'], yHatClass20)

    df = pd.DataFrame(
        [['Full Training Data', fullTestAuc, fullTestAcc],
         ['5% Removed', TestAuc5, TestAcc5],
         ['10% Removed', TestAuc10, TestAcc10],
         ['20% Removed', TestAuc20, TestAcc20]],
        columns=['Model', 'Test AUROC', 'Test Accuracy']
    )

    return df

def train_knn(xTrain, yTrain, xTest, yTest):
    # --- FULL TRAINING DATA ---
    # create model object
    knn = KNeighborsClassifier(n_neighbors=21)

    # fit model
    knn.fit(xTrain, np.ravel(yTrain,order='C'))

    # evaluate on testing data
    yTrue = yTest
    yHatProb = knn.predict_proba(xTest)
    yHatClass = knn.predict(xTest)
    fullTestAuc = roc_auc_score(yTrue['label'], yHatProb[:,1])
    fullTestAcc = accuracy_score(yTrue['label'], yHatClass)

    # --- REMOVE 5% TRAINING DATA ---
    # create model object
    knn1 = KNeighborsClassifier(n_neighbors=21)

    # Randomly remove 5% of data
    remove_n = int(0.05 * len(xTrain))
    drop_indices = np.random.choice(xTrain.index, remove_n, replace=False)
    xTrain5 = xTrain.drop(drop_indices)
    yTrain5 = yTrain.drop(drop_indices)

    # fit model
    knn1.fit(xTrain5, np.ravel(yTrain5,order='C'))

    # evaluate on testing data
    yHatProb5 = knn1.predict_proba(xTest)
    yHatClass5 = knn1.predict(xTest)
    TestAuc5 = roc_auc_score(yTrue['label'], yHatProb5[:, 1])
    TestAcc5 = accuracy_score(yTrue['label'], yHatClass5)

    # --- REMOVE 10% TRAINING DATA ---
    # create model object
    knn2 = KNeighborsClassifier(n_neighbors=21)

    # Randomly remove 10% of data
    remove_n = int(0.1 * len(xTrain))
    drop_indices = np.random.choice(xTrain.index, remove_n, replace=False)
    xTrain10 = xTrain.drop(drop_indices)
    yTrain10 = yTrain.drop(drop_indices)

    # fit model
    knn2.fit(xTrain10, np.ravel(yTrain10,order='C'))

    # evaluate on testing data
    yHatProb10 = knn2.predict_proba(xTest)
    yHatClass10 = knn2.predict(xTest)
    TestAuc10 = roc_auc_score(yTrue['label'], yHatProb10[:, 1])
    TestAcc10 = accuracy_score(yTrue['label'], yHatClass10)

    # --- REMOVE 20% TRAINING DATA ---
    # create model object
    knn3 = KNeighborsClassifier(n_neighbors=21)

    # Randomly remove 20% of data
    remove_n = int(0.2 * len(xTrain))
    drop_indices = np.random.choice(xTrain.index, remove_n, replace=False)
    xTrain20 = xTrain.drop(drop_indices)
    yTrain20 = yTrain.drop(drop_indices)

    # fit model
    knn3.fit(xTrain20, np.ravel(yTrain20,order='C'))

    # evaluate on testing data
    yHatProb20 = knn3.predict_proba(xTest)
    yHatClass20 = knn3.predict(xTest)
    TestAuc20 = roc_auc_score(yTrue['label'], yHatProb20[:, 1])
    TestAcc20 = accuracy_score(yTrue['label'], yHatClass20)

    df = pd.DataFrame(
        [['Full Training Data', fullTestAuc, fullTestAcc],
        ['5% Removed', TestAuc5, TestAcc5],
        ['10% Removed', TestAuc10, TestAcc10],
        ['20% Removed', TestAuc20, TestAcc20]],
        columns=['Model', 'Test AUROC', 'Test Accuracy']
    )

    return df


def opt_param(xFeat, y):
    best_knn = [-float("inf"), None]
    best_dt = [-float("inf"), None]
    for k in range(2, 11):
        # KNN
        knn = GridSearchCV(
            KNeighborsClassifier(),
            [{'n_neighbors': range(1, 100, 5), 'metric': ['euclidean', 'manhattan']}], cv=k, scoring='roc_auc')
        knn.fit(xFeat, y['label'])
        y_true, y_pred = y, knn.predict(xFeat)
        knnBestParam = knn.best_params_
        # print(knnBestParam)
        means = knn.cv_results_['mean_test_score']

        for i, mean in enumerate(means):
            if mean > best_knn[0]:
                best_knn = [mean, knnBestParam, k]


        # Decision Tree
        dt = GridSearchCV(
            DecisionTreeClassifier(),
            [{'max_depth': range(1, 100, 5)}], cv=k, scoring='roc_auc')
        dt.fit(xFeat, y['label'])
        # y_true, y_pred = yTest, dt.predict(xTest)
        dtBestParam = dt.best_params_
        means = dt.cv_results_['mean_test_score']

        for i, mean in enumerate(means):
            if mean > best_dt[0]:
                best_dt = [mean, dtBestParam, k]

    return {
        "K-NN Optimal K": best_knn[1]['n_neighbors'],
        "DT Optimal Max Depth":best_dt[1]['max_depth'],
        "K (K Fold) Value Used (KNN)":best_knn[2],
        "K (K Fold) Value Used (DT)": best_dt[2]
    }


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()

    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    best_params = opt_param(xTrain, yTrain)
    print(best_params)

    knn_eval = train_knn(xTrain, yTrain, xTest, yTest)
    print(knn_eval)

    dt_eval = train_dt(xTrain, yTrain, xTest, yTest)
    print(dt_eval)

if __name__ == "__main__":
    main()

