import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def normalize(xTrain, xTest):
    stdScale = StandardScaler()
    xTrainScaled = stdScale.fit_transform(xTrain)
    xTestScaled = stdScale.fit_transform(xTest)

    xTrainScaled = pd.DataFrame(xTrainScaled, columns=xTrain.columns)
    xTestScaled = pd.DataFrame(xTestScaled, columns=xTest.columns)

    return xTrainScaled, xTestScaled

def logit_unreg(xTrain, yTrain, xTest):
    lg = LogisticRegression()
    lg.fit(xTrain, yTrain.to_numpy().ravel())
    yPred = lg.predict_proba(xTest)
    return yPred

def pca_fit(xTrain, xTest):
    pca = PCA(n_components=9) # n = 9
    xTrainPCA = pca.fit_transform(xTrain)
    xTestPCA = pca.transform(xTest)
    cum_variance = 0
    for ratio in pca.explained_variance_ratio_:
        cum_variance += ratio
    df = pd.DataFrame(pca.components_, columns=xTrain.columns, index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])
    df = df.iloc[0:3,:]
    print("Cumulative Variance:", cum_variance)
    print(df)
    return xTrainPCA, xTestPCA

def logit_pca(xTrain, yTrain, xTest):
    lg = LogisticRegression()
    lg.fit(xTrain, yTrain.to_numpy().ravel())
    yPred = lg.predict_proba(xTest)
    return yPred

def main():
    """
    Main file to run from the command line.
    """
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
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    np.random.seed(args.seed)

    # q1a)
    xTrainScaled, xTestScaled = normalize(xTrain, xTest)
    yPred = logit_unreg(xTrainScaled, yTrain, xTestScaled)
    yPred = yPred[:,1]
    fpr, tpr, _ = roc_curve(yTest, yPred)
    auc = round(roc_auc_score(yTest, yPred), 4)
    plt.plot(fpr, tpr, label="Normalized Logit = " + str(auc))
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")


    # q1b)
    xTrainPCA, xTestPCA = pca_fit(xTrainScaled, xTestScaled)

    # q1c)
    yPred = logit_pca(xTrainPCA, yTrain, xTestPCA)
    yPred = yPred[:, 1]
    fpr, tpr, _ = roc_curve(yTest, yPred)
    auc = round(roc_auc_score(yTest, yPred), 4)
    plt.plot(fpr, tpr, label="PCA Logit = " + str(auc))
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()