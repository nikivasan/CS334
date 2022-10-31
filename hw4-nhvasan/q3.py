import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")


def num_mistakes(yTrue, yPred):
    num = 0
    for i in range(len(yPred)):
        if yPred[i] != yTrue[i]:
            num += 1
    return num

def naivebayes(xTrain, xTest, yTrain, yTest):
    nb = MultinomialNB()
    nb.fit(xTrain, yTrain.to_numpy().ravel())
    yPred = nb.predict(xTest)
    mistakes = num_mistakes(yTest.to_numpy(), yPred)
    return mistakes

def logit(xTrain, xTest, yTrain, yTest):
    lg = LogisticRegression(max_iter=1000, C=1)
    lg.fit(xTrain, yTrain.to_numpy().ravel())
    yPred = lg.predict(xTest)
    mistakes = num_mistakes(yTest.to_numpy(), yPred)
    return mistakes

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrainBin",
                        default="xTrainBin.csv",
                        help="filename of the binary training dataset")
    parser.add_argument("--xTestBin",
                        default="xTestBin.csv",
                        help="filename of the binary testing dataset")
    parser.add_argument("--xTrainCount",
                        default="xTrainCount.csv",
                        help="filename of the count training dataset")
    parser.add_argument("--xTestCount",
                        default="xTestCount.csv",
                        help="filename of the count testing data")
    parser.add_argument("--yTrain",
                        default="yTrain.csv",
                        help="filename of the training labels")
    parser.add_argument("--yTest",
                        default="yTest.csv",
                        help="filename of the testing labels")
    args = parser.parse_args()

    # read in files
    xTrainBin = pd.read_csv(args.xTrainBin)
    xTestBin = pd.read_csv(args.xTestBin)
    xTrainCount = pd.read_csv(args.xTrainCount)
    xTestCount = pd.read_csv(args.xTestCount)
    yTrain = pd.read_csv(args.yTrain)
    yTest = pd.read_csv(args.yTest)

    # logit models
    binary_logit = logit(xTrainBin, xTestBin, yTrain, yTest)
    count_logit = logit(xTrainCount, xTestCount, yTrain, yTest)
    print("Logit Number of Mistakes [Binary Data]", binary_logit)
    print("Logit Number of Mistakes [Count Data]", count_logit)

    # naive bayes models
    binary_nb = naivebayes(xTrainBin, xTestBin, yTrain, yTest)
    count_nb = naivebayes(xTrainCount, xTestCount, yTrain, yTest)
    print("Naive Bayes Number of Mistakes [Binary Data]", binary_nb)
    print("Naive Bayes Number of Mistakes [Count Data]", count_nb)

if __name__ == "__main__":
    main()