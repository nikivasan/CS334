from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.metrics import accuracy_score
import pandas as pd

def dt_train_test(xTrain, yTrain, xTest, yTest, criterion, maxDepth, minSamples):
    # Create Model
    dt = DecisionTreeClassifier(random_state=42, criterion=criterion, max_depth=maxDepth, min_samples_leaf=minSamples)
    dt.fit(xTrain,yTrain)

    # predict training data
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)

    # predict testing data
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    # parser.add_argument("md",
    #                     type=int,
    #                     help="maximum depth")
    # parser.add_argument("mls",
    #                     type=int,
    #                     help="minimum leaf samples")
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

    # MaxDepth: 6
    # MinLeafSamples: 5

    trainAcc1, testAcc1 = dt_train_test(xTrain, yTrain, xTest, yTest, "gini", 6, 5)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)

    # dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(xTrain, yTrain, xTest, yTest, "entropy", 6, 5)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

if __name__ == "__main__":
    main()