import argparse
import numpy as np
import pandas as pd
import pdb
import math
from sklearn.metrics import accuracy_score


class DecisionTree(object):
    maxDepth = 0  # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None  # splitting criterion
    dt_dict = dict()

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int
            Maximum depth of the decision tree
        minLeafSample : int
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def entropy(self, attribute):
        # initialize result variable
        entropy = 0

        # Find and store class probabilities
        count = np.bincount(attribute)
        probs = count / len(attribute)

        # for each probability, multiply by base 2 of prob
        for prob in probs:
            if prob == 1:
                pass
            else:
                entropy += prob * math.log(prob, 2)

        # multiply by -1
        entropy = -entropy
        return entropy

    def information_gain(self, left, right):
        left_entropy = self.entropy(left)
        right_entropy = self.entropy(right)

        total_entropy = left_entropy + right_entropy

        total_size = len(left) + len(right)

        info_gain = (len(left) / total_size * left_entropy) + (len(right) / total_size * right_entropy)

        return info_gain


    def gini(self, label):
        gini = 0
        zeros = 0
        ones = 0
        for y in label:
            if y == 0:
                zeros += 1
            else:
                ones += 1
        if len(label) == 0:
            prob = 1
        else:
            zeros /= len(label)
            ones /= len(label)

            prob = zeros * ones

        return 1 - prob

    def split(self, xFeat, y):
        print("!")
        n_rows, n_cols = xFeat.shape
        info_score = 0
        # For each feature in input
        best_split = {}
        best_info_score = -float("inf")

        combined = xFeat
        combined['y'] = y

        for col in range(1, n_cols):
            sorted_feats = combined.sort_values(by=xFeat.columns[col])
            for row in range(1, n_rows):
                df_left = sorted_feats.iloc[:][:row]
                df_right = (sorted_feats.iloc[:][row+1:])

                y_left = combined['y'].iloc[:row]
                y_right = (combined['y'].iloc[row + 1:])

                # Calculate the information criteria and save the split parameters
                # if the current split if better than the previous best

                if self.criterion == "gini1":
                    info_score = self.information_gain(y_left, y_right)
                else:
                    info_score = self.information_gain(y_left, y_right)

                    #info_score = self.entropy(y_left) + self.entropy(y_right)
                if info_score > best_info_score:
                    print(row, col)
                    best_split = {
                        'feature_index': col,
                        'split_value': row,
                        'df_left': df_left,
                        'df_right': df_right,
                        'gain': info_score
                    }
                    best_info_score = info_score

        return best_split

    def decision_tree(self, xFeat, y, depth=0):
        # recompose training data
        train_df = xFeat
        train_df['y'] = y

        is_leaf = False # flag to determine if node is a leaf

        # if maximum depth is met, return is_leaf and majority class of y
        if depth == self.maxDepth:
            is_leaf = True # set flag to true
            common = 0
            mode = 0
            for label in y:
                if label == 1:
                    common += 1
                else:
                    common -= 1

            if common > 0:
                mode = 1
            else:
                mode = 0

            return {
                'Is Leaf': is_leaf,
                'label': mode
            }
        # find best split
        print(xFeat.shape, y.shape)
        best_split = self.split(xFeat, y)

        col = best_split['feature_index']
        val = best_split['split_value']

        # partition data into two splits
        left = train_df[train_df.iloc[:, col] <= val]
        right = train_df[train_df.iloc[:, col] > val]

        # decompose df
        yL = left['y'] # store labels for left split
        xFeatL = left.iloc[:, :-1] # store features for left split
        yR = right['y'] # store labels for right split
        xFeatR = right.iloc[:, :-1] # store features for right split

        # recursive call
        return {
            "left": self.decision_tree(xFeatL, yL, depth+1),
            "right": self.decision_tree(xFeatR, yR, depth+1),
            "Feature": col,
            "Split Value": val,
            "Is Leaf": is_leaf
        }

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.dt_dict = self.decision_tree(xFeat,y)
        print(self.dt_dict)
        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = []  # variable to store the estimated class label
        for row in xFeat:
            while not self.dt_dict['Is Leaf']:
                if self.dt_dict['Is Leaf']:
                    yHat.append(self.dt_dict['label'])
                    break
                else:
                    feat = self.dt_dict['Feature']
                    val = self.dt_dict['Split Value']
                    if xFeat.iloc[row][feat] <= val:
                        self.dt_dict = self.dt_dict['left']
                    else:
                        self.dt_dict = self.dt_dict['right']
        return yHat

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    breakpoint()
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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

    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
