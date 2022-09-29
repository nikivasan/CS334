import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    tree = dict()      # dictionary to store tree nodes

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
        self.lab_counts = None
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def calc_gini(self, num_zeros, num_ones):
        # if either split is empty, set count to 0
        if num_zeros is None:
            num_zeros = 0
        if num_ones is None:
            num_ones = 0
        if (num_zeros + num_ones) == 0: # if there are no rows left, gini is 0
            return 0.0
        else:
            prob_left = num_zeros / (num_ones + num_zeros)
            prob_right = num_ones / (num_ones + num_zeros)

            gini = 1 - (prob_left ** 2 + prob_right ** 2)

        return gini

    def best_gini(self, xFeat, attribute):
        max_gini = float("Inf")
        split_attribute = None
        split_value = None

        for value in xFeat[attribute]: # iterate through every column
            y_less = xFeat[xFeat[attribute] < value]['y'] # left split
            y_more = xFeat[xFeat[attribute] >= value]['y'] # right split
            left_counts = Counter(y_less) # count number of 0s and 1s in left split
            right_counts = Counter(y_more) # count number of 0s and 1s in right split

            # get counts of 0s and 1s per each split
            lab0_l = left_counts.get(0, 0)
            lab1_l = left_counts.get(1, 0)
            lab0_r = right_counts.get(0,0)
            lab1_r = right_counts.get(1, 0)

            # calculate the gini score of left and right split
            gini_left = self.calc_gini(lab0_l, lab0_l)
            gini_right = self.calc_gini(lab0_r, lab1_r)

            # calculate weighted sum of left and right splits
            weight_left = (lab0_l + lab1_l) / ((lab0_l + lab1_l) + (lab0_r + lab1_r))
            weight_right = (lab0_r + lab1_r) / ((lab0_l + lab1_l) + (lab0_r + lab1_r))
            gini = weight_left * gini_left + weight_right * gini_right

            if gini < max_gini:
                split_attribute = attribute
                split_value = value
                max_gini = gini

        return [split_attribute, split_value, max_gini]

    def calc_entropy(self, labels):
        # if there are elements in the array, calculate the probability
        # else, prob is 0
        if len(labels) != 0:
            prob = len(np.extract(labels==0, labels))/ len(labels)
        else:
            prob = 0
        # calculate entropy
        if prob != 0:
            entropy = np.sum(prob * np.log2(prob))
        else:
            entropy = 0
        return -entropy


    def best_entropy(self, xFeat, attribute):
        max_entropy = 1
        split_attribute = None
        split_value = None

        for value in xFeat[attribute]:
            # entropy of left split
            lab_left = xFeat[xFeat[attribute] <= value]['y']
            entropy_left = self.calc_entropy(lab_left)

            # entropy of right split
            lab_right = xFeat[xFeat[attribute] > value]['y']
            entropy_right = self.calc_entropy(lab_right)

            # weighted by sample size
            weight_left = len(lab_left) / (len(lab_left) + len(lab_right))
            weight_right = len(lab_right) / (len(lab_left) + len(lab_right))
            total_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)

            if total_entropy <= max_entropy:
                max_entropy = total_entropy
                split_attribute = attribute
                split_value = value

        return [split_attribute, split_value, max_entropy]

    def split_values(self, xFeat):
        # find best split
        e = float("Inf")
        g = float("Inf")
        best_attribute = None
        best_value = None

        for col in xFeat.iloc[:, :-1]:
            # for each column in xFeat, find the best entropy or gini split values/attributes
            if self.criterion == "entropy":
                best_entropy = self.best_entropy(xFeat, col)
                if best_entropy[2] <= e:
                    e = best_entropy[2]
                    best_attribute = best_entropy[0]
                    best_value = best_entropy[1]
            else:
                best_gini = self.best_gini(xFeat, col)
                if best_gini[2] < g:
                    g = best_gini[2]
                    best_attribute = best_gini[0]
                    best_value = best_gini[1]

        return [best_attribute, best_value]

    def build_tree(self, xFeat, y, depth=0):
        # recompile dataframe
        train_df = xFeat
        train_df['y'] = y
        best_attribute = None
        best_value = None
        split_values = None
        lab = None

        # get class label
        self.lab_counts = Counter(y)
        lab_counts_sort = list(sorted(self.lab_counts.items(), key=lambda item: item[1]))

        # get label of majority class
        if len(lab_counts_sort) > 0:
            lab = lab_counts_sort[-1][0]

        # create temp dataframe
        temp = train_df.copy()

        # if y is greater than minimum leaf samples and depth is not greater than maxDepth
        if (len(y) >= self.minLeafSample) and (depth < self.maxDepth):
            # get best split criteria
            split_values = self.split_values(xFeat)
            best_attribute = split_values[0]
            best_value = split_values[1]

            # partition the data into left and right split
            left_df = temp[temp[best_attribute] <= best_value].copy()
            right_df = temp[temp[best_attribute] > best_value].copy()

            # create new input sets
            xFeatL = left_df.iloc[:, :-1]
            xFeatR = right_df.iloc[:, :-1]

            # build tree recursively using a dictionary
            return {
                "Left": self.build_tree(xFeatL, left_df['y'], depth + 1),
                "Right": self.build_tree(xFeatR, right_df['y'], depth + 1),
                "Feature": best_attribute,
                "Split Value": best_value,
                "Depth": depth,
                "Label": lab
            }
        else:
            return {
                "Left": xFeat,
                "Right": xFeat,
                "Feature": None,
                "Split Value": None,
                "Depth": depth,
                "Label": lab
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
        self.tree = self.build_tree(xFeat, y)
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
        yHat = [] # result array to hold labels
        for idx, x in xFeat.iterrows(): # iterate through each row in X
            vals = dict()
            for feature in xFeat:       # for each column, store col and element in a dictionary
                vals.update({feature: x[feature]})
            yHat.append(self.predict_observation(vals))

        return yHat

    def predict_observation(self, vals):
        dt = self.tree
        # Traverse tree
        while dt.get('Depth') < self.maxDepth and dt.get('Feature') in vals:
            best_feat = dt.get('Feature')
            best_val = dt.get('Split Value')

            if dt.get('Feature') is None:
                return dt.get('Label')

            # Recursive Case
            if vals.get(best_feat) < best_val:
                if dt.get('Left') is not None:
                    dt = dt.get('Left')
            else:
                if dt.get('Right') is not None:
                    dt = dt.get('Right')

            # Else return observation
        return dt.get('Label')


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
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc

def plot(xTrain, yTrain, xTest, yTest):
    # Fixing Min Leaf Sample: 5
    # Fixing Max Depth: 5

    df = pd.DataFrame()
    train_accs = []
    test_accs = []
    max_depth_vals = []
    min_leaf_vals = []

    for i in range(1,21):
        # create Knn object
        max_depth_vals.append(i)
        dt = DecisionTree(minLeafSample=5, criterion="entropy", maxDepth=i)
        # dt = DecisionTree(minLeafSample=i, criterion="entropy", maxDepth=5)

        # train model
        dt.train(xTrain, yTrain['label'])

        # predict the training accuracy
        yHatTrain = dt.predict(xTrain)
        trainAcc = accuracy_score(yTrain['label'], yHatTrain)
        train_accs.append(trainAcc)

        # predict the test accuracy
        yHatTest = dt.predict(xTest)
        testAcc = accuracy_score(yTest['label'], yHatTest)
        test_accs.append(testAcc)

    # add accuracies to dataframe
    df['Max Depth Values'] = max_depth_vals
    # df['Min Leaf Values'] = min_leaf_vals
    df['train_acc'] = train_accs
    df['test_acc'] = test_accs

    # plot
    # plt.plot(df['Min Leaf Values'], df['train_acc'], marker='.', color='r', label='train')
    # plt.plot(df['Min Leaf Values'], df['test_acc'], marker='+', color='b', label='test')
    plt.plot(df['Max Depth Values'], df['train_acc'], marker='.', color='r', label='train')
    plt.plot(df['Max Depth Values'], df['test_acc'], marker='+', color='b', label='test')
    plt.legend();
    # plt.title("Accuracy vs Max Depth Values")
    plt.title("Accuracy vs Min Leaf Values")
    # plt.xlabel("Min Leaf Values")
    plt.xlabel("Max Depth Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

def main():

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

"""
def main():
    # Adjusted Main Method for Plots

    # removes argparse for k
    parser = argparse.ArgumentParser()

    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()

    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    plot(xTrain, yTrain, xTest, yTest)
"""

if __name__ == "__main__":
    main()

