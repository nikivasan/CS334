import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    tree = dict()

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
        max_gini = 0
        split_attribute = None
        split_value = None

        for value in xFeat[attribute]:
            y_less = xFeat[xFeat[attribute] < value]['y']
            y_more = xFeat[xFeat[attribute] >= value]['y']
            left_counts = Counter(y_less) # count number of 0s and 1s in left split
            right_counts = Counter(y_more) # count number of 0s and 1s in right split

            lab0_l = left_counts.get(0, 0)
            lab1_l = left_counts.get(1, 0)
            lab0_r = right_counts.get(0,0)
            lab1_r = right_counts.get(1, 0)

            # calculate the gini score
            gini_left =  self.calc_gini(lab0_l, lab0_l)
            gini_right = self.calc_gini(lab0_l, lab1_r)

            # calculate weighted sum of left and right splits
            weight_left = (lab0_l + lab1_l) / ((lab0_l + lab1_l) + (lab0_r + lab1_r))
            weight_right = (lab0_r + lab1_r) / ((lab0_l + lab1_l) + (lab0_r + lab1_r))
            gini = (weight_left) * gini_left + (weight_right) * gini_right

            if gini > max_gini:
                split_attribute = attribute
                split_value = value
                max_gini = gini

        return [split_attribute, split_value, max_gini]

    def calc_entropy(self, labels):
        # if there are elements in the array, calculate the probability
        # else, prob is 0
        if len(labels) != 0:
            prob = len(np.extract(labels==0, labels))/ len(labels) # prob that label is 0 over all labels
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
        e = 10
        g = 10
        best_attribute = None
        best_value = None

        for col in xFeat.iloc[:, :-1]:
            if self.criterion == "entropy":
                best_entropy = self.best_entropy(xFeat, col)
                if best_entropy[2] <= e:
                    e = best_entropy[2]
                    best_attribute = best_entropy[0]
                    best_value = best_entropy[1]
            else:
                best_gini = self.best_gini(xFeat, col)
                if best_gini[2] <= g:
                    g = best_gini[2]
                    best_attribute = best_gini[0]
                    best_value = best_gini[1]

        return [best_attribute, best_value]

    def build_tree(self, xFeat, y, depth=0):
        # recompile dataframe
        train_df = xFeat
        train_df['y'] = y

        # get class label
        self.lab_counts = Counter(y)
        sorted_y = self.lab_counts.items().sort(key=lambda x: x[1])
        lab_counts_sort = list(sorted_y)
        if len(lab_counts_sort) > 0:
            lab = lab_counts_sort[-1][0]

        temp = train_df.copy()

        # change stopping condition
        # if y is greater than minimum leaf samples and depth is not greater than maxDepth
        if (len(y) >= self.minLeafSample) and (depth < self.maxDepth):
            # get best split criteria
            split_values = self.split_values(xFeat)
            best_attribute = split_values[0]
            best_value = split_values[1]

            if best_attribute is not None:
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
            else: # what do I do here
                return {
                    "Left": None,
                    "Right": None,
                    "Feature": None,
                    "Split Value": None,
                    "Depth": None,
                    "Label": None
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
        yHat = [] # variable to store the estimated class label
        for index, row in xFeat.iterrows():
            feat_vals = dict()
            for feat in xFeat:
                feat_vals.update({feat: row[feat]})
                dt = self.tree
                # Traverse tree
                while dt.get('Depth') < self.maxDepth and dt.get('Feature') in feat_vals:
                    best_feat = dt.get('Feature')
                    best_val = dt.get('Split Value')
                    # Base Case
                    if dt.get('Left') or dt.get('Right') is None:
                        return yHat.append(dt.get('Label'))
                    # Recursive Case
                    if feat_vals.get(best_feat) < best_val:
                        if dt.get('Left') is not None:
                            dt = dt.get('Left')
                    else:
                        if dt.get('Right') is not None:
                            dt = dt.get('Right')

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
    yHatTrain = dt.predict(xTrain)
    # print("yHatTrain", yHatTrain)
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

