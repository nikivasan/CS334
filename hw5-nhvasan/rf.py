import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    forest = []        # all trees in forest

    class Tree(object):
        oob_samples = []
        features = []
        train_samples = []
        model = DecisionTreeClassifier()

        def __init__(self, model):
            self.model = model


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat

    def create_forest(self):
        for _ in range(self.nest):
            self.forest.append(self.Tree(DecisionTreeClassifier(criterion=self.criterion,
                                                                max_depth=self.maxDepth,
                                                                min_samples_leaf=self.minLeafSample)))

    def bootstrap_indices(self, xFeat, y):
        row_indices = np.random.choice(len(xFeat), size=len(xFeat), replace=True)
        oob_indices = np.array([i for i in range(len(xFeat)) if i not in row_indices])
        col_indices = np.random.choice(len(xFeat[0]), size=self.maxFeat, replace=False)
        return row_indices, oob_indices, col_indices

    def calc_oob(self, xFeat, y):
        total_predictions = []
        for idx1, tree in enumerate(self.forest):
            X_oob = xFeat[tree.oob_samples[:,None], tree.features]
            curr_tree_preds = tree.model.predict(X_oob) # testing
            predictions = []
            i = 0
            pred_ix = 0
            while i < len(xFeat):
                if i in tree.oob_samples:
                    predictions.append(curr_tree_preds[pred_ix])
                    pred_ix += 1
                else:
                    predictions.append(-1)
                i += 1
            total_predictions.append(predictions)
        total_predictions = np.array(total_predictions)
        total_predictions = np.transpose(total_predictions)

        yPreds = []
        for row in range(len(total_predictions)):
            num_zeros = 0
            num_ones = 0
            for col in range(len(total_predictions[0])):
                if total_predictions[row][col] == 0:
                    num_zeros += 1
                elif total_predictions[row][col] == 1:
                    num_ones += 1
            if num_zeros < num_ones:
                yPreds.append(1)
            else:
                yPreds.append(0)

        misclass_error = 1 - accuracy_score(y, yPreds)
        return misclass_error


    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """

        # find train, oob samples + fit all trees in forest
        self.create_forest()
        for tree in self.forest:
            # add forest
            row_indices, oob_indices, col_indices = self.bootstrap_indices(xFeat, y)
            tree.train_samples = row_indices
            tree.features = col_indices
            tree.oob_samples = oob_indices
            X = xFeat[row_indices[:, None], col_indices]
            y = y[row_indices, :]
            tree.model.fit(X, y)

        stats = self.calc_oob(xFeat, y)

        return stats

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
            Predicted response per sample
        """
        yHat = []
        predictions = []
        for tree in self.forest:
            xFeat_sub = xFeat[:, tree.features]
            predictions.append(tree.model.predict(xFeat_sub))

        predictions = np.array(predictions)
        predictions = np.transpose(predictions)
        for row in range(len(predictions)):
            num_zeros = 0
            num_ones = 0
            for col in range(len(predictions[0])):
                if predictions[row][col] == 0:
                    num_zeros += 1
                elif predictions[row][col] == 1:
                    num_ones += 1
            if num_zeros <= num_ones:
                yHat.append(1)
            else:
                yHat.append(0)

        return yHat

def get_params(xTrain, yTrain, xTest, yTest):
    nest = [2, 4, 6, 8, 10]
    num_feats = [3, 4, 5, 6, 9]
    criterions = ["gini", "entropy"]
    max_depths = [1, 5, 10, 20, 25]
    min_ls = [1, 5, 10, 25, 50]

    cols = ['nest', 'maxFeat', 'criterion', 'maxDepth', 'minLeafSample', 'test_misclass_error']
    stats = pd.DataFrame(columns=cols)
    for n in nest:
        for f in num_feats:
            for c in criterions:
                for d in max_depths:
                    for ls in min_ls:
                        model = RandomForest(nest=n, maxFeat=f, criterion=c, maxDepth=d, minLeafSample=ls)
                        trainStats = model.train(xTrain, yTrain)
                        preds = model.predict(xTest)
                        error = round(1 - accuracy_score(yTest, preds),5)
                        new_row = pd.Series([n, f, c, d, ls, error], index=cols)
                        stats = stats.append(new_row, ignore_index=True)

    df = stats[stats['test_misclass_error'] == stats['test_misclass_error'].min()]
    return df.to_string()

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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
    parser.add_argument("--epoch", default=5, type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)

    # 2a) model parameters for training
    # (optimal params from 2c)
    nest = 6
    maxFeat = 4
    criterion = "gini"
    maxDepth = 10
    minLeafSample = 1

    model = RandomForest(nest, maxFeat, criterion, maxDepth, minLeafSample)
    trainStats = model.train(xTrain, yTrain)
    print("Average OOB Misclassification Error:", trainStats)
    yHat = model.predict(xTest)
    test_error = 1 - accuracy_score(yTest, yHat)
    print("Test Misclassification Error:", test_error)

    # 2b) hyperparameter tuning
    # stats = get_params(xTrain, yTrain, xTest, yTest)
    # print(stats)


if __name__ == "__main__":
    main()