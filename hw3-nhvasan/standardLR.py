import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy

class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        start = time.time() # initialize timer
        trainStats = {}

        # Training Data
        arr_ones_train = np.ones((len(xTrain),1))
        X_train = np.concatenate((arr_ones_train, xTrain), axis=1)
        self.beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)),yTrain)
        train_mse = self.mse(X_train, yTrain) # train mse

        # Testing Data
        arr_ones_test = np.ones((len(xTest), 1))
        X_test = np.concatenate((arr_ones_test, xTest), axis=1)
        self.beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_test), X_test)), np.transpose(X_test)), yTest)
        test_mse = self.mse(X_test, yTest)

        end = time.time()
        time_elapse = end - start

        trainStats['0'] = {
            'time': time_elapse,
            'train-mse': train_mse,
            'test-mse': test_mse
        }

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
