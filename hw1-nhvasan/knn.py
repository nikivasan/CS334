import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Knn(object):
    k = 0    # number of neighbors to use
    x_train = None # store dataframe
    y_train = None  # store attributes
    def __init__(self, k):
        """
        Knn constructor
        Parameters
        ----------
        k : int
            Number of neighbors to use.
        """
        self.k = k


    def train(self, xFeat, y):
        """
        Train the k-nn model.
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

        # Store data (x) + labels (y)
        self.x_train = xFeat
        self.y_train = y
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

        # APPROACH 1
        # loop through each row in input
        # for i in range(len(xFeat)):
            # array to store distances between new data point and rest of records
            # distances = []

            # loop through training data
            # for m in range(len(self.x_train)):
                # calculate euclidean distance between ith row of xFeat and kth row of x_train
                # dist = self.euclidean_dist(np.array(self.x_train.iloc[m]), xFeat.iloc[i])
                # distances.append(dist)

            # sort array in ascending order
            # distances = np.array(distances)
            # dist_sorted = np.argsort(distances) # preserve index in sort

            # keep first k rows
            # dist_sorted = dist_sorted[:self.k]

            # labels of above k rows
            # labels = self.y_train.iloc[dist_sorted]

            # use majority class
            # mode_label = st.mode(labels,axis=None)[0]
            # yHat.append(mode_label)

        # APPROACH 2
        # iterate through each row in xTest (input)
        for i in range(len(xFeat)):
            # calculate euclidean distance between new observation i and training data
            distances = np.sqrt(np.sum((self.x_train - xFeat.iloc[i])**2, axis=1))

            # find indices of k nearest neighbors (smallest distance)
            ind = np.argsort(distances, axis=0)[:self.k]

            # find majority class
            y_hat = self.y_train[ind].mode()[0]

            # add predicted class to outcome array
            yHat.append(y_hat)

        # convert to numpy array
        yHat = np.array(yHat)

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    correct = 0
    # calculate number of correct labels
    for i in range(len(yTrue)):
        if yTrue[i] == yHat[i]:
            correct += 1

    # calculate number of correct over all labels * 100
    acc = (correct / float(len(yTrue))) * 100

    return acc

def plot(xtrain, ytrain, xtest, ytest):
    df = pd.DataFrame()
    train_accs = []
    test_accs = []
    k_values = []

    for k in range(1, 25): # change to 10 or N
        # create Knn object
        k_values.append(k)
        knn = Knn(k)

        # train model
        knn.train(xtrain, ytrain.iloc[:,0])

        # predict the training accuracy
        yHatTrain = knn.predict(xtrain)
        trainAcc = accuracy(yHatTrain, ytrain.iloc[:,0])
        train_accs.append(trainAcc)

        # predict the test accuracy
        yHatTest = knn.predict(xtest)
        testAcc = accuracy(yHatTest, ytest.iloc[:,0])
        test_accs.append(testAcc)

    # add accuracies to dataframe
    df['k'] = k_values
    df['train_acc'] = train_accs
    df['test_acc'] = test_accs

    print(df)

    # Plot Train Accuracy
    plt.plot(df['k'],df['train_acc'], color='red')
    plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Training Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    # Plot Test Accuracy
    plt.plot(df['k'],df['test_acc'], color='blue')
    plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Test Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    # Plot both
    plt.plot(df['k'], df['train_acc'], marker='.', color='r', label='train')
    plt.plot(df['k'], df['test_acc'], marker='+', color='b', label='test')
    plt.legend();
    # plt.xticks(np.arange(min(df['k']), max(df['k']) + 1, 1.0))
    plt.title("Accuracy vs K Values")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy (%)")
    plt.show()

    return df

def main():
    # Main file to run from the command line.
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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

    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])

    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])

    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])

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

