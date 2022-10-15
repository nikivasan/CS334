import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from lr import LinearRegression, file_to_numpy
import standardLR


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000  # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}  # initialize result dictionary

        arr_ones_train = np.ones((len(xTrain), 1))
        X_train = np.concatenate((arr_ones_train, xTrain), axis=1)  # create xTrain matrix
        arr_ones_test = np.ones((len(xTest), 1))
        X_test = np.concatenate((arr_ones_test, xTest), axis=1)  # create xTest matrix

        self.beta = np.ones((13,1))  # initialize betas to any value (1s)

        # for Q3b -- sample 40% of training data
        # num_rows = int(len(X_train) * 0.4)
        # X_train = X_train[np.random.choice(X_train.shape[0], num_rows, replace=False), :]
        # yTrain = yTrain[np.random.choice(yTrain.shape[0], num_rows, replace=False), :]

        iteration = 0
        epoch_train_mse = [] # array to store epoch train mse (Q3b,c)
        epoch_test_mse = [] # array to store epoch test mse (Q3b,c)
        times = []
        for ep in range(1, self.mEpoch+1):
            indices = np.random.permutation(len(X_train)) # shuffle data
            X_train = X_train[indices]
            Y_train = yTrain[indices]

            x_batches = np.array_split(X_train, len(X_train)/self.bs) # split into x batches
            y_batches = np.array_split(Y_train, len(Y_train)/self.bs) # split into y batches

            start = time.time()
            for batch in range(len(x_batches)):
                x = x_batches[batch]
                y = y_batches[batch]
                batch_grads = np.zeros((13, 1))
                for sample in range(len(x)):
                    gradient = np.multiply(np.transpose(x[sample]), y[sample] - np.matmul(x[sample], self.beta))
                    gradient = gradient.reshape(-1,1)
                    batch_grads = np.add(gradient, batch_grads)
                batch_avg = np.divide(batch_grads,self.bs)
                self.beta = self.beta + self.lr * batch_avg

                train_mse = self.mse(X_train, yTrain)
                test_mse = self.mse(X_test, yTest)
                timeElapsed = time.time() - start
                trainStats[iteration] = {
                    'time':timeElapsed,
                    'train-mse': train_mse,
                    'test-mse': test_mse
                }
                start = time.time()
                iteration += 1

            # For Q3 and Q4
            # epoch_train_mse.append(train_mse)
            # epoch_test_mse.append(test_mse)
            # times.append(timeElapsed)
        # For Q3 and Q4
        # overall_train_mse = epoch_train_mse[-1]
        # overall_test_mse = epoch_test_mse[-1]
        # overall_batch_time = times[-1]
        # return overall_batch_time, overall_train_mse, overall_test_mse
        # return trainStats, epoch_train_mse, epoch_test_mse, times

        return trainStats

def plot_3b(xTrain, yTrain, xTest, yTest):
    rates = [0.00001, 0.0001, 0.001, 0.01, 0.05]
    train_mses = []

    for lr in rates:
        model = SgdLR(lr, 1, 100)
        trainStats, epoch_train_mse, epoch_test_mse, times = model.train_predict(xTrain, yTrain, xTest, yTest)
        train_mses.append(epoch_train_mse)

    df = pd.DataFrame()
    for i in range(len(train_mses)):
        df[i] = train_mses[i]

    plt.plot(df)
    plt.legend([0.00001, 0.0001, 0.001, 0.01, 0.05])
    plt.title("Train MSE vs Epochs with Batch Size of One")
    plt.xlabel("Epochs")
    plt.ylabel("Train MSE")
    plt.ylim(0.2, 1.3)
    plt.show()

def plot_3c(xTrain, yTrain, xTest, yTest):
    model = SgdLR(0.01, 1, 100)
    trainStats, epoch_train_mse, epoch_test_mse, times = model.train_predict(xTrain, yTrain, xTest, yTest)

    df = pd.DataFrame(columns=['epoch', 'train-mse', 'test-mse'])
    df['train-mse'] = epoch_train_mse
    df['test-mse'] = epoch_test_mse

    plt.plot(df)
    plt.legend(['train-mse', 'test-mse'])
    plt.title("Train and Test MSE vs Epochs with Batch Size of One")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()

def plot_4a(xTrain, yTrain, xTest, yTest):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 16770]
    batch_times = []
    train_mses = []
    test_mses = []
    for size in batch_sizes:
        model = SgdLR(0.01, size, 15)
        overall_batch_time, overall_train_mse, overall_test_mse = model.train_predict(xTrain, yTrain, xTest, yTest)
        batch_times.append(overall_batch_time)
        train_mses.append(overall_train_mse)
        test_mses.append(overall_test_mse)

    df_train = pd.DataFrame()
    df_train['Batch Times'] = batch_times
    df_train['train-mse'] = train_mses
    df_train = df_train.iloc[:-1, :]

    df_test = pd.DataFrame()
    df_test['Batch Times'] = batch_times
    df_test['test-mse'] = test_mses
    df_test = df_test.iloc[:-1, :]

    lr = standardLR.StandardLR()
    lr_trainStats = lr.train_predict(xTrain, yTrain, xTest, yTest)
    lr_time = lr_trainStats['0']['time']
    lr_train_mse = lr_trainStats['0']['train-mse']
    lr_test_mse = lr_trainStats['0']['test-mse']

    # Train MSE Plot
    plt.scatter(df_train['Batch Times'], df_train['train-mse'], label='sgd mse', color='blue')
    plt.scatter([lr_time], [lr_train_mse], color="red", label='closed form mse')
    plt.legend()
    plt.title("Train MSE vs Time for 15 Epochs")
    plt.xlabel("Batch Time")
    plt.ylabel("Train MSE")
    # plt.xlim(0.01)
    # plt.ylim(0.75)
    plt.show()

    # Test MSE Plot
    plt.scatter(df_test['Batch Times'], df_test['test-mse'], label='sgd mse', color='blue')
    plt.scatter([lr_time], [lr_test_mse], color="red", label='closed form mse')
    plt.legend()
    plt.title("Test MSE vs Time for 15 Epochs")
    plt.xlabel("Batch Time")
    plt.ylabel("Test MSE")
    plt.show()


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")


    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)

    # plot_3b(xTrain, yTrain, xTest, yTest)
    # plot_3c(xTrain,yTrain,xTest,yTest)
    # plot_4a(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()

