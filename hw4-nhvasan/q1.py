import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    df = pd.read_csv(filename, header=None)
    df = df.rename(columns={0:"col"})
    df['Text'] = df['col'].str[1:]
    df['Label'] = df['col'].str[0]
    df = df.drop(columns=['col'])

    xTrain, xTest, yTrain, yTest = train_test_split(df['Text'], df['Label'], train_size=0.7, random_state=42)
    # yTrain = np.array(yTrain).reshape(-1,1)
    # yTest = np.array(yTest).reshape(-1, 1)

    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)
    yTrain = pd.DataFrame(yTrain)
    yTest = pd.DataFrame(yTest)

    yTrain.to_csv("yTrain.csv", index=False, header=False)
    yTest.to_csv("yTest.csv", index=False, header=False)

    return xTrain, xTest, yTrain, yTest


def build_vocab_map(xTrain):
    cv = CountVectorizer(min_df=30) # initalize CountVectorizer object
    word_freq = cv.fit_transform(xTrain['Text']).sum(axis=0) # train CV on data and get word frequency
    vocab_map = dict()
    for word, index in cv.vocabulary_.items(): # store each word/frequency pair in dictionary
        vocab_map[word] = word_freq[0,index]
    return vocab_map


def construct_binary(xTrain, xTest, vocab_map):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    xTrainBin = []
    xTestBin = []
    for i in range(len(xTrain)): # for each row in dataframe
        email = xTrain['Text'].iloc[i] # select each email
        email_split = email.split() # split the email on whitespace
        arr = []
        for word in vocab_map: # for each word in dict
            if word in email_split: # if word occurs in arr of email text
                arr.append(1) # append 1
            else:
                arr.append(0) # else, append 0
        xTrainBin.append(arr)

    for j in range(len(xTest)):
        email = xTest['Text'].iloc[j]
        email_split = email.split()
        arr = []
        for word in vocab_map:
            if word in email_split:
                arr.append(1)
            else:
                arr.append(0)
        xTestBin.append(arr)

    arr_ones_train = np.ones((len(xTrainBin), 1))
    arr_ones_test = np.ones((len(xTestBin), 1))

    xTrainBin = np.concatenate((arr_ones_train, xTrainBin), axis=1)
    xTestBin = np.concatenate((arr_ones_test, xTestBin), axis=1)

    xTrainBin = pd.DataFrame(xTrainBin)
    xTestBin = pd.DataFrame(xTestBin)

    xTrainBin.to_csv("xTrainBin.csv", index=False, header=False)
    xTestBin.to_csv("xTestBin.csv", index=False, header=False)

    return xTrainBin, xTestBin


def construct_count(xTrain, xTest, vocab_map):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    xTrainCount = []
    xTestCount = []
    for i in range(len(xTrain)): # for each row in dataframe
        email = xTrain['Text'].iloc[i] # select each email
        email_split = email.split() # split the email on whitespace
        arr = []
        for word in vocab_map: # for each word in dict
            arr.append(email_split.count(word)) # append count
        xTrainCount.append(arr)

    for j in range(len(xTest)): # for each row in dataframe
        email = xTest['Text'].iloc[j] # select each email
        email_split = email.split() # split the email on whitespace
        arr = []
        for word in vocab_map: # for each word in dict
            arr.append(email_split.count(word)) # append count
        xTestCount.append(arr)

    arr_ones_train = np.ones((len(xTrainCount), 1))
    arr_ones_test = np.ones((len(xTestCount), 1))

    xTrainCount = np.concatenate((arr_ones_train, xTrainCount), axis=1)
    xTestCount = np.concatenate((arr_ones_test, xTestCount), axis=1)

    xTrainCount = pd.DataFrame(xTrainCount)
    xTestCount = pd.DataFrame(xTestCount)

    xTrainCount.to_csv("xTrainCount.csv", index=False, header=False)
    xTestCount.to_csv("xTestCount.csv", index=False, header=False)

    return xTrainCount, xTestCount


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    xTrain, xTest, yTrain, yTest = model_assessment(args.data)
    vocab_map = build_vocab_map(xTrain)
    print(vocab_map)
    construct_binary(xTrain, xTest, vocab_map)
    construct_count(xTrain, xTest, vocab_map)


if __name__ == "__main__":
    main()
