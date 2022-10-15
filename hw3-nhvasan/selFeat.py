import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler

# QUESTION
# 1. How do I call / incorporate Q1b in this file
def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # translate datetime to year, month, day, hour
    df['Year'] = pd.to_datetime(df['date']).dt.year
    df['Month'] = pd.to_datetime(df['date']).dt.month
    df['Day'] = pd.to_datetime(df['date']).dt.day
    df['Hour'] = pd.to_datetime(df['date']).dt.hour
    df = df.drop(columns=['date'])
    return df


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """

    df = df[['Month', 'lights', 'T2', 'T6', 'RH_2', 'RH_5', 'RH_6', 'RH_out', 'Press_mm_hg', 'Windspeed',
             'Visibility', 'Tdewpoint']]

    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data
    testDF : pandas dataframe
        Test data
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # Create and fit Standard Scaler object
    scaler = StandardScaler()
    stdScale = scaler.fit(trainDF)

    # Apply transform to train and test input data
    trainDF_scaled = stdScale.transform(trainDF)
    testDF_scaled = stdScale.transform(testDF)

    # Convert to DataFrame
    trainDF = pd.DataFrame(trainDF_scaled, columns=trainDF.columns)
    testDF = pd.DataFrame(testDF_scaled, columns=testDF.columns)

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)

    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)

    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)

    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)

    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
