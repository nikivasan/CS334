import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

pd.options.mode.chained_assignment = None  # default='warn'

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

def calcPearson(df):
    """
    Calculate Pearson Correlation Matrix

    Save matrix

    Parameters
    ----------
    xFeat : Input Features
    y : Predicted Variable

    Returns
    -------
    Matrix : pearson correlation matrix
    """
    # plot correlation matrix
    matrix = df.corr()

    plt.figure(figsize=(13, 13))
    sns.heatmap(matrix, annot=True, annot_kws={"fontsize": 3}, cmap="PiYG")
    plt.title("Correlation Matrix of Energy Data")
    plt.show()
    plt.savefig('corr_matrix_heatmap.png')

    # find highly correlated features
    upper_corr_mat = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    unstacked = upper_corr_mat.abs().unstack().dropna()
    sorted_mat = unstacked.sort_values(kind="quicksort", ascending=False)
    high_corr = sorted_mat[sorted_mat > 0.8]
    df_high_corr = pd.DataFrame(high_corr)
    print(df_high_corr)

    return df_high_corr




def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="eng_xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="eng_yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="eng_xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="eng_yTest.csv",
                        help="filename for labels associated with the test data")

    # load the train and test data
    args = parser.parse_args()
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # add y var to training data
    dfTrain = xTrain
    dfTrain['y'] = yTrain
    cols = list(dfTrain.columns)
    cols.reverse()
    dfTrain = dfTrain[cols]

    # add y var to testing data
    # dfTest = xTest
    # dfTest['y'] = yTest
    # cols = list(dfTest.columns)
    # cols.reverse()
    # dfTest = dfTest[cols]

    # extract the new features
    dfNewTrain = extract_features(dfTrain)
    # dfNewTest = extract_features(dfTest)

    # create matrix
    matrixTrain = calcPearson(dfNewTrain)
    # matrixTest = calcPearson(dfNewTest)

if __name__ == "__main__":
    main()