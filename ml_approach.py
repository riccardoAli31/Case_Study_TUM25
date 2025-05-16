import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def preprocess_gesis(df: pd.DataFrame, columns: list=[], names: dict={}, cutoff: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """preprocess gesis voter data, returning only columns specified in *columns* and change column names to *names*

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of gesis data
    columns : list, optional
        all other columns are dropped, by default []
    names : list, optional
        new names for columns, by default []
    cutoff : int, optional
        all rows in which values are below are dropped, by default 0

    Returns
    -------
    pd.DataFrame
        preprocessed dataframe
    pd.Series
        how often each unique answer was given
    """

    # drop all unfinished surveys
    df = df.drop(df[df["kp27_dispcode"] == 22].index).reset_index()

    # only get columns for feature space
    if columns:
        df = df[columns]

    # get rid of all values below cutoff (e.g. encoding for )
    df = df[df >= 0].reset_index(drop=True)

    # rename columns
    df.rename(names, inplace=True, axis=1)

    # how often each answer was given
    count = df.value_counts()

    return df, count


def plot_scatter(count: pd.Series, normalize: bool=False, labels: list=[]) -> None:
    """plots 3d scatter of 2d policie space and how often each answer was given

    Parameters
    ----------
    count : pd.Series
        how often each unique answer was given
    normalize : bool, optional
        if values should be normalized to [0, 1], by default False
    labels: list, optional
        [xlabel, ylabel]
    """
    x_min, x_max = count.index.levels[0].min(), count.index.levels[0].max()
    y_min, y_max = count.index.levels[1].min(), count.index.levels[1].max()

    X = np.arange(x_min, x_max+1)
    Y = np.arange(y_min, y_max+1)

    X, Y = np.meshgrid(X, Y)

    Z = count.unstack().values.T
    if normalize:
        Z = Z / Z.sum()

    ax = plt.figure().add_subplot()
    ax.scatter(X, Y, Z)

    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    plt.show()


def do_regression(count: pd.Series, deg: int = 1, normalize: bool=False) -> LinearRegression:
    """simple polynomial regression of degree *deg*

    Parameters
    ----------
    count : pd.Series
        how often each unique answer was given
    deg : int, optional
        degree of the polynomial, by default 1
    normalize : bool, optional
        if values should be normalized to [0,1] before fitting the model, by default False

    Returns
    -------
    LinearRegression
        fitted model
    """
    poly = PolynomialFeatures(degree=deg)

    x_min, x_max = count.index.levels[0].min(), count.index.levels[0].max()
    y_min, y_max = count.index.levels[1].min(), count.index.levels[1].max()

    X = np.arange(x_min, x_max+1)
    Y = np.arange(y_min, y_max+1)
    X, Y = np.meshgrid(X, Y)
    X, Y = X.flatten(), Y.flatten()

    Z = count.unstack().values.T.flatten()
    if normalize:
        Z = Z / Z.sum()

    input_pts = np.stack([X, Y]).T
    in_features = poly.fit_transform(input_pts)

    model = LinearRegression(fit_intercept=False)
    model.fit(in_features, Z)

    return model


def plot(model):
    pass