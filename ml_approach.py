import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def preprocess_gesis(df: pd.DataFrame, columns: list=[], names: list=[], cutoff: int = 0):
    # drop all unfinished surveys
    df = df.drop(df[df["kp27_dispcode"] == 22].index).reset_index()

    # only get columns for feature space
    if columns:
        df = df[columns]

    # get rid of all values below cutoff (e.g. encoding for )
    df = df[df >= 0].reset_index(drop=True)

    # rename columns
    if len(columns) == len(names):
        df.columns = names
    else:
        raise ValueError("'columns' and 'names' need to have the same lenght.")

    # how often each answer was given
    count = df.value_counts()

    return df, count


def plot_scatter(count: pd.Series, normalize: bool=False):
    # only 3d plot
    x_min, x_max = count.index.levels[0].min(), count.index.levels[0].max()
    y_min, y_max = count.index.levels[1].min(), count.index.levels[1].max()

    X = np.arange(x_min, x_max+1)
    Y = np.arange(y_min, y_max+1)

    X, Y = np.meshgrid(X, Y)

    Z = count.unstack().values.T
    if normalize:
        Z = Z / Z.sum()

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(X, Y, Z)

    plt.show()


def do_regression(count: pd.Series, deg: int = 1, normalize: bool=False):
    # only 2d feature space
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