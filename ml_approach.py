import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import data_preprocessing.data_preprocess as dp


party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var='Planned Economy', y_var='Environmental Protection')

party_pca, voter_pca = dp.center_rotate_data_cloud(party_scaled, voter_scaled, x_var='Planned Economy', y_var='Environmental Protection')


def fit_multinomial_logit(voter_pca: pd.DataFrame, party_pca: pd.DataFrame, pc_cols: tuple[str,str] = ('PC1','PC2')) -> tuple[np.ndarray, pd.DataFrame, float]:
    """
    Fit U_ij = lambda_j - beta * ||y_i - z_j||^2 by treating each (i,j) pair
    as a sample and labeling it with class j. Returns:
      lambda_val : array, shape (p,) intercepts = hat{lambda}_j
      lambda_df  : DataFrame with ['class_index','party_name','lambda_hat']
      beta_val   : float, distance‐sensitivity hat{beta}
    """
    # 1) extract coords
    Y = voter_pca[list(pc_cols)].to_numpy()   # n×2
    Z = party_pca[list(pc_cols)].to_numpy()   # p×2
    n, p = Y.shape[0], Z.shape[0]

    # 2) build the “long” feature = -||y_i - z_j||^2
    dist2  = ((Y[:,None,:] - Z[None,:,:])**2).sum(axis=2)  # (n,p)
    X_long = -dist2.reshape(n*p, 1)                        # (n*p,1)

    # 3) label each row by its party‐index j
    y_long = np.tile(np.arange(p), n)                     # length n*p

    # 4) fit multinomial logit with intercepts only
    clf = LogisticRegression(
        penalty=None,
        solver='lbfgs',
        multi_class='multinomial',
        fit_intercept=True
    )
    clf.fit(X_long, y_long)

    # 5) extract parameters
    lambda_val = clf.intercept_.copy()          # length‐p array
    beta_val   = -clf.coef_.mean()              # scalar

    # 6) build the lambda_df
    classes    = clf.classes_                   # should be [0..p-1]
    party_names = party_pca['Party_Name'].reset_index(drop=True)
    lambda_df = pd.DataFrame({
        'class_index': classes,
        'Party_Name' : party_names.loc[classes].values,
        'Valence' : lambda_val
    }).sort_values('Valence', ascending=False).reset_index(drop=True)

    return lambda_val, lambda_df, beta_val


lambda_values, lambda_df, beta = fit_multinomial_logit(voter_pca=voter_pca, party_pca=party_pca)


# # 6) build the vote‐share function V_j(z):
# def V_of_z(Z_new):
#     """
#     Z_new: p×2 array of any hypothetical party locations in PC-space.
#     returns: length‐p array of predicted vote shares.
#     """
#     d2 = ((Y[:,None,:] - Z_new[None,:,:])**2).sum(axis=2)  # (n,p)
#     U  = lambda_hat[None,:] - beta_hat * d2               # (n,p)
#     P  = np.exp(U)
#     P /= P.sum(axis=1)[:,None]
#     return P.mean(axis=0)                                 # average over voters

# # 7) check that at the fitted party locations you recover your sample shares
# Z_obs = Z  # the PC1/PC2 from party_pca
# fitted_shares = V_of_z(Z_obs)
# print("Fitted vote shares:", fitted_shares)




# def do_regression(count: pd.Series, deg: int = 1, normalize: bool=False) -> LinearRegression:
#     """simple polynomial regression of degree *deg*

#     Parameters
#     ----------
#     count : pd.Series
#         how often each unique answer was given
#     deg : int, optional
#         degree of the polynomial, by default 1
#     normalize : bool, optional
#         if values should be normalized to [0,1] before fitting the model, by default False

#     Returns
#     -------
#     LinearRegression
#         fitted model
#     """
#     poly = PolynomialFeatures(degree=deg)

#     x_min, x_max = count.index.levels[0].min(), count.index.levels[0].max()
#     y_min, y_max = count.index.levels[1].min(), count.index.levels[1].max()

#     X = np.arange(x_min, x_max+1)
#     Y = np.arange(y_min, y_max+1)
#     X, Y = np.meshgrid(X, Y)
#     X, Y = X.flatten(), Y.flatten()

#     Z = count.unstack().values.T.flatten()
#     if normalize:
#         Z = Z / Z.sum()

#     input_pts = np.stack([X, Y]).T
#     in_features = poly.fit_transform(input_pts)

#     model = LinearRegression(fit_intercept=False)
#     model.fit(in_features, Z)

#     return model


# def plot(model):
#     pass