import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp
from numpy.linalg import eig

x_var='Democracy'
y_var='Political Corruption'

party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)

party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f"{x_var}_voter_lin Scaled", f"{y_var}_voter_lin Scaled", "Label"]].rename(columns={
    f'{x_var}_voter_lin Scaled': f'{x_var} Scaled',
    f'{y_var}_voter_lin Scaled': f'{y_var} Scaled'})

party_pca, voter_pca = dp.center_rotate_data_cloud(party_scaled_df, voter_scaled, x_var=x_var, y_var=y_var)


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
        # max_iter=1000
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


# X = voter_pca[['PC1','PC2']].values
# V_star = np.cov(X, rowvar=False, bias=True)   # 2×2 matrix

# results = []
# for j, party in lambda_df.iterrows():
#     pj = np.exp(lambda_values[j]) / np.exp(lambda_values).sum()
#     A_j = beta * (1 - 2*pj)
#     Cj  = 2*A_j * V_star - np.eye(2)
#     eigvals, eigvecs = eig(Cj)
#     results.append({
#       'Party': party.Party_Name,
#       'eigvals': eigvals,
#       'eigvecs': eigvecs
#     })

# # print summary
# for r in results:
#     print(r['Party'], 'eigenvalues =', np.round(r['eigvals'],3))
