import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp

x_var='Planned Economy'
y_var='Environmental Protection'

party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)

party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f"{x_var}_voter_lin Scaled", f"{y_var}_voter_lin Scaled", "Label"]].rename(columns={
    f'{x_var}_voter_lin Scaled': f'{x_var} Scaled',
    f'{y_var}_voter_lin Scaled': f'{y_var} Scaled'})

party_pca, voter_pca = dp.center_rotate_data_cloud(party_scaled_df, voter_scaled, x_var='Planned Economy', y_var='Environmental Protection')


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


def vote_share_function(Z_new):
    """
    Z_new: either a numpy array (p×2) or a DataFrame with ['PC1','PC2'] columns
    returns: length-p array of vote shares
    """
    # if they passed a DataFrame, grab its PC columns
    if hasattr(Z_new, 'loc'):
        Z_mat = Z_new[['PC1','PC2']].to_numpy()
    else:
        Z_mat = np.asarray(Z_new)

    Y_coords = voter_pca[['PC1','PC2']].to_numpy()
    # now do the usual squared‐distance trick
    d2 = ((Y_coords[:,None,:] - Z_mat[None,:,:])**2).sum(axis=2)  # (n,p)
    U  = lambda_values[None,:] - beta * d2                       # (n,p)
    P  = np.exp(U)
    P /= P.sum(axis=1)[:,None]
    return P.mean(axis=0)


# 1) check electoral mean
mean_coords = voter_pca[['PC1','PC2']].mean().values
print("Electoral mean (PC1, PC2):", mean_coords)

# 2) numeric gradient of vote shares at Z=0
p = party_pca.shape[0]
eps = 1e-6
Z0 = np.zeros((p,2))
grad = np.zeros((p,2))
for j in range(p):
    for d in (0,1):
        Zp = Z0.copy(); Zp[j,d] += eps
        Zm = Z0.copy(); Zm[j,d] -= eps
        Pp = vote_share_function(Zp)
        Pm = vote_share_function(Zm)
        grad[j,d] = (Pp[j] - Pm[j])/(2*eps)

print("Numerical ∂V_j/∂z_j at Z0 (j×d):\n", grad)
# for an equilibrium we want all entries ≈ 0

# 3) build the electoral covariance matrix ∇* = (1/n) YᵀY
Y = voter_pca[['PC1','PC2']].to_numpy()
n = Y.shape[0]
Sigma = (Y.T @ Y) / n
print("∇* =\n", Sigma)

# find the lowest‐valence party
lambda_df = lambda_df.copy().reset_index(drop=True)
j_low = lambda_df['Valence'].idxmin()
λ = lambda_values
λ_low = λ[j_low]

# compute ρ_low = [1 + Σ_{k≠low} exp(λ_k - λ_low)]^{-1}
rho_low = 1.0 / (1.0 + sum(np.exp(λ[k] - λ_low) for k in range(p) if k != j_low))
A_low   = beta * (1 - 2*rho_low)

# characteristic matrix and its eigenvalues
C_low = 2*A_low * Sigma - np.eye(2)
eigs  = np.linalg.eigvals(C_low)
print("Eigenvalues of C_low:", eigs)

# convergence coefficient c = 2β(1-2ρ_low) ν²   where ν² = trace(Σ)
nu2 = np.trace(Sigma)
c   = 2*beta*(1-2*rho_low)*nu2
print("Convergence coefficient c:", c)

# --- checks ---
# • if all(eigs < 0) then the Hessian at z=0 is negative‐definite → LSNE second‐order cond
# • in 2D, a sufficient cond is c < 1; a necessary cond is c ≤ 2

