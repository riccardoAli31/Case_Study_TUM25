import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp
from numpy.linalg import eig
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import logsumexp

x_var='Democracy'
y_var='Education Expansion'


def fit_multinomial_mle(voter_pca, party_pca, pc_cols=(f"{x_var} Centered", f"{y_var} Centered"), choice_col='party_choice'):
    """
    Direct MLE for
      P_ij ∝ exp(λ_j – β * ||y_i – z_j||^2)
    Returns (lambda_vals, lambda_df, beta_hat).
    """
    # 1) pull out coords and choices
    Y = voter_pca[list(pc_cols)].to_numpy()   # (n,2)
    Z = party_pca[list(pc_cols)].to_numpy()   # (p,2)
    n, p = Y.shape[0], Z.shape[0]
    # assume choice_col is integer party‐index 0…p-1
    choices = voter_pca[choice_col].to_numpy().astype(int)

    # 2) compute the matrix of squared distances D2[i,j]
    D2 = ((Y[:,None,:] - Z[None,:,:])**2).sum(axis=2)  # (n,p)

    # 3) negative log‐likelihood
    def negloglik(params):
        lam = params[:-1]        # length p
        beta = params[-1]        # scalar
        # U[i,j] = λ_j – β D2[i,j]
        U = lam[None,:] - beta * D2
        # log‐denominator for each i
        denom = logsumexp(U, axis=1)
        # log‐prob of the chosen alternative
        logp = U[np.arange(n), choices] - denom
        return -logp.sum()

    # 4) initial guess: zero lambdas, beta=1
    init = np.concatenate([np.zeros(p), [1.0]])
    # (optionally: enforce beta>0 with bounds)
    bnds = [(None,None)]*p + [(1e-6, None)]

    res = minimize(negloglik, init, bounds=bnds, method='L-BFGS-B')
    if not res.success:
        raise RuntimeError("MLE failed: " + res.message)

    lam_hat = res.x[:-1]
    beta_hat = res.x[-1]

    # 5) build lambda_df
    party_names = party_pca['Party_Name'].reset_index(drop=True)
    lambda_df = (
      pd.DataFrame({
        'class_index': np.arange(p),
        'Party_Name' : party_names,
        'Valence'    : lam_hat
      })
      .sort_values('Valence', ascending=False)
      .reset_index(drop=True)
    )

    return lam_hat, lambda_df, beta_hat


if __name__ == "__main__":

  party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)

  party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f"{x_var}_voter_lin Scaled", f"{y_var}_voter_lin Scaled", "Label"]].rename(
                                  columns={f'{x_var}_voter_lin Scaled': f'{x_var} Scaled', f'{y_var}_voter_lin Scaled': f'{y_var} Scaled'})

  party_centered, voter_centered = dp.center_party_voter_data(party_scaled_df, voter_scaled, x_var=x_var, y_var=y_var)
    
  # lambda_values, lambda_df, beta = fit_multinomial_mle(voter_pca=voter_centered, party_pca=party_centered)

  # # 1) Identify the low‐valence party (party “1” in Schofield’s notation)
  # j0 = np.argmin(lambda_values)
  # party0 = party_centered['Party_Name'].iloc[j0]

  # # 2) Build its A₁ and C₁
  # expL = np.exp(lambda_values)
  # rho  = expL / expL.sum()                # steady‐state shares ρ_j
  # A    = beta * (1 - 2*rho)               # A_j = β(1–2ρ_j)
  # A1   = A[j0]                            # this is A₁

  # # 3) Characteristic matrix C₁ = 2 A₁ V* – I
  # I2   = np.eye(2)
  # C1   = 2 * A1 * V_star - I2

  # # 4) Eigen‐decompose C₁
  # eigvals_C1, eigvecs_C1 = eig(C1)

  # print(f"Lowest‐valence party is {party0!r}")
  # print("Eigenvalues of C₁:", np.round(eigvals_C1,3))
  # print("Eigenvectors (as columns):\n", np.round(eigvecs_C1,3))

  # # —–– 1) Necessary condition for joint origin LSNE —––
  # nec = np.all(eigvals_C1 < 0)
  # print("Necessary condition (all eig(C₁)<0):", nec)

  # # —–– 2) Sufficient condition (Corollary 2) —––
  # # In 2D, ν² = trace(V*)
  # nu2 = np.trace(V_star)
  # c   = 2 * A1 * nu2
  # print(f"Convergence coeff. c = 2·A₁·ν² = {c:.3f}")

  # suf = (c < 1)
  # print("Sufficient condition (c<1):", suf)
