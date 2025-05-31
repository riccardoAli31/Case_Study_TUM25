import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp
from numpy.linalg import eig

x_var='Democracy'
y_var='Education Expansion'


def fit_multinomial_logit(voter_centered: pd.DataFrame, party_centered: pd.DataFrame, 
                          pc_cols: tuple[str,str] = (f"{x_var} Centered", f"{y_var} Centered")) -> tuple[np.ndarray, pd.DataFrame, float]:
    
    # 1) Extract the 2D coordinates of voters (Y) and parties (Z)
    Y = voter_centered[list(pc_cols)].to_numpy()   # shape = (n, 2)
    Z = party_centered[list(pc_cols)].to_numpy()   # shape = (p, 2)
    # n, p = Y.shape[0], Z.shape[0]

    # 2) Build the (n × p) matrix of negative squared distances:
    #      D[i, j] = -|| Y[i,:] - Z[j,:] ||^2
    dist2 = ((Y[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)  # (n, p) of ||y_i - z_j||^2
    D = -dist2                                                  # (n, p) of -||y_i - z_j||^2

    # 3) Extract the “true choice” of each voter (an integer in 0..p-1)
    if "party_choice" not in voter_centered.columns:
        raise KeyError("voter_centered must contain a column 'party_choice' with integer party indices 0..p-1.")
    y_choices = voter_centered["party_choice"].to_numpy().astype(int)  # shape = (n,)

    # 4) Fit a standard p‐class multinomial logit on (D, y_choices)
    clf = LogisticRegression(
        penalty=None,                       # no penalty so that intercepts + slopes are unrestricted
        solver='lbfgs',                     # LBFGS handles multiclass natively
        multi_class='multinomial',
        fit_intercept=True)
    clf.fit(D, y_choices)

    # 5) Extract each party’s intercept λ_j
    lambda_vals = clf.intercept_.copy()   # shape = (p,)

    # 6) Recover a single β by looking at the diagonal of clf.coef_.
    #    _coef_ has shape (p, p): row j is the slope‐vector for class=j,
    #    column k is the coefficient on D[:,k] (which is -||y - z_k||^2).
    #
    #    In a pure conditional logit U_{i, j} = λ_j + β * D[i, j],
    #    the only nonzero coefficient in row j would be the j-th column (and that value would be β).
    #    Off‐diagonals should be ≈ 0.  Thus, we estimate a single β as
    #       β_hat ≈ − mean( diag( clf.coef_ ) ).
    coef_matrix = clf.coef_            # shape = (p, p)
    diag_coeffs = np.diag(coef_matrix) # length‐p: [ coef[j,j] for j in 0..p-1 ]
    beta_val = -diag_coeffs.mean()     # take negative of average diagonal

    # 7) Build a DataFrame for λ_j (Valence) + party names
    #    We assume party_centered["Party_Name"] is aligned so that row-index j corresponds
    #    to the j-th party.  If there's any re‐indexing, make sure to reset_index(drop=True).
    party_names = party_centered["Party_Name"].reset_index(drop=True)
    classes = clf.classes_.astype(int)  # should be array([0,1,2,...,p-1])

    lambda_df = pd.DataFrame({
        "class_index": classes,
        "Party_Name":   party_names.loc[classes].values,
        "Valence":      lambda_vals
    })
    # Sort by descending Valence so that highest‐valence party comes first
    lambda_df = lambda_df.sort_values("Valence", ascending=False).reset_index(drop=True)

    return lambda_vals, lambda_df, beta_val


if __name__ == "__main__":

    party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)

    party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                                columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})

    party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)
    
    lambda_values, lambda_df, beta = fit_multinomial_logit(voter_centered=voter_centered, party_centered=party_centered)

    # 1) Identify the low‐valence party (party “1” in Schofield’s notation)
    j0 = np.argmin(lambda_values)
    party0 = party_centered['Party_Name'].iloc[j0]

    # 2) Build its A₁ and C₁
    expL = np.exp(lambda_values)
    rho  = expL / expL.sum()                # steady‐state shares ρ_j
    A    = beta * (1 - 2*rho)               # A_j = β(1–2ρ_j)
    A1   = A[j0]                            # this is A₁

    # 3) Characteristic matrix C₁ = 2 A₁ V* – I
    xi_1 = voter_centered[f'{x_var} Centered'].values
    xi_2 = voter_centered[f'{y_var} Centered'].values
    covariance_matrix = np.zeros((2,2))
    covariance_matrix[0,0] = np.dot(xi_1, xi_1)
    covariance_matrix[1,1] = np.dot(xi_2, xi_2)
    covariance_matrix[0,1] =  covariance_matrix[1,0] = np.dot(xi_1, xi_2)
    covariance_matrix *= 1 / len(xi_1)

    I2   = np.eye(2)
    C1   = 2 * A1 * covariance_matrix - I2

    # 4) Eigen‐decompose C₁
    eigvals_C1, eigvecs_C1 = eig(C1)

    print(f"Lowest‐valence party is {party0!r}")
    print("Eigenvalues of C₁:", np.round(eigvals_C1,3))
    print("Eigenvectors (as columns):\n", np.round(eigvecs_C1,3))

    # —–– 1) Necessary condition for joint origin LSNE —––
    nec = np.all(eigvals_C1 < 0)
    print("Necessary condition (all eig(C₁)<0):", nec)

    # —–– 2) Sufficient condition (Corollary 2) —––
    # In 2D, ν² = trace(V*)
    nu2 = np.trace(covariance_matrix)
    c   = 2 * A1 * nu2
    print(f"Convergence coeff. c = 2·A₁·ν² = {c:.3f}")

    suf = (c < 1)
    print("Sufficient condition (c<1):", suf)
