import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp
from numpy.linalg import eig

x_var='Democracy'
y_var='Education Expansion'


def fit_multinomial_logit(voter_scaled: pd.DataFrame, party_scaled: pd.DataFrame, 
                          scaled_cols: tuple[str,str] = (f"{x_var} Scaled", f"{y_var} Scaled")) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fits a “simple” multinomial logit using sklearn on the (n × p) matrix D of negative squared distances:
        D[i,j] = -|| x_i - z_j ||^2,
    where x_i is voter i’s 2D coordinate and z_j is party j’s 2D coordinate.

    Returns:
      - lambda_vals: length‐p array of intercepts (λ_j), ordered so lambda_vals[j] is party‐j’s intercept.
      - lambda_df  : DataFrame with columns ["class_index","Party_Name","Valence"], sorted by descending Valence.
      - beta_hat   : a single scalar, taken as the average of diag(coef_matrix).  (Warning printed if off‐diagonals are large.)
    """

    # 1) Extract voter and party coordinates as NumPy arrays
    Y = voter_centered[list(scaled_cols)].to_numpy(dtype=float)   # shape = (n, 2)
    Z = party_centered[list(scaled_cols)].to_numpy(dtype=float)   # shape = (p, 2)
    n, p = Y.shape[0], Z.shape[0]

    # 2) Build the distance matrix dist2[i,j] = ||Y[i] - Z[j]||^2, then D = -dist2
    dist2 = ((Y[:, None, :] - Z[None, :, :])**2).sum(axis=2)  # shape = (n, p)
    D = -dist2                                              # shape = (n, p)

    # 3) Extract the integer “true choice” y_i ∈ {0,...,p-1} from voter_centered["party_choice"]
    if "party_choice" not in voter_scaled.columns:
        raise KeyError("voter_scaled must contain a column 'party_choice' with integer party indices 0..p-1.")
    y_choices = voter_scaled["party_choice"].to_numpy(dtype=int)
    # Sanity check: the unique labels should be exactly {0,1,...,p-1}
    unique_labels = np.unique(y_choices)
    if set(unique_labels.tolist()) != set(range(p)):
        raise ValueError(
            f"Found labels = {unique_labels.tolist()}, but expected exactly [0..{p-1}]."
        )

    # 4) Fit sklearn’s multinomial logit on (D, y_choices):
    clf = LogisticRegression(penalty=None, solver='lbfgs', multi_class='multinomial', fit_intercept=True)
    clf.fit(D, y_choices)

    # 5) Extract intercepts 
    raw_intercepts = clf.intercept_  
    classes        = clf.classes_.astype(int)  # e.g. [0,1,...,p-1]

    # Build lambda_vals[j] = intercept for class=j
    lambda_vals = np.zeros(p, dtype=float)
    for idx, cls in enumerate(classes):
        lambda_vals[cls] = raw_intercepts[idx]

    # 8) Build a DataFrame of (class_index, Party_Name, Valence=λ_j) sorted descending by Valence
    party_names = party_scaled["Party_Name"].reset_index(drop=True)
    if len(party_names) != p:
        raise ValueError(f"party_centered has {len(party_names)} rows but expected p={p}.")
    lambda_df = pd.DataFrame({
        "class_index": np.arange(p),
        "Party_Name":   party_names.values,
        "Valence":      lambda_vals
    }).sort_values("Valence", ascending=False).reset_index(drop=True)

    return lambda_vals, lambda_df


if __name__ == "__main__":

    party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)

    party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                                columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})

    party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)

    lambda_values, lambda_df = fit_multinomial_logit(voter_scaled=voter_scaled, party_scaled=party_scaled)
    beta = 0.6

    # 1) Identify the low‐valence party (party “1” in Schofield’s notation)
    j0 = np.argmin(lambda_values)
    party0 = party_scaled['Party_Name'].iloc[j0]

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
