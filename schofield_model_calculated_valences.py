import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp
from numpy.linalg import eig, eigh
import plotly.express as px

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


def plot_centered_with_arrow(voter_centered: pd.DataFrame, party_centered: pd.DataFrame, x_var: str, y_var: str,
                             eigvals_C1: np.ndarray, eigvecs_C1: np.ndarray, j0: int, party0: str, arrow_length: float = 0.5) -> None:
    """
    Plots all centered voter+party points and, if there is a positive eigenvalue in C1,
    draws an arrow from the lowest‐valence party's centered coordinate in the direction
    of that eigenvector.
    """

    # 1) Concatenate the two DataFrames so we can plot them together:
    concatenated_df = pd.concat([voter_centered, party_centered], ignore_index=True)

    # 2) Build the scatter plot of all CENTERED points, colored & symbolized by "Label"
    fig = px.scatter(
        concatenated_df,
        x=f"{x_var} Centered",
        y=f"{y_var} Centered",
        color="Label",
        symbol="Label",
        title="Centered Voter & Party Positions with LSNE Arrow"
    )
    fig.update_traces(marker=dict(size=8))

    # 3) Find the index of the *first* positive eigenvalue in eigvals_C1:
    idx_pos = None
    for idx, val in enumerate(eigvals_C1):
        if val > 0:
            idx_pos = idx
            break

    if idx_pos is not None:
        # 3a) Extract the corresponding eigenvector and normalize it
        vec_pos = eigvecs_C1[:, idx_pos]
        vec_pos = vec_pos / np.linalg.norm(vec_pos)

        # 3b) Tail of arrow = the j0-th party's CENTERED coordinates:
        x0 = float(party_centered.loc[j0, f"{x_var} Centered"])
        y0 = float(party_centered.loc[j0, f"{y_var} Centered"])

        # 3c) Compute the head of the arrow by adding arrow_length * (unit eigenvector)
        dx = vec_pos[0] * arrow_length
        dy = vec_pos[1] * arrow_length

        # 3d) Add the arrow annotation to the figure
        fig.add_annotation(
            x=x0 + dx,
            y=y0 + dy,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            standoff=2,
            text=f"j₀ = {party0}",    # Label at the arrowhead
            font=dict(size=12, color="black")
        )
    else:
        # No positive eigenvalue → no arrow drawn
        print("No positive eigenvalue found in C1; arrow not drawn.")

    # 4) Final layout touches
    fig.update_layout(
        xaxis_title=f"{x_var} Centered",
        yaxis_title=f"{y_var} Centered",
        legend_title="Label (party index)"
    )

    # 5) Show the figure
    fig.show()


def compute_characteristic_matrices(lambda_values: np.ndarray, beta: float, voter_centered: pd.DataFrame, party_centered: pd.DataFrame,
                                    x_var: str, y_var: str) -> pd.DataFrame:

    # 1) Number of parties (p) and voter‐covariance V*
    p = len(lambda_values)

    # 1a) Compute steady‐state shares rho_j = exp(lambda_j) / sum_k exp(lambda_k)
    expL = np.exp(lambda_values)
    rho = expL / expL.sum()

    # 1b) Compute V* = 1/n sum_i ( [x_i^2, x_i y_i; x_i y_i, y_i^2] ), using centered coords
    xi_1 = voter_centered[f"{x_var} Centered"].to_numpy()
    xi_2 = voter_centered[f"{y_var} Centered"].to_numpy()
    n = len(xi_1)

    Vstar = np.zeros((2,2), dtype=float)
    Vstar[0,0] = np.dot(xi_1, xi_1) / n
    Vstar[1,1] = np.dot(xi_2, xi_2) / n
    Vstar[0,1] = Vstar[1,0] = np.dot(xi_1, xi_2) / n

    # 1c) The 2×2 identity
    I2 = np.eye(2)

    # 2) Prepare containers for results
    rows = []

    # 3) Iterate over each party j = 0..p-1
    for j in range(p):
        # 3a) Compute A_j = beta * (1 - 2*rho_j)
        Aj = beta * (1 - 2 * rho[j])

        # 3b) Build C_j = 2 * A_j * Vstar - I2
        Cj = 2 * Aj * Vstar - I2

        # 3c) Compute eigenvalues & eigenvectors of Cj
        #     We use eigh() since Cj is symmetric. eigh returns them in ascending order, so we'll reverse.
        eigvals, eigvecs = eigh(Cj)  
        # eigh returns eigvals sorted: [mu_small, mu_large], and columns of eigvecs are their eigenvectors
        mu1, mu2 = eigvals[::-1]            # now mu1 >= mu2
        v1 = eigvecs[:, ::-1].T[0]           # eigenvector for mu1
        v2 = eigvecs[:, ::-1].T[1]           # eigenvector for mu2

        # 3d) Decide on the “action” based on signs of mu1, mu2
        if (mu1 < 0) and (mu2 < 0):
            action = "No movement needed (local max)."
        elif (mu1 > 0 and mu2 < 0):
            action = "Saddle → move along eigenvector for μ₁>0."
        elif (mu2 > 0 and mu1 < 0):
            action = "Saddle → move along eigenvector for μ₂>0."
        elif (mu1 == 0 or mu2 == 0):
            action = "Zero eigenvalue → boundary or degenerate case."
        else:
            # If both > 0, that is a local minimum at the origin (rare in LSNE context)
            action = "Both μ>0 → origin is local minimum (move away in any linear combo)."

        # 3e) Grab the party name from party_centered
        party_name = party_centered.loc[j, "Party_Name"]

        # 3f) Append a result row
        rows.append({
            "class_index": j,
            "Party_Name":  party_name,
            "A_j":         Aj,
            "mu_1":        float(np.round(mu1, 6)),
            "mu_2":        float(np.round(mu2, 6)),
            "v_1":         v1,   # length-2 array
            "v_2":         v2,   # length-2 array
            "action":      action
        })

    # 4) Build a DataFrame from the rows
    result_df = pd.DataFrame(rows)
    # Sort rows by class_index just for neatness (optional)
    result_df = result_df.sort_values("class_index").reset_index(drop=True)
    return result_df


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

    # Plot data and moving direction of lowest-valence party
    plot_centered_with_arrow(voter_centered=voter_centered, party_centered=party_centered,
                             x_var=x_var, y_var=y_var, eigvals_C1=eigvals_C1, eigvecs_C1=eigvecs_C1, j0=j0, party0=party0)
    
    # Compute matrices C_j for each party and gather eigen‐info
    char_df = compute_characteristic_matrices(lambda_values=lambda_values, beta=beta, voter_centered=voter_centered, party_centered=party_centered,
                                            x_var=x_var, y_var=y_var)
    pd.set_option('display.max_columns', None)
    print("\n----- Characteristic Matrices & Movement Recommendations -----\n")
    print(char_df)
    print("\n")
