import pandas as pd
import numpy as np
from numpy.linalg import eig, eigh
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import data_preprocessing.data_preprocess as dp
import data_preprocessing.data_loading as dl


def fit_multinomial_logit(voter_centered: pd.DataFrame, party_centered: pd.DataFrame, x_var:str, y_var:str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fits a “simple” multinomial logit using sklearn on the (n × p) matrix D of negative squared distances:
        D[i,j] = -|| x_i - z_j ||^2,
    where x_i is voter i’s 2D coordinate and z_j is party j’s 2D coordinate.

    Returns:
      - lambda_vals: length‐p array of intercepts (λ_j), ordered so lambda_vals[j] is party‐j’s intercept.
      - lambda_df  : DataFrame with columns ["class_index","Party_Name","Valence"], sorted by descending Valence.
      - beta_hat   : a single scalar, taken as the average of diag(coef_matrix).  (Warning printed if off‐diagonals are large.)
    """
    centered_cols = (f"{x_var} Centered", f"{y_var} Centered")
    # 1) Extract voter and party coordinates as NumPy arrays
    Y = voter_centered[list(centered_cols)].to_numpy(dtype=float)   # shape = (n, 2)
    Z = party_centered[list(centered_cols)].to_numpy(dtype=float)   # shape = (p, 2)
    n, p = Y.shape[0], Z.shape[0]

    # 2) Build the distance matrix dist2[i,j] = ||Y[i] - Z[j]||^2, then D = -dist2
    dist2 = ((Y[:, None, :] - Z[None, :, :])**2).sum(axis=2)  # shape = (n, p)
    D = -dist2                                              # shape = (n, p)

    # 3) Extract the integer “true choice” y_i ∈ {0,...,p-1} from voter_centered["party_choice"]
    if "party_choice" not in voter_centered.columns:
        raise KeyError("voter_centered must contain a column 'party_choice' with integer party indices 0..p-1.")
    y_choices = voter_centered["party_choice"].to_numpy(dtype=int)
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
    party_names = party_centered["Party_Name"].reset_index(drop=True)
    if len(party_names) != p:
        raise ValueError(f"party_centered has {len(party_names)} rows but expected p={p}.")
    lambda_df = pd.DataFrame({
        "class_index": np.arange(p),
        "Party_Name":   party_names.values,
        "valence":      lambda_vals
    }).sort_values("valence", ascending=False).reset_index(drop=True)

    return lambda_vals, lambda_df


def get_external_valences(lambda_df_logit, year):
    # extract the politician names
    party_map = dl.load_party_leaders(year=year)
   
    # fetch the external valences
    valences = dp.get_valence_from_gesis(politicians=party_map, year=year)  

    class_index_map = dict(zip(lambda_df_logit["Party_Name"], lambda_df_logit["class_index"]))
    max_idx       = lambda_df_logit["class_index"].max()
    default_idx   = max_idx + 1  # for any new party
    # Attach a class_index column to the external DF
    valences["class_index"] = valences["Party_Name"].map(lambda p: class_index_map.get(p, default_idx))

    # Re‐index and sort 
    valences = (valences.set_index("class_index").sort_index().reset_index() )

    return valences["valence"].values, valences


def compute_characteristic_matrices(lambda_values: np.ndarray, beta: float, voter_centered: pd.DataFrame, party_centered: pd.DataFrame, lambda_df: pd.DataFrame,
                                    x_var: str, y_var: str) -> pd.DataFrame:

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
    for i, row in lambda_df.reset_index(drop=True).iterrows():
        class_idx  = int(row["class_index"])   
        party_name = row["Party_Name"]
        # 3a) Compute A_j = beta * (1 - 2*rho_j)
        Aj = beta * (1 - 2 * rho[i])

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

        # 3e) Append a result row
        rows.append({
            "class_index": i,
            "Party_Name":  party_name,
            "mu_1":        float(np.round(mu1, 6)),
            "mu_2":        float(np.round(mu2, 6)),
            # "v_1":         v1,   # length-2 array
            # "v_2":         v2,   # length-2 array
            "action":      action
        })

    # 4) Build a DataFrame from the rows
    result_df = pd.DataFrame(rows)
    # Sort rows by class_index just for neatness (optional)
    result_df = result_df.sort_values("class_index").reset_index(drop=True)
    return result_df


def compute_optimal_movement_saddle_position(lambda_values: np.ndarray, lambda_df: pd.DataFrame, voter_centered: pd.DataFrame, party_centered: pd.DataFrame,
                                            beta: float, x_var: str, y_var: str, target_party_name: str) -> tuple[np.ndarray, float, float]:
    """
    Given:
      - lambda_values: length‐p array of intercepts (λ₀, …, λ_{p−1}) from your original fit
      - voter_centered, party_centered: both already centered on (x_var, y_var)
      - beta: the LSNE β parameter
      - x_var, y_var: the two dimension names (e.g. 'Democracy', 'Education Expansion')
      - target_party_name: e.g. 'AfD'

    Returns:
      - v_pos   : a length‐2 numpy array (unit‐normalized eigenvector for C_j’s positive eigenvalue)
      - t_opt   : scalar t ≥ 0 that maximizes the party’s average vote share
      - share_opt: the resulting max average vote share
    """

    # 1) Find the integer index j_idx of the target party
    if target_party_name not in party_centered["Party_Name"].values:
        raise ValueError(f"Party '{target_party_name}' not found in party_centered['Party_Name'].")
    
    row = lambda_df[lambda_df["Party_Name"] == target_party_name].iloc[0]    
    j_idx = lambda_df.index.get_loc(row.name)

    # 2) Recover Y (n×2) and Z (p×2) from the centered DataFrames:
    Y = voter_centered[[f"{x_var} Centered", f"{y_var} Centered"]].to_numpy(dtype=float)   # shape = (n,2)
    Z = party_centered[[f"{x_var} Centered", f"{y_var} Centered"]].to_numpy(dtype=float)   # shape = (p,2)

    # 3) Compute ρ from the provided lambda_values:
    expL = np.exp(lambda_values)
    rho  = expL / expL.sum()   # length‐p array

    # 4) Build the “electoral covariance” V* from the voter coordinates:
    xi = voter_centered[f"{x_var} Centered"].to_numpy()
    yi = voter_centered[f"{y_var} Centered"].to_numpy()
    n  = len(xi)

    Vstar = np.zeros((2,2), dtype=float)
    Vstar[0,0] = np.dot(xi, xi) / n
    Vstar[1,1] = np.dot(yi, yi) / n
    Vstar[0,1] = Vstar[1,0] = np.dot(xi, yi) / n

    I2 = np.eye(2)

    # 5) Characteristic matrix for party j:  C_j = 2·A_j·V* − I₂,  where A_j = β·(1−2ρ_j)
    A = beta * (1 - 2 * rho)    # length‐p
    A_j = A[j_idx]
    C_j = 2 * A_j * Vstar - I2

    # 6) Eigen‐decompose C_j; pick the eigenvector for the strictly positive eigenvalue
    eigvals, eigvecs = eigh(C_j)  # eigh returns sorted ascending
    μ_small, μ_large = eigvals[0], eigvals[1]
    if μ_large <= 0:
        raise RuntimeError(
            f"Party '{target_party_name}' has no strictly positive eigenvalue (largest eigenvalue = {μ_large:.6f})."
        )
    # The column eigvecs[:,1] corresponds to μ_large:
    v_pos = eigvecs[:, 1].real
    v_pos = v_pos / np.linalg.norm(v_pos)   # normalize to length 1

    # 7) Define the average vote share function
    def vote_share_given_t(t_scalar: float) -> float:
        """
        Move party j to (t_scalar * v_pos). Recompute D_new = -||Y − Z_new||²,
        then form logit-numerators D_new[i,j] + λ_j. Finally, do a rowwise softmax
        and return the average probability that i chooses j.
        """
        # 7a) Make a copy of Z and move party j:
        Z_new = Z.copy()
        Z_new[j_idx, :] = t_scalar * v_pos

        # 7b) Compute new squared distances and D_new = -dist²:
        dist2_new = ((Y[:, None, :] - Z_new[None, :, :])**2).sum(axis=2)  # shape (n,p)
        D_new     = -dist2_new

        # 7c) Build logit_numerators (n×p) = D_new + λ_j * 1_nrow
        #     λ_j is length p, so broadcast down each row:
        logit_num = D_new + lambda_values[None, :]    # shape (n,p)

        # 7d) Convert row‐by‐row to probabilities via softmax:
        row_max = np.max(logit_num, axis=1, keepdims=True)
        exp_t   = np.exp(logit_num - row_max)
        denom   = exp_t.sum(axis=1, keepdims=True)
        probs   = exp_t / denom   # shape (n,p)

        # 7e) The average probability that each voter i chooses party j_idx:
        return float(probs[:, j_idx].mean())

    # 8) Now find t_opt that maximizes vote_share
    #    We do a bounded 1D optimization 
    def neg_share(t_scalar: float) -> float:
        return -vote_share_given_t(t_scalar)

    bracket = (-100, 100)
    result  = minimize(
        fun = neg_share,
        x0  = np.array([0]),    
        bounds = [bracket],
        method = "L-BFGS-B"
    )
    t_opt     = float(result.x[0])
    share_opt = vote_share_given_t(t_opt)

    return v_pos, t_opt, share_opt


def compute_optimal_movement_local_min_position(lambda_values: np.ndarray, lambda_df: pd.DataFrame, voter_centered: pd.DataFrame, party_centered: pd.DataFrame, 
                                                target_party_name: str, x_var: str, y_var: str):
    """
    Perform a full 2-D optimization for party `target_party_name` to maximize its average MNL vote share.
    This function does the following steps:
      1) Runs a multi-start optimizer (L-BFGS-B, with Nelder-Mead fallback) from several random jitters.
      2) Prints and returns the best location and share found, plus a label of which method succeeded.

    Inputs:
      - lambda_values   : shape (p,), the intercept array λ₀,...,λ_{p−1} from fit_multinomial_logit.
      - voter_centered  : DataFrame with columns [f"{x_var} Centered", f"{y_var} Centered", "party_choice", ...].
      - party_centered  : DataFrame with columns ["Party_Name", f"{x_var} Centered", f"{y_var} Centered", ...].
      - target_party_name: A string matching one row of party_centered["Party_Name"], e.g. "FDP".
      - x_var, y_var    : The two dimension names, e.g. "Democracy" and "Education Expansion".

    Returns:
      - best_z    : np.ndarray of shape (2,), the optimal (centered) coordinate for the target party.
      - best_share: float, the maximum average vote share achieved by this party.
      - best_info : str, description of how that optimum was found
                    (e.g. "original", "L-BFGS-B from jitter...", or "Nelder-Mead from jitter...").
    """

    # -------------------------------------
    # Extract Y (n×2) and Z_all (p×2) & find j_idx
    # -------------------------------------
    Y = voter_centered[[f"{x_var} Centered", f"{y_var} Centered"]].to_numpy(dtype=float)  # shape = (n,2)
    Z_all = party_centered[[f"{x_var} Centered", f"{y_var} Centered"]].to_numpy(dtype=float)  # shape = (p,2)

    if target_party_name not in party_centered["Party_Name"].values:
        raise ValueError(f"Party '{target_party_name}' not found in party_centered.")
    
    row = lambda_df[lambda_df["Party_Name"] == target_party_name].iloc[0]    
    j_idx = lambda_df.index.get_loc(row.name)
    z0 = Z_all[j_idx].copy()  

    n, p = Y.shape[0], Z_all.shape[0]

    # -------------------------------------
    # Define vote_share(z): average MNL share if party j sits at z
    # -------------------------------------
    def vote_share(z_vec: np.ndarray) -> float:
        """
        Compute the average MNL probability that voters pick party j
        when that party’s coordinate is z_vec.
        """
        # 2a) Compute U_new for all voters i, all parties k:
        #     U_{i j} = -|| x_i - z_vec ||^2 + λ_j
        #     U_{i k} = -|| x_i - Z_all[k] ||^2 + λ_k     (for k != j)
        diff_j = Y - z_vec                                          # shape = (n,2)
        dist2_j = np.sum(diff_j**2, axis=1)                         # shape = (n,)
        U_j = -dist2_j + lambda_values[j_idx]                       # shape = (n,)

        U_new = np.zeros((n, p), dtype=float)
        U_new[:, j_idx] = U_j
        for k in range(p):
            if k == j_idx:
                continue
            diff_k = Y - Z_all[k]                                    # (n,2)
            dist2_k = np.sum(diff_k**2, axis=1)                      # (n,)
            U_new[:, k] = -dist2_k + lambda_values[k]                # (n,)

        # 2b) Convert U_new → probabilities via rowwise softmax
        row_max = np.max(U_new, axis=1, keepdims=True)               # (n,1)
        expU = np.exp(U_new - row_max)                               # (n,p)
        denom = np.sum(expU, axis=1, keepdims=True)                 # (n,1)
        probs = expU / denom                                         # (n,p)
        p_j = probs[:, j_idx]                                        # (n,)

        # 2c) Return the average probability for party j
        return float(np.mean(p_j))

    # -------------------------------------
    # Define obj_and_grad(z): returns (−f(z), −∇f(z)) for SciPy minimize
    # -------------------------------------
    def obj_and_grad(z_vec: np.ndarray):
        """
        Return (−f(z), −∇f(z)) so that SciPy’s `minimize` can minimize −f.
        We use the fact that:
          ∇_z f(z) = (−2/n) ∑_i (z - x_i) p_{i j}(z) [1 - p_{i j}(z)].
        """
        # 3a) Build U_new exactly like in vote_share
        diff_j = Y - z_vec
        dist2_j = np.sum(diff_j**2, axis=1)
        U_j = -dist2_j + lambda_values[j_idx]

        U_new = np.zeros((n, p), dtype=float)
        U_new[:, j_idx] = U_j
        for k in range(p):
            if k == j_idx:
                continue
            diff_k = Y - Z_all[k]
            dist2_k = np.sum(diff_k**2, axis=1)
            U_new[:, k] = -dist2_k + lambda_values[k]

        # 3b) Rowwise softmax → probabilities
        row_max = np.max(U_new, axis=1, keepdims=True)
        expU = np.exp(U_new - row_max)
        denom = np.sum(expU, axis=1, keepdims=True)
        probs = expU / denom                                   # (n,p)
        p_j = probs[:, j_idx]                                  # (n,)

        # 3c) Compute f(z) = ∑ p_j / n
        f_val = np.mean(p_j)

        # 3d) Compute gradient: (−2/n) ∑_i (z - x_i) p_j[i] [1 - p_j[i]]
        w = p_j * (1.0 - p_j)                                   # (n,)
        weighted_diff = diff_j * w[:, None]                     # (n,2)
        grad_f = (-2.0 / n) * np.sum(weighted_diff, axis=0)     # (2,)

        return -f_val, -grad_f

    # -------------------------------------
    # Diagnostic at the original point z0
    # -------------------------------------
    neg_f0, neg_grad0 = obj_and_grad(z0)
    f0 = -neg_f0
    grad0 = -neg_grad0

    # -------------------------------------
    # Multi-start optimization
    # -------------------------------------
    best_z = z0.copy()
    best_share = f0
    best_info = "original (no uphill found)"

    # Pre-generate random offsets
    rng = np.random.default_rng(seed=42)
    jitter_radii = [0.1, 0.2, 0.5, 1.0]

    print("\n=== Multi-start Optimization ===")
    for rad in jitter_radii:
        for trial in range(5):
            θ = 2 * np.pi * rng.random()
            r = rad * rng.random()
            offset = np.array([r * np.cos(θ), r * np.sin(θ)])
            z_start = z0 + offset

            # Try L-BFGS-B from this start
            res = minimize(
                fun=obj_and_grad,
                x0=z_start,
                jac=True,
                method="L-BFGS-B",
                bounds=[(None, None), (None, None)],
                options={"gtol": 1e-4, "ftol": 1e-8, "maxiter": 500, "disp": False}
            )
            if res.success:
                share_res = vote_share(res.x)
                if share_res > best_share + 1e-8:
                    best_share = share_res
                    best_z = res.x.copy()
                    best_info = f"L-BFGS-B from rad={rad:.2f}, trial={trial}"
            else:
                # Fallback to Nelder-Mead if L-BFGS-B fails
                res_nm = minimize(
                    fun=lambda z: -vote_share(z),
                    x0=z_start,
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000, "disp": False}
                )
                if res_nm.success:
                    share_nm = vote_share(res_nm.x)
                    if share_nm > best_share + 1e-8:
                        best_share = share_nm
                        best_z = res_nm.x.copy()
                        best_info = f"Nelder-Mead from rad={rad:.2f}, trial={trial}"

    # -------------------------------------
    # Final results
    # -------------------------------------
    print("\n=== Optimization Results ===")
    print(f"Best location for {target_party_name}: {best_z.round(4)}")
    print(f"Best average share: {best_share:.6f}")
    print(f"Found by: {best_info}")

    return best_z, best_share, best_info


def plot_equilibrium_positions(all_party_movements_df: pd.DataFrame, equilibrium_results_df: pd.DataFrame,
                               voter_centered: pd.DataFrame, party_centered: pd.DataFrame, x_var: str, y_var: str):
    """
    all_party_movements_df: must have columns ['Model','Party_Name','action']
    equilibrium_results_df: must have ['Model','party','type','direction_x','direction_y','t_opt','optimal_position']
    voter_centered:      DataFrame with ['Party_Name', f"{x_var} Centered", f"{y_var} Centered"]
    party_centered:      same structure, but one row per party
    x_var,y_var:         base variable names (e.g. "Democracy","Environmental Protection")
    """
    # 1) restrict to the parties in your movements DF
    parties = all_party_movements_df["Party_Name"].unique()
    vc = voter_centered[ voter_centered["Party_Name"].isin(parties) ]
    pc = party_centered[ party_centered["Party_Name"].isin(parties) ]
    
    # 2) build a quick lookup of each party's current coords
    party_coords = {
        r["Party_Name"]: np.array([r[f"{x_var} Centered"], r[f"{y_var} Centered"]])
        for _, r in pc.iterrows()
    }
    
    # 3) assemble one row per (Model,Party)
    rows = []
    for _, mrow in all_party_movements_df.iterrows():
        model = mrow["Model"]
        party = mrow["Party_Name"]
        action = mrow["action"].lower()
        
        # decide type
        if "local max" in action:
            typ = "no_move"
        elif "saddle" in action:
            typ = "saddle"
        elif "local min" in action:
            typ = "local_min"
        else:
            typ = "no_move"
        
        origin = party_coords[party]
        
        # compute optimum
        if typ == "saddle":
            er = equilibrium_results_df.query(
                "Model==@model and party==@party and type=='saddle'"
            ).iloc[0]
            v     = np.array([er.direction_x, er.direction_y])
            z_opt = origin + er.t_opt * v
        
        elif typ == "local_min":
            er    = equilibrium_results_df.query(
                "Model==@model and party==@party and type=='local_min'"
            ).iloc[0]
            z_opt = np.array(er.optimal_position)
        
        else:  # no_move
            z_opt = origin
        
        rows.append({
            "Model": model,
            "Party": party,
            "Type":  typ,
            "x_opt": float(z_opt[0]),
            "y_opt": float(z_opt[1])
        })
    
    opt_df = pd.DataFrame(rows)
    
    # 4) Plotly‐Express scatter
    fig = px.scatter(
        opt_df,
        x="x_opt", y="y_opt",
        color="Model",
        symbol="Type",
        symbol_map={
            "saddle":    "triangle-up",
            "local_min": "star",
            "no_move":   "circle"
        },
        text="Party",
        title=f"Equilibrium Positions: {x_var} vs {y_var}",
        labels={
            "x_opt": f"{x_var} (Centered)",
            "y_opt": f"{y_var} (Centered)"
        }
    )
    
    # make symbols bigger
    fig.update_traces(marker_size=16)
    
    # 5) add voter cloud (light gray, no legend)
    voter_scatter = px.scatter(
        vc,
        x=f"{x_var} Centered",
        y=f"{y_var} Centered"
    ).update_traces(
        marker=dict(size=4, color="lightgray"),
        showlegend=False
    ).data[0]
    fig.add_trace(voter_scatter)
    
    # 6) add current party centroids (black) + labels
    cent = pc.copy()
    cent["x0"] = cent[f"{x_var} Centered"]
    cent["y0"] = cent[f"{y_var} Centered"]
    cent_scatter = px.scatter(
        cent,
        x="x0", y="y0",
        text="Party_Name"
    ).update_traces(
        marker=dict(size=12, color="black"),
        textposition="top center",
        showlegend=False
    ).data[0]
    fig.add_trace(cent_scatter)
    
    # 7) final layout
    fig.update_layout(
        legend_title="Model & Type"
    )
    
    return fig