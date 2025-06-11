import pandas as pd
import numpy as np
from numpy.linalg import eig, eigh
import data_preprocessing.data_preprocess as dp
import pipeline_helper_functions.schofield_model_helper as sm

x_var = "Opposition to Immigration"
y_var = "Welfare State"
year  = "2021"

# ------------------------------------------------------------- Data Preprocessing ------------------------------------------------------------------------------------------------
party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var, year=year)
party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                            columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})
party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)

# ------------------------------------------- Valences from Multinomial Logistic Regression and from DATA  ------------------------------------------------------------------------
lambda_values_logit, lambda_df_logit = sm.fit_multinomial_logit(voter_centered=voter_centered, party_centered=party_centered, x_var=x_var, y_var=y_var)

lambda_values_external, lambda_df_external = sm.get_external_valences(lambda_df_logit=lambda_df_logit, year=year)

# common party indices between two models
common_idx = sorted(set(lambda_df_logit["class_index"]) & set(lambda_df_external["class_index"]))

# filter & re‐order DataFrames for the common parties between two models
lambda_df_logit  = (lambda_df_logit.loc[lambda_df_logit["class_index"].isin(common_idx)].set_index("class_index")
                    .loc[common_idx].reset_index())
lambda_df_external = (lambda_df_external.loc[lambda_df_external["class_index"].isin(common_idx)].set_index("class_index")
                      .loc[common_idx].reset_index())

# subset your valence‐arrays by those same indices
lambda_values_logit    = lambda_df_logit["valence"].to_numpy()
lambda_values_external = lambda_df_external["valence"].to_numpy()

beta = 0.7

# --------------------------------------------------------- Equilibrium conditions Check ---------------------------------------------------------------------------------------
models = [("logit", lambda_values_logit, lambda_df_logit), ("external", lambda_values_external, lambda_df_external)]
equilibrium_conditions = []

for model_name, lambda_values, lambda_df in models:
    # 1) identify the low‐valence party
    min_row = lambda_df.loc[lambda_df["valence"].idxmin()]
    party0  = min_row["Party_Name"]
    j0 = lambda_df.reset_index().query("valence == @lambda_df.valence.min()").index[0]

    # 2) steady‐state shares ρ_j
    expL = np.exp(lambda_values)
    rho  = expL / expL.sum()
    
    # 3) compute A_j = β(1–2ρ_j), then A₁ = A[j0]
    A = beta * (1 - 2*rho)
    A1 = A[j0]
    
    # 4) build covariance matrix of the two centered dimensions
    xi1 = voter_centered[f'{x_var} Centered'].values
    xi2 = voter_centered[f'{y_var} Centered'].values
    cov = np.cov(np.stack([xi1, xi2]), bias=True)
    
    # 5) C₁ = 2·A₁·V* – I
    C1 = 2 * A1 * cov - np.eye(2)
    
    # 6) eigen‐decomposition
    eigvals, eigvecs = eig(C1)
    
    # 7) check necessary and sufficient conditions
    nec = np.all(eigvals < 0)
    nec_label = "satisfied" if not nec else "not satisfied"
    nu2 = np.trace(cov)
    c   = 2 * A1 * nu2
    suf = (c < 1)
    suf_label = "satisfied" if not nec else "not satisfied"
    
    # 8) append a record with flattened entries
    equilibrium_conditions.append({
        "Model": model_name,
        "LowVal_Party": party0,
        "Eigval_1": eigvals[0],
        "Eigval_2": eigvals[1],
        "Vec1_x": eigvecs[0,0],
        "Vec1_y": eigvecs[1,0],
        "Vec2_x": eigvecs[0,1],
        "Vec2_y": eigvecs[1,1],
        "Necessary_Condition": nec_label,
        "Convergence_Coeff": c,
        "Sufficient_Condition": suf_label
    })

# build the equilibrium_conditions_df
equilibrium_conditions_df = pd.DataFrame(equilibrium_conditions)

# ------------------------------------------------------- Movement Recommendations for each party ---------------------------------------------------------------------------------------
# Compute matrices C_j for each party and gather eigen‐info
all_party_movements = []

for model_name, lambda_values, lambda_df in models:
    print(model_name)
    char_df = sm.compute_characteristic_matrices(lambda_values=lambda_values,
                                                beta=beta,
                                                voter_centered=voter_centered,
                                                party_centered=party_centered,
                                                lambda_df=lambda_df,
                                                x_var=x_var,
                                                y_var=y_var)
    
    # tag it with which model produced it:
    char_df = char_df.copy()
    char_df["Model"] = model_name
    all_party_movements.append(char_df)

# stitch them together
all_party_movements_df = pd.concat(all_party_movements, ignore_index=True)

pd.set_option('display.max_columns', None)
print("\n----- Characteristic Matrices & Movement Recommendations -----\n")

cols = ["class_index", "Party_Name", "mu_1", "mu_2", "action"]
for model, df in all_party_movements_df.sort_values(["Model","class_index"]).groupby("Model"):
    print(f"\n===== {model.upper()} =====\n")
    print(df[cols].to_string(index=False))

# -------------------------------------------------------- ANALYZE Saddle and Local Min. Points -----------------------------------------------------------------------------------
# map model‐name → lambda_values
lambda_map = {"logit": lambda_values_logit, "external": lambda_values_external}
lambda_map_df = {"logit": lambda_df_logit, "external": lambda_df_external}
equilibrium_results = []

for model, mov_df in all_party_movements_df.groupby("Model"):
    λ_vals = lambda_map[model]
    λ_df = lambda_map_df[model]
    parties = λ_df["Party_Name"].tolist()

    party_sub = party_centered.set_index("Party_Name").loc[parties].reset_index()
    # Re-align your λ_vals to exactly that same order:
    λ_series    = pd.Series(λ_vals, index=λ_df["Party_Name"])
    lambda_aligned = λ_series.loc[parties].to_numpy()
    
    # automatically pick out which parties need saddle‐point moves
    saddle_targets = mov_df.loc[mov_df["action"].str.contains("Saddle", case=False), "Party_Name"].tolist()
    
    # and which ones need local‐min‐point moves
    local_min_targets = mov_df.loc[mov_df["action"].str.contains("local minimum", case=False), "Party_Name"].tolist()
    
    # --- compute saddle results ---
    for party in saddle_targets:
        v_pos, t_opt, share_opt = sm.compute_optimal_movement_saddle_position(
            lambda_values     = lambda_aligned,
            lambda_df         = λ_df.loc[λ_df["Party_Name"].isin(parties)],
            voter_centered    = voter_centered,   
            party_centered    = party_sub,         
            beta              = beta,
            x_var             = x_var,
            y_var             = y_var,
            target_party_name = party)
        equilibrium_results.append({
            "Model":        model,
            "party":        party,
            "type":         "saddle",
            "direction_x":  float(v_pos[0]),
            "direction_y":  float(v_pos[1]),
            "t_opt":        float(t_opt),
            "share_opt":    float(share_opt)})
    
    # --- compute local‐min results ---
    for party in local_min_targets:
        z_opt, share_opt, info = sm.compute_optimal_movement_local_min_position(
            lambda_values     = lambda_aligned,
            lambda_df         = λ_df.loc[λ_df["Party_Name"].isin(parties)],
            voter_centered    = voter_centered,
            party_centered    = party_sub,
            target_party_name = party,
            x_var             = x_var,
            y_var             = y_var)
        equilibrium_results.append({
            "Model":           model,
            "party":           party,
            "type":            "local_min",
            "optimal_position": z_opt,
            "share_opt":       float(share_opt)})
        
# put it all in one DataFrame
equilibrium_results_df = pd.DataFrame(equilibrium_results)
print(equilibrium_results_df)

# --------------------------------------------------- Plot data cloud and equilibrium positions -----------------------------------------------------------------------------------

fig = sm.plot_equilibrium_positions(
    all_party_movements_df = all_party_movements_df,
    equilibrium_results_df = equilibrium_results_df,
    voter_centered         = voter_centered,
    party_centered         = party_centered,
    x_var                  = x_var,
    y_var                  = y_var,
    year                   = year
)
fig.show()
