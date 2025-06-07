import pandas as pd
import numpy as np
from numpy.linalg import eig, eigh
import data_preprocessing.data_preprocess as dp
import pipeline_helper_functions.schofield_model_helper as sm

x_var="Democracy"
y_var="Environmental Protection"

# ------------------------------------------------------------- Data Preprocessing ---------------------------------------------------------------------------------------
party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)
party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                            columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})
party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)

# --------------------------------------------------- Valences from Multinomial Logistic Regression ------------------------------------------------------------------------
lambda_values, lambda_df = sm.fit_multinomial_logit(voter_centered=voter_centered, party_centered=party_centered, x_var=x_var, y_var=y_var)

# external valences
def get_external_valences():
    politicians = {"chrupalla": "AfD", "soeder": "CDU/CSU", "scholz": "SPD", "laschet": "CDU/CSU", "baerbock": "90/Greens", "weidel": "AfD", 
                "lindner": "FDP", "wissler": "LINKE", "bartsch": "LINKE"}
    valences = dp.get_valence_from_gesis(politicians)

    # sort it in the same order as lambda_values
    custom_dict = {"90/Greens": 0, "LINKE": 1, "SPD": 2, "FDP": 3, "CDU/CSU": 4, "AfD": 5}
    valences = valences.sort_values(by="party", key=lambda x: x.map(custom_dict))
    
    return valences["valence"].values

lambda_values = get_external_valences()


beta = 0.6

# ---------------------------------------------------------- Equilibrium conditions Check ---------------------------------------------------------------------------------------
# Identify the low‐valence party (party “1” in Schofield’s notation)
j0 = np.argmin(lambda_values)
party0 = party_scaled['Party_Name'].iloc[j0]
# Build its A₁ and C₁
expL = np.exp(lambda_values)
rho  = expL / expL.sum()                # steady‐state shares ρ_j
A    = beta * (1 - 2*rho)               # A_j = β(1–2ρ_j)
A1   = A[j0]                            # this is A₁
# Characteristic matrix C₁ = 2 A₁ V* – I
xi_1 = voter_centered[f'{x_var} Centered'].values
xi_2 = voter_centered[f'{y_var} Centered'].values
covariance_matrix = np.zeros((2,2))
covariance_matrix[0,0] = np.dot(xi_1, xi_1)
covariance_matrix[1,1] = np.dot(xi_2, xi_2)
covariance_matrix[0,1] =  covariance_matrix[1,0] = np.dot(xi_1, xi_2)
covariance_matrix *= 1 / len(xi_1)
I2   = np.eye(2)
C1   = 2 * A1 * covariance_matrix - I2
# Eigen‐decompose C₁
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

# ------------------------------------------------- Movement Recommendations for each party ---------------------------------------------------------------------------------------
# Compute matrices C_j for each party and gather eigen‐info
char_df = sm.compute_characteristic_matrices(lambda_values=lambda_values, beta=beta, voter_centered=voter_centered, party_centered=party_centered,
                                        x_var=x_var, y_var=y_var)
pd.set_option('display.max_columns', None)
print("\n----- Characteristic Matrices & Movement Recommendations -----\n")
print(char_df)
print("\n")

# ---------------------------------------------------------- ANALYZE Saddle Points -----------------------------------------------------------------------------------
saddle_targets = ["AfD", "CDU/CSU", "LINKE"]   
results_saddle = []

if saddle_targets:
    for tgt in saddle_targets:
        v_pos, t_opt, share_opt = sm.compute_optimal_movement_saddle_position(
            lambda_values     = lambda_values,
            voter_centered    = voter_centered,
            party_centered    = party_centered,
            beta              = beta,
            x_var             = x_var,
            y_var             = y_var,
            target_party_name = tgt)
        results_saddle.append({
            "party":        tgt,
            "direction_x":  float(v_pos[0]),
            "direction_y":  float(v_pos[1]),
            "t_opt":        float(t_opt),
            "share_opt":    float(share_opt)
        })
equilibrium_saddle_df = pd.DataFrame(results_saddle)
print(equilibrium_saddle_df)

# -------------------------------------------------------- ANALYZE Local Minimum Points -----------------------------------------------------------------------------------
local_min_targets = []  
local_min_results = []

if local_min_targets:
    for tgt in local_min_targets:
        z_opt, share_opt, info = sm.compute_optimal_movement_local_min_position(
            lambda_values     = lambda_values,
            voter_centered    = voter_centered,
            party_centered    = party_centered,
            target_party_name = tgt,
            x_var             = x_var,
            y_var             = y_var
        )
        local_min_results.append({
            "party":        tgt,
            "optimal_position":  z_opt,
            "share_opt":    float(share_opt)
        })

local_min_df = pd.DataFrame(local_min_results)
print(local_min_df)

# --------------------------------------------------- Plot data cloud and equilibrium positions -----------------------------------------------------------------------------------
directions_saddle_points = {}
for _, row in equilibrium_saddle_df.iterrows():
    party = row["party"]
    v_pos = np.array([row["direction_x"], row["direction_y"]])
    t_opt = row["t_opt"]
    directions_saddle_points[party] = (v_pos, t_opt)

optima_min_points = {}
for _, row in local_min_df.iterrows():
    party = row["party"]
    optimal_position = row["optimal_position"]
    optima_min_points[party] = optimal_position

fig = sm.plot_optima(voter_centered = voter_centered, party_centered = party_centered,
                  directions = directions_saddle_points, optima = optima_min_points, x_var = x_var, y_var = y_var)
fig.show()

