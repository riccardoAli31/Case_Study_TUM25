import pandas as pd
import numpy as np
from numpy.linalg import eig, eigh
import data_preprocessing.data_preprocess as dp
import pipeline_helper_functions.schofield_model_helper as sm

x_var='Democracy'
y_var='Education Expansion'

# --------------------------------------------------------------------------- Data Preprocessing ---------------------------------------------------------------------------------------
party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var)
party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                            columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})
party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)

# -------------------------------------------------------------- Valences from Multinomial Logistic Regression ------------------------------------------------------------------------
lambda_values, lambda_df = sm.fit_multinomial_logit(voter_centered=voter_centered, party_centered=party_centered)
beta = 0.7

# ---------------------------------------------------------------------- Equilibrium conditions Check ---------------------------------------------------------------------------------------
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

# -------------------------------------------------------------- Movement Recommendations for each party ---------------------------------------------------------------------------------------
# Compute matrices C_j for each party and gather eigen‐info
char_df = sm.compute_characteristic_matrices(lambda_values=lambda_values, beta=beta, voter_centered=voter_centered, party_centered=party_centered,
                                        x_var=x_var, y_var=y_var)
pd.set_option('display.max_columns', None)
print("\n----- Characteristic Matrices & Movement Recommendations -----\n")
print(char_df)
print("\n")

# -------------------------------------------------------------------------- ANALYZE Saddle Points -----------------------------------------------------------------------------------
target_1 = "AfD"
v_pos_afd, t_opt_afd, share_opt_afd = sm.compute_optimal_movement_saddle_position(
    lambda_values   = lambda_values,
    voter_centered  = voter_centered,
    party_centered  = party_centered,
    beta            = beta,
    x_var           = x_var,
    y_var           = y_var,
    target_party_name = target_1)
print(f"Party {target_1} moves along direction {v_pos_afd.round(3)}; "
      f"optimal t = {t_opt_afd:.3f}; max share ≈ {share_opt_afd:.3f}")

# -------------------------------------------------------------------------- ANALYZE Local Minimum Points -----------------------------------------------------------------------------------
z_fdp_opt, fdp_share_opt, info = sm.compute_optimal_movement_local_min_position(
    lambda_values    = lambda_values,
    voter_centered   = voter_centered,
    party_centered   = party_centered,
    target_party_name= "FDP",
    x_var            = x_var,
    y_var            = y_var)

z_cdu_opt, cdu_share_opt, info = sm.compute_optimal_movement_local_min_position(
    lambda_values    = lambda_values,
    voter_centered   = voter_centered,
    party_centered   = party_centered,
    target_party_name= "CDU/CSU",
    x_var            = x_var,
    y_var            = y_var)

# -------------------------------------------------------------------- Plot data cloud and equilibrium positions -----------------------------------------------------------------------------------
directions = {"AfD": (v_pos_afd, t_opt_afd)}

optima = {"FDP": z_fdp_opt, "CDU/CSU": z_cdu_opt}

fig = sm.plot_optima(voter_centered = voter_centered, party_centered = party_centered,
                  directions = directions, optima = optima, x_var = x_var, y_var = y_var)
fig.show()

