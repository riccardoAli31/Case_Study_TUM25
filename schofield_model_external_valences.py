import pandas as pd
import numpy as np
from numpy.linalg import eig, eigh
import data_preprocessing.data_preprocess as dp
import pipeline_helper_functions.schofield_model_helper as sm
import matplotlib.pyplot as plt

x_var = "Opposition to Immigration"
y_var = "Welfare State"
year  = "2021"
CHANGE_OPINION = False

# ------------------------------------------------------------- Data Preprocessing ------------------------------------------------------------------------------------------------
party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var, year=year)
party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                            columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})
party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)

if CHANGE_OPINION is True:
    voter_centered = pd.DataFrame()

# ------------------------------------------------------------- Valences from DATA  ------------------------------------------------------------------------
lambda_values, lambda_df = sm.get_external_valences_independent(year=year)

beta = 0.7

# Filter for common parties for 2025 since we don't have new data for party manifesto
if year == '2025':
    parties = sorted(set(lambda_df["Party_Name"]) & set(party_centered["Party_Name"]))
    party_centered = party_centered[party_centered['Party_Name'].isin(parties)]
    lambda_df = lambda_df[lambda_df['Party_Name'].isin(parties)]
    lambda_values = lambda_df["valence"].values

# --------------------------------------------------------- Equilibrium conditions Check ---------------------------------------------------------------------------------------

equilibrium_conditions_df = sm.check_equilibrium_conditions(lambda_df=lambda_df, lambda_values=lambda_values, beta=beta,
                                                            voter_centered=voter_centered, x_var=x_var, y_var=y_var)
print(equilibrium_conditions_df)

# ----------------------------------------------------- Movement Recommendations for each party ---------------------------------------------------------------------------------------
# Compute matrices C_j for each party and gather eigen‐info
all_party_movements_df = sm.compute_characteristic_matrices(lambda_values=lambda_values,
                                            beta=beta,
                                            voter_centered=voter_centered,
                                            party_centered=party_centered,
                                            lambda_df=lambda_df,
                                            x_var=x_var,
                                            y_var=y_var)

pd.set_option('display.max_columns', None)
print("\n----- Characteristic Matrices & Movement Recommendations -----\n")
print(f"\n===== MODEL EXTERNAL VALENCES =====\n")
print(all_party_movements_df.to_string(index=False))

# ------------------------------------------------------ ANALYZE Saddle and Local Min. Points -----------------------------------------------------------------------------------
# sort λ-DF by class_index
lambda_df2 = lambda_df.sort_values("class_index").reset_index(drop=True)
# grab the ordered party names & their indices
party_order = lambda_df2["Party_Name"]
indices     = lambda_df2["class_index"].to_numpy(dtype=int)
# re-index by Party_Name
party_centered2 = party_centered.set_index("Party_Name").loc[party_order].reset_index()
all_party_movements2 = all_party_movements_df.drop(columns="class_index").set_index("Party_Name").loc[party_order].reset_index().assign(class_index=indices)      
# reorder array of lambda values
lambda_values2 = lambda_df2['valence'].values

# Collect equilibrium results
equilibrium_results = []
# automatically pick out which parties need saddle‐point moves
saddle_targets = all_party_movements2.loc[all_party_movements2["action"].str.contains("Saddle", case=False), "Party_Name"].tolist()
# and which ones need local‐min‐point moves
local_min_targets = all_party_movements2.loc[all_party_movements2["action"].str.contains("local minimum", case=False), "Party_Name"].tolist()

# --- compute saddle results ---
for party in saddle_targets:
    v_pos, t_opt, share_opt = sm.compute_optimal_movement_saddle_position(
        lambda_values     = lambda_values2,
        lambda_df         = lambda_df2,
        voter_centered    = voter_centered,   
        party_centered    = party_centered2,         
        beta              = beta,
        x_var             = x_var,
        y_var             = y_var,
        target_party_name = party)
    equilibrium_results.append({
        "party":        party,
        "type":         "saddle",
        "direction_x":  float(v_pos[0]),
        "direction_y":  float(v_pos[1]),
        "t_opt":        float(t_opt),
        "share_opt":    float(share_opt)})

# --- compute local‐min results ---
for party in local_min_targets:
    z_opt, share_opt, info = sm.compute_optimal_movement_local_min_position(
        lambda_values     = lambda_values2,
        lambda_df         = lambda_df2,
        voter_centered    = voter_centered,
        party_centered    = party_centered2,
        target_party_name = party,
        x_var             = x_var,
        y_var             = y_var)
    equilibrium_results.append({
        "party":           party,
        "type":            "local_min",
        "optimal_position": z_opt,
        "share_opt":       float(share_opt)})
        
# put it all in one DataFrame
equilibrium_results_df = pd.DataFrame(equilibrium_results)
print(equilibrium_results_df)

# --------------------------------------------------- Plot data cloud and equilibrium positions -----------------------------------------------------------------------------------
fig_1, ax = sm.plot_external_valence_equilibrium(equilibrium_results_df=equilibrium_results_df, voter_centered=voter_centered,
                                                 party_centered=party_centered, x_var=x_var, y_var=y_var, year=year)
plt.show()