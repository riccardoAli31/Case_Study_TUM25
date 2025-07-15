import pandas as pd
import data_preprocessing.data_preprocess as dp
import numpy as np
import pipeline_helper_functions.schofield_model_helper as sm
import pipeline_helper_functions.opinion_dyn_helper as od
import data_preprocessing.sociodemographic_variables_effect as sd
import matplotlib.pyplot as plt

x_var = "Opposition to Immigration"
y_var = "Welfare State"
year = "2009"

include_sociodemographic_variables = False

CHANGE_OPINION = True
sigma_noise = 0.1
gmm_components = 3
alpha = 0.0140
beta = 0.6681
gamma = 2
time = 4

# ------------------------------------------------------------- Data Preprocessing ------------------------------------------------------------------------------------------------
party_scaled, voter_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var, year=year)
party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                                columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})
party_centered, voter_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)
voter_centered = sd.add_age_col_to_voters(voter_df=voter_centered, year=year)
# Fill NaN values with median of the column
medians = voter_centered[[f"{x_var} Centered", f"{y_var} Centered"]].median()
voter_centered[[f"{x_var} Centered", f"{y_var} Centered"]] = voter_centered[[f"{x_var} Centered", f"{y_var} Centered"]].fillna(medians)

print(voter_centered)

x = voter_centered[f"{x_var} Centered"].values
y = voter_centered[f"{y_var} Centered"].values
data = np.vstack([x, y])

if CHANGE_OPINION is True:
    sim = od.run_simulation(data=data, T=time, sigma_noise=sigma_noise,
                            gmm_components=gmm_components, alpha=alpha, beta=beta, gamma=gamma, random_seed=42)
    simulated_df = pd.DataFrame(
        sim.T, columns=[f"{x_var} Centered", f"{y_var} Centered"])
    voter_centered[[f"{x_var} Centered", f"{y_var} Centered"]] = simulated_df

    party_scaled = dp.get_scaled_party_voter_data(x_var=x_var, y_var=y_var, year=str(int(year) + time))[0]
    party_scaled_df = party_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', f'{x_var} Combined', f'{y_var} Combined', 'Label']].rename(
                                    columns={f'{x_var} Combined': f'{x_var} Scaled', f'{y_var} Combined': f'{y_var} Scaled'})
    party_centered = dp.center_party_voter_data(voter_df=voter_scaled, party_df=party_scaled_df, x_var=x_var, y_var=y_var)[0]

# ------------------------------------------------------------- Valences from DATA  ------------------------------------------------------------------------
lambda_values, lambda_df = sm.get_external_valences_independent(year=year)

print("\n===== external valences =====\n")
print(lambda_df)

# alpha: weight for the socio demographic variables
alpha = 3
beta = 0.85

# Filter for common parties for 2025 since we don't have new data for party manifesto
if year == '2025':
    parties = sorted(set(lambda_df["Party_Name"]) & set(party_centered["Party_Name"]))
    party_centered = party_centered[party_centered['Party_Name'].isin(parties)]
    lambda_df = lambda_df[lambda_df['Party_Name'].isin(parties)]
    lambda_values = lambda_df["valence"].values

# ------------------------------------------------------ Sociodemographic Variables Preprocessing ------------------------------------------------------------------------------------------------
######## -----  AGE  ----- ########
theta_age = sd.get_age_effect(voter_centered)
# pad each array with zeros so they all have length == max_len
max_len = max(v.size for v in theta_age.values())
for party, arr in theta_age.items():
    if arr.size < max_len:
        pad_width = max_len - arr.size
        theta_age[party] = np.pad(
            arr, (0, pad_width), 'constant', constant_values=0)
        
# Long dataframe format of theta values for each party and age bracket
theta_df = sd.get_age_effect(voter_centered, translated=True).fillna(0).pivot(index='party', columns='Age_Bracket', values='share')
print("\n===== theta_age =====\n")
print(theta_df)

######## -----  GENDER  ----- ########
theta_gender = sd.get_gender_effect(voter_centered)

######## -----  prepare S matrix - age and gender one-hot encoded variables  ----- ########
ages = voter_centered["Age_Bracket"].to_numpy()
genders = voter_centered["gender"].to_numpy()
n = len(ages)
n_age_brackets = voter_centered["Age_Bracket"].nunique()
n_genders = voter_centered["gender"].nunique()

S = np.zeros((n, n_age_brackets+n_genders))

for i, (a, g) in enumerate(zip(ages, genders)):
    if not np.isnan(a):
        idx = int(a)
        if 0 <= idx < n_age_brackets:
            S[i, idx] = 1
    # male
    if g == 0: S[i, -2] = 1
    # female
    elif g == 1: S[i, -1] = 1

# build the theta_vals matrix by stacking
ordered = lambda_df.sort_values("class_index")["Party_Name"].tolist()
theta_vals = np.vstack([np.concatenate([theta_age[p], theta_gender[p]]) for p in ordered])

# --------------------------------------------------------- Equilibrium conditions Check ---------------------------------------------------------------------------------------

equilibrium_conditions_df = sm.check_equilibrium_conditions(lambda_df=lambda_df, lambda_values=lambda_values, beta=beta,
                                                            voter_centered=voter_centered, x_var=x_var, y_var=y_var)

print("\n===== equilibrium conditions =====\n")
print(equilibrium_conditions_df)

# ----------------------------------------------------- Movement Recommendations for each party ---------------------------------------------------------------------------------------
# Compute matrices C_j for each party and gather eigen‐info
all_party_movements_df = sm.compute_characteristic_matrices(lambda_values=lambda_values,
                                                            beta=beta,
                                                            voter_centered=voter_centered,
                                                            lambda_df=lambda_df,
                                                            x_var=x_var,
                                                            y_var=y_var)

pd.set_option('display.max_columns', None)
print(f"\n===== movement recommendations =====\n")
print(all_party_movements_df.to_string(index=False))

# ------------------------------------------------------ ANALYZE Saddle and Local Min. Points -----------------------------------------------------------------------------------
# sort λ-DF by class_index
lambda_df2 = lambda_df.sort_values("class_index").reset_index(drop=True)
# grab the ordered party names & their indices
party_order = lambda_df2["Party_Name"]
indices = lambda_df2["class_index"].to_numpy(dtype=int)
# re-index by Party_Name
party_centered2 = party_centered.set_index("Party_Name").loc[party_order].reset_index()
all_party_movements2 = all_party_movements_df.drop(columns="class_index").set_index("Party_Name").loc[party_order].reset_index().assign(class_index=indices)
# reorder array of lambda values
lambda_values2 = lambda_df2['valence'].values

# automatically pick out which parties need saddle‐point moves
saddle_targets = all_party_movements2.loc[all_party_movements2["action"].str.contains(
    "Saddle", case=False), "Party_Name"].tolist()
# and which ones need local‐min‐point moves
local_min_targets = all_party_movements2.loc[all_party_movements2["action"].str.contains(
    "local minimum", case=False), "Party_Name"].tolist()

equilibrium_results = []
# --- compute saddle point results ---
for party in saddle_targets:
    v_pos, t_opt, share_opt = sm.compute_optimal_movement_saddle_position(
        lambda_values=lambda_values2,
        lambda_df=lambda_df2,
        voter_centered=voter_centered,
        party_centered=party_centered2,
        beta=beta,
        x_var=x_var,
        y_var=y_var,
        target_party_name=party,
        include_sociodemographic=include_sociodemographic_variables,
        sociodemographic_matrix=S,
        theta_vals=theta_vals,
        alpha=alpha)
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
        lambda_values=lambda_values2,
        lambda_df=lambda_df2,
        voter_centered=voter_centered,
        party_centered=party_centered2,
        target_party_name=party,
        x_var=x_var,
        y_var=y_var,
        beta=beta,
        alpha=alpha,
        include_sociodemographic=include_sociodemographic_variables,
        sociodemographic_matrix=S,
        theta_vals=theta_vals)
    equilibrium_results.append({
        "party":           party,
        "type":            "local_min",
        "optimal_position": z_opt,
        "share_opt":       float(share_opt)})

# put it all in one DataFrame
equilibrium_results_df = pd.DataFrame(equilibrium_results)

print(f"\n===== equilibrium results =====\n")
print(equilibrium_results_df)

# --------------------------------------------------- Plot data cloud and equilibrium positions -----------------------------------------------------------------------------------
fig_1, ax = sm.plot_external_valence_equilibrium(equilibrium_results_df=equilibrium_results_df, voter_centered=voter_centered,
                                                 party_centered=party_centered, x_var=x_var, y_var=y_var, year=year)
plt.show()
