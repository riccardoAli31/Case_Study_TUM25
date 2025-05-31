import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import data_preprocessing.data_loading as dl
CONFIG_PATH = "data_preprocessing/configs.json"


def get_raw_party_voter_data(x_var: str, y_var: str, file_dir: str = 'data_folder', country: str = 'Germany') -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- Load and filter party data ---
    party_df = dl.get_party_positions_data(file_dir=file_dir, country=country)
    party_week_filtered = party_df[party_df['Calendar_Week'] == party_df['Calendar_Week'].max()].reset_index(drop=True)

    # --- Load voter data ---
    voter_df, _ = dl.get_gesis_data(path=file_dir)

    # # only keep common variables
    common_items_mapping = dl.load_common_variables_mapping(CONFIG_PATH)

    # --- Filter that mapping to what actually exists in this voter_df ---
    filtered_mapping = {
        dim: [col for col in cols if col in voter_df.columns]
        for dim, cols in common_items_mapping.items()}

    # Make sure our two chosen policy dims survived the filter
    for dim in (x_var, y_var):
        if not filtered_mapping.get(dim):
            raise KeyError(
                f'None of the columns for policy‐dimension "{dim}" '
                "were found in your voter data")

    # --- Subset voter_df to only the surviving common items (plus your fixed ones) ---
    policy_columns = set().union(*filtered_mapping.values())
    columns_to_keep = ["bundesland", "who did you vote for:second vote(a)", "do you incline towards a party, if so which one(a)", 
                    "how strongly do you incline towards this party"] + sorted(policy_columns)
    voter_df = voter_df.loc[:, voter_df.columns.intersection(columns_to_keep)]

    # Create aggregated features for the voters'data 
    # Since for one variable of the party's data, we have a few corresponding ones from the voters'data, we'll have to aggregate the voter's feature into one
    # ---> Average of the variables for each policy dimension
    def feature_aggregation(row, features):
        vals = row[features].astype(float).values
        if vals.size == 0:
            return np.nan
        sorted_vals = np.sort(vals)[::-1]
        # build an array of 1/n for however many items there are
        weights = np.ones(sorted_vals.size) / sorted_vals.size
        return np.dot(sorted_vals, weights)

    # --- Aggregate voter features for each dimension ---
    for dim in (x_var, y_var):
        vars_to_agg = filtered_mapping[dim]
        voter_df[dim] = voter_df.apply(lambda row: feature_aggregation(row, vars_to_agg), axis=1)

    voter_agg = voter_df[[x_var, y_var, 'who did you vote for:second vote(a)', 'do you incline towards a party, if so which one(a)', 
                        'how strongly do you incline towards this party']].copy()

    # --- map voter codes → Party_Name (must match party_scaled['Party_Name']) ---
    code2party = {
        4:   "SPD",
        1:   "CDU/CSU",
        6:   "90/Greens",
        5:   "FDP",
        322: "AfD",
        7:   "LINKE",
        392: "SSW"
    }
    voter_agg['Party_Name'] = (voter_agg['who did you vote for:second vote(a)'].map(code2party))
    # drop everything else (801, -98, etc.)
    voter_agg = voter_agg.dropna(subset=['Party_Name']).reset_index(drop=True)

    # --- build integer choice 0..p-1 in the same order as party_scaled ---
    party_order = list(voter_agg['Party_Name'].unique())
    name2idx = {name:i for i,name in enumerate(party_order)}
    voter_agg['party_choice'] = voter_agg['Party_Name'].map(name2idx).astype(int)

    return party_week_filtered, voter_agg


def party_position_weighted(df, x_var, y_var, weight_manifesto=0.2, weight_voters_mean=0.8):
    # Build the expected column names for x_var
    x_scaled_col = f"{x_var} Scaled"
    x_mean_col   = f"{x_var} Voters_Mean"
    x_comb_col   = f"{x_var} Combined"
    # Build the expected column names for y_var
    y_scaled_col = f"{y_var} Scaled"
    y_mean_col   = f"{y_var} Voters_Mean"
    y_comb_col   = f"{y_var} Combined"
    # Check that the required columns exist
    for col in (x_scaled_col, x_mean_col, y_scaled_col, y_mean_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")
    # Compute the “combined” columns
    df[x_comb_col] = weight_manifesto * df[x_scaled_col] + weight_voters_mean * df[x_mean_col]
    df[y_comb_col] = weight_manifesto * df[y_scaled_col] + weight_voters_mean * df[y_mean_col]

    return df


def get_scaled_party_voter_data(x_var: str, y_var: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    party_week_filtered, voter_agg = get_raw_party_voter_data(x_var=x_var, y_var=y_var)

    # --- Standardize all clouds ---
    vot_pts = voter_agg[[x_var, y_var]].copy()
    party_pts = party_week_filtered[[x_var, y_var]].copy()

    # Scale voter data independently
    voter_scaler = StandardScaler()
    voter_scaler.fit(vot_pts)
    v_scaled = voter_scaler.transform(vot_pts)

    # Scale party data independently
    party_scaler = StandardScaler()
    party_scaler.fit(party_pts)
    p_scaled = party_scaler.transform(party_pts)

    voter_df_scaled = voter_agg.copy()
    party_df_scaled = party_week_filtered.copy()
    voter_df_scaled[[f"{x_var} Scaled", f"{y_var} Scaled"]] = v_scaled
    party_df_scaled[[f"{x_var} Scaled", f"{y_var} Scaled"]] = p_scaled

    # --- Compute voter means for party positions for each scaled dimension ---
    party_means = (voter_df_scaled.groupby('Party_Name')[[f"{x_var} Scaled", f"{y_var} Scaled"]].mean().reset_index()
                            .rename(columns={f"{x_var} Scaled": f'{x_var} Voters_Mean', f"{y_var} Scaled": f'{y_var} Voters_Mean'}))
    mean_df = party_means[[f'{x_var} Voters_Mean', f'{y_var} Voters_Mean', 'Party_Name']].copy()

    # --- Merge scaled manifesto data with voters mean positions ---
    party_df_scaled = party_df_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', x_var, y_var, f"{x_var} Scaled", f"{y_var} Scaled"]]
    party_df_scaled = party_df_scaled.merge(mean_df, on='Party_Name')

    # --- Final party posiitons: 0.2*manifesto + 0.8*voters mean ---
    party_df_scaled = party_position_weighted(df=party_df_scaled, x_var=x_var, y_var=y_var)

    party_df_scaled['Label'] = party_df_scaled['Party_Name']
    voter_df_scaled['Label'] = 'Voter'

    return party_df_scaled, voter_df_scaled


def center_party_voter_data(voter_df, party_df, x_var, y_var):
    # Build the exact column names
    x_scaled_col = f"{x_var} Scaled"
    y_scaled_col = f"{y_var} Scaled"
    x_centered_col = f"{x_var} Centered"
    y_centered_col = f"{y_var} Centered"

    # Sanity check: make sure the scaled‐columns exist in both DataFrames
    for col in (x_scaled_col, y_scaled_col):
        if col not in voter_df.columns:
            raise KeyError(f"Column '{col}' not found in voter_df.")
        if col not in party_df.columns:
            raise KeyError(f"Column '{col}' not found in party_df.")

    # 1) Compute the voter‐means for x_scaled_col and y_scaled_col
    mean_series = voter_df[[x_scaled_col, y_scaled_col]].astype(float).mean()

    # 2) Subtract the voter‐means and store results in two new columns in voter_df
    voter_df[x_centered_col] = voter_df[x_scaled_col].astype(float) - mean_series[x_scaled_col]
    voter_df[y_centered_col] = voter_df[y_scaled_col].astype(float) - mean_series[y_scaled_col]

    # 3) Subtract the same voter‐means in party_df
    party_df[x_centered_col] = party_df[x_scaled_col].astype(float) - mean_series[x_scaled_col]
    party_df[y_centered_col] = party_df[y_scaled_col].astype(float) - mean_series[y_scaled_col]

    return party_df, voter_df
