import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import data_preprocessing.data_loading as dl
CONFIG_PATH = "data_preprocessing/configs.json"


def get_scaled_party_voter_data(x_var: str, y_var: str, file_dir: str = 'data_folder', country: str = 'Germany', mapping_path: str = CONFIG_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load party positions and voter survey data, aggregate and scale both
    onto the same [0,10] range for two chosen policy dimensions.

    Parameters
    ----------
    x_var : str
        Name of the first policy dimension (e.g. "Planned Economy").
    y_var : str
        Name of the second policy dimension (e.g. "Environmental Protection").
    file_dir : str
        Path to folder containing the party CSV and GESIS data.
    country : str, default 'Germany'
        Country filter for party positions.
    mapping_path : str, default dl.CONFIG_PATH
        Path to the common-variables mapping JSON.

    Returns
    -------
    party_df_scaled : pd.DataFrame
    voter_df_scaled : pd.DataFrame
    """

    # --- Load and filter party data ---
    party_df = dl.get_party_positions_data(file_dir=file_dir, country=country)
    party_week_filtered = party_df[party_df['Calendar_Week'] == party_df['Calendar_Week'].max()].reset_index(drop=True)
    
    # --- Load voter data ---
    voter_list = dl.get_gesis_data(path=file_dir)
    voter_df = voter_list[0].copy()

    # --- Load common variables ---
    mapping = dl.load_common_variables_mapping(path=mapping_path)

    # Create aggregated features for the voters'data 
    # Since for one variable of the party's data, we have a few corresponding ones from the voters'data, we'll have to aggregate the voter's feature into one
    # ---> Weighted Average of the variables for each policy dimension
    weights = np.array([0.5, 0.3, 0.2])
    def feature_aggregation(row, features):
        vals = row[features].values
        sorted_vals = np.sort(vals)[::-1]
        return np.dot(sorted_vals, weights)
    # --- Aggregate voter features for each dimension ---
    for dim in (x_var, y_var):
        vars_to_agg = mapping[dim]
        voter_df[dim] = voter_df.apply(lambda r: feature_aggregation(r, vars_to_agg), axis=1)
    voter_agg = voter_df[[x_var, y_var, 'who did you vote for:second vote(a)']].copy()

    # The party data for the chosen variables are in the range [0, 10] for both dimensions
    # Scale the voters data to be in the same range as the parties
    # Fit the scaler on voters data alone 
    scaler = MinMaxScaler(feature_range=(0, 10))
    # Transform voters and parties separately
    scaled_voters = scaler.fit_transform(voter_agg[[x_var, y_var]])
    scaled_parties = scaler.fit_transform(party_week_filtered[[x_var, y_var]])
    # Assign scaled values back
    voter_df_scaled = voter_agg.copy()
    voter_df_scaled[[f"{x_var} Scaled", f"{y_var} Scaled"]] = scaled_voters
    party_df_scaled = party_week_filtered.copy()
    party_df_scaled[[f"{x_var} Scaled", f"{y_var} Scaled"]] = scaled_parties

    party_df_scaled = party_df_scaled[['Country', 'Date', 'Calendar_Week', 'Party_Name', x_var, y_var, f"{x_var} Scaled", f"{y_var} Scaled"]]
    voter_df_scaled["Label"] = "Voter"
    party_df_scaled["Label"] = party_df_scaled["Party_Name"]

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
    voter_df_scaled['Party_Name'] = (voter_df_scaled['who did you vote for:second vote(a)'].map(code2party))
    # drop everything else (801, -98, etc.)
    voter_df_scaled = voter_df_scaled.dropna(subset=['Party_Name']).reset_index(drop=True)
    
    # --- build integer choice 0..p-1 in the same order as party_scaled ---
    party_order = list(party_df_scaled['Party_Name'])
    name2idx = {name:i for i,name in enumerate(party_order)}
    voter_df_scaled['party_choice'] = voter_df_scaled['Party_Name'].map(name2idx).astype(int)

    return party_df_scaled, voter_df_scaled


def center_rotate_data_cloud(party_df: pd.DataFrame, voter_df: pd.DataFrame, x_var: str, y_var: str, pc1_name: str = "PC1", pc2_name: str = "PC2") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Center both clouds by the MEAN of the voters (on the scaled x,y),
    rotate into the voters' principal-axis frame, and store PC1/PC2
    back into each DataFrame.

    Returns only the two modified DataFrames.
    """
    # pick off the scaled columns
    cols = [f"{x_var} Scaled", f"{y_var} Scaled"]
    Y = voter_df[cols].to_numpy()
    Z = party_df[cols].to_numpy()

    # 1) translate by voter mean
    mean_v = Y.mean(axis=0)
    Yc = Y - mean_v
    Zc = Z - mean_v

    # 2) PCA on voter cloud
    cov      = np.cov(Yc, rowvar=False)
    eigv, eigvecs = np.linalg.eigh(cov)
    order    = np.argsort(eigv)[::-1]
    Q        = eigvecs[:,order]

    # 3) rotate
    Y_rot = Yc.dot(Q)
    Z_rot = Zc.dot(Q)

    # 4) write PCs back into copies of the DataFrames
    v2 = voter_df.copy()
    p2 = party_df.copy()
    v2[pc1_name], v2[pc2_name] = Y_rot[:,0], Y_rot[:,1]
    p2[pc1_name], p2[pc2_name] = Z_rot[:,0], Z_rot[:,1]

    return p2, v2

