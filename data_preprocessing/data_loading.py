import os
import json
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
CONFIG_PATH = "data_preprocessing/configs.json"


def load_common_variables_mapping(path: str) -> dict:
    """Load the mapping for common variables between the two datasets
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    try:
        return cfg["common_items"]
    except KeyError:
        raise KeyError(f"'common_items_mapping' not found in {path}")
    

def load_party_manifesto_mapping(config_path: str) -> dict:
    """
    Pull the `manifesto_mapping` dict from the JSON config.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mapping = cfg.get("party_manifesto_mapping", {})
    if not mapping:
        raise KeyError("'party_manifesto_mapping' not found (or empty) in config.json")
    return mapping


def get_party_positions_data(file_dir, country, file_name='party_dataset.csv') -> pd.DataFrame:
    """
    Load the party positions dataset and return rows that match a given country
    """
    csv_path = os.path.join(file_dir, file_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter by country and date
    df_filtered = df[df["countryname"] == country]

    # Columns that aren’t needed for analysis
    cols_to_drop = ["country", "oecdmember", "eumember", "party", "partyname", "parfam",
        "candidatename", "coderid", "manual", "coderyear", "id_perm", "testresult",
        "testeditsim", "pervote", "voteest", "presvote", "absseat", "totseats", "progtype", 
        "datasetorigin", "corpusversion", "total", "peruncod", "datasetversion"]
    
    df_filtered = df_filtered.drop(columns=cols_to_drop).reset_index(drop=True)
    
    # Rename Manifesto variables 
    mapping = load_party_manifesto_mapping(CONFIG_PATH)
    df_filtered.rename(columns=mapping, inplace=True)

    # Preprocessing dataframe - NaN values of numerical columns 
    df_filtered = df_filtered.apply(pd.to_numeric, errors="ignore")
    num_cols = df_filtered.select_dtypes(include="number").columns
    for col in num_cols:
        df_filtered[col] = df_filtered[col].fillna(0)

    # Preprocessing dataframe - 0 valued columns
    zero_share = (df_filtered[num_cols] == 0).sum() / len(df_filtered)  
    zero_cols_to_drop = zero_share[zero_share >= 0.80].index.tolist()
    df_filtered.drop(columns=zero_cols_to_drop, inplace=True)

    # Preprocessing dataframe - datatypes and awkward column names
    df_filtered["Calendar_Week"] = df_filtered["Calendar_Week"].astype(str)
    df_filtered = df_filtered.loc[:, ~df_filtered.columns.str.startswith("+ per505")]

    common_items_mapping = load_common_variables_mapping(CONFIG_PATH)
    base_columns = ["Country", "Date", "Calendar_Week", "Party_Name"]
    policy_columns = list(common_items_mapping.keys())
    df_filtered = df_filtered[base_columns + policy_columns]

    return df_filtered


def load_gesis_mapping(fp: str) -> dict:
    """Load the mapping for column names of gesis data
    """
    with open(fp, "r") as f:
        cfg = json.load(f)

    try:
        return cfg["voter_positions_mapping"]
    except KeyError:
        raise KeyError(f"'voter_positions_mapping' not found in {fp}")


def get_gesis_data(path: str, cutoff: int = -70, file_name='voter_dataset.sav') -> tuple[pd.DataFrame, pd.Series]:
    """Load the gesis dataset, which represents voter positions, into a dataframe and does some preprocessing 

    Parameters
    ----------
    path : str
        path to the file
    cutoff : int, optional
        all items with value below this are replaced with NaN, by default -70
        no answer is often encoded with -71

    Returns
    -------
    pd.DataFrame
        dataset loaded into a dataframe
    pd.Series
        how often each unique answer was given

    Raises
    ------
    FileNotFoundError
        file at *path* not found
    """
    sav_path = os.path.join(path, file_name)

    if not os.path.isfile(sav_path):
        raise FileNotFoundError(f"CSV file not found at: {sav_path}")

    df = pd.read_spss(path=sav_path, convert_categoricals=False)

    # drop all unfinished surveys
    df = df.drop(df[df["kp27_dispcode"] == 22].index).reset_index(drop=True)

    # drop all unneeded columns
    cols_to_drop = ["study", "version", "doi", "field_start", "field_end", "sample", "lfdn", "kp27_dispcode", 
                    "kp27_intstatus", "kp27_modus", "kp27_device", "kp27_smartphone",
                    "kp27_tablet", "kp27_speederindex", "kp27_lastpage", "kp27_datetime", "kp27_date_of_last_access",
                    "kp27_850a", "kp27_850b", "kp27_870a", "kp27_870b", "kp27_4380", "kp27_4390aa", "kp27_4390ab", 
                    "kp27_4390ba", "kp27_4390bb", "kp27_4480", "kp27_4380", "kp27_4490aa", "kp27_4490ab", 
                    "kp27_4490ba", "kp27_4490bb", "kp27_4580", "kp27_4590aa", "kp27_4590ab", 
                    "kp27_4590ba", "kp27_4590bb"]
    
    df = df.drop(columns=cols_to_drop).reset_index(drop=True)

    # replace all values below cutoff with NaN (e.g. encoding for "no answer given")
    df = df[df >= cutoff]
    df = df.fillna(0)

    # rename columns
    mapping = load_gesis_mapping(CONFIG_PATH)
    df.rename(mapping, inplace=True, axis=1)

    common_items_mapping = load_common_variables_mapping(CONFIG_PATH)
    policy_columns = set(col for cols in common_items_mapping.values() for col in cols)
    columns_to_keep = ["bundesland"] + list(policy_columns)
    df = df[columns_to_keep]

    # how often each answer was given
    count = df.value_counts()

    return df, count


def get_scaled_party_voter_data(x_var: str, y_var: str, file_dir: str = 'data_folder', country: str = 'Germany', mapping_path: str = CONFIG_PATH) -> (pd.DataFrame, pd.DataFrame):
    
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
    party_df = get_party_positions_data(file_dir=file_dir, country=country)
    party_week_filtered = party_df[party_df['Calendar_Week'] == party_df['Calendar_Week'].max()].reset_index(drop=True)
    
    # --- Load voter data ---
    voter_list = get_gesis_data(path=file_dir)
    voter_df = voter_list[0].copy()

    # --- Load common variables ---
    mapping = load_common_variables_mapping(path=mapping_path)

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
    voter_agg = voter_df[[x_var, y_var]].copy()

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

    return party_df_scaled, voter_df_scaled