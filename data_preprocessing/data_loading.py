import os
import json
import pandas as pd 
CONFIG_PATH = "data_preprocessing/configs.json"


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


def get_party_positions_data(file_dir, file_name, country) -> pd.DataFrame:
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

    return df_filtered


def get_gesis_data(path: str) -> pd.DataFrame:
    """Load the gesis dataset, which represents voter positions, into a dataframe

    Parameters
    ----------
    path : str
        path to the file

    Returns
    -------
    pd.DataFrame
        dataset loaded into a dataframe

    Raises
    ------
    FileNotFoundError
        file at *path* not found
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found at: {path}")

    df = pd.read_spss(path=path, convert_categoricals=False)

    return df

