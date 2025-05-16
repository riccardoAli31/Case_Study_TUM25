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


def get_gesis_data(path: str, names: dict={}, cutoff: int = -70) -> tuple[pd.DataFrame, pd.Series]:
    """Load the gesis dataset, which represents voter positions, into a dataframe and does some preprocessing 

    Parameters
    ----------
    path : str
        path to the file
    cutoff : int, optional
        all items with value below this are replaced with NaN, by default -70

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
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found at: {path}")

    df = pd.read_spss(path=path, convert_categoricals=False)

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

    # rename columns
    df.rename(names, inplace=True, axis=1)

    # how often each answer was given
    count = df.value_counts()

    return df, count

