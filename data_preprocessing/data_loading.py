import os
import json
import pandas as pd 
import numpy as np
CONFIG_PATH = "data_preprocessing/configs.json"
VOTER_DATA_FILE_NAME = "voters_2021.sav"


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
    # Filter by country 
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
    # Preprocessing dataframe - datatypes and awkward column names
    df_filtered["Calendar_Week"] = df_filtered["Calendar_Week"].astype(str)
    df_filtered['Year'] = (pd.to_datetime(df_filtered['Date'], dayfirst=True).dt.year).astype(str)
    df_filtered = df_filtered.loc[:, ~df_filtered.columns.str.startswith("+ per505")]
    common_items_mapping = load_common_variables_mapping(CONFIG_PATH)
    base_columns = ["Country", "Year", "Date", "Calendar_Week", "Party_Name"]
    policy_columns = list(common_items_mapping.keys())
    df_filtered = df_filtered[base_columns + policy_columns]
    return df_filtered


def load_gesis_mapping(fp: str=CONFIG_PATH, file_name=VOTER_DATA_FILE_NAME) -> dict:
    """Load the mapping for column names of gesis data
    """
    with open(fp, "r") as f:
        cfg = json.load(f)
    year = file_name[-8:-4]
    try:
        return cfg[f"voter_positions_{year}_mapping"]
    except KeyError:
        raise KeyError(f"'voter_positions_{year}_mapping' not found in {fp}")
    

def load_party_leaders(fp: str=CONFIG_PATH, year: str = None) -> dict:
    """Load the party leaders of that year to calculate valences
    """
    with open(fp, "r") as f:
        cfg = json.load(f)
    try:
        return cfg["party_leaders"][year]
    except KeyError:
        raise KeyError(f"no party leaders for {year} found in {fp}")
    

def get_gesis_data(path: str="data_folder", lower_cutoff: int=-70, upper_cutoff: int=800, year: str = None, fill: bool=True) -> pd.DataFrame:
    """Load the gesis dataset, which represents voter positions, into a dataframe and does some preprocessing 
    Parameters
    ----------
    path : str
        path to the file
    lower_cutoff : int, optional
        all items with value below this are replaced with NaN, by default -70
        no answer is often encoded with -71
    upper_cutoff : int, optional
        all items with value above this are replaced with NaN, by default 800
        no answer is often encoded with -71
    year : str, optional
        the year for which data is to be loaded, by default None
        should be in ["2009", "2013", "2017", "2021"]
    fill : bool, optional
        if np.nan values should be filled with median for numerics and 0 for other columns, by default True

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
    if year is not None:
        pref = os.path.join(path, f"voters_{year}.sav")               # first choice
        alt  = os.path.join(path, f"voter_dataset_{year}.sav")        # second choice
        if os.path.isfile(pref):
            sav_path = pref
        elif os.path.isfile(alt):
            sav_path = alt
        else:
            # fallback: any file in the folder containing the year
            candidates = [fn for fn in os.listdir(path) if fn.endswith(".sav") and year in fn]
            if len(candidates) == 1:
                sav_path = os.path.join(path, candidates[0])
            elif len(candidates) > 1:
                raise FileNotFoundError(f"Multiple GESIS files match year={year!r}: {candidates}")
            else:
                raise FileNotFoundError(f"No GESIS file found for year={year!r}")
    else:
        sav_path = os.path.join(path, VOTER_DATA_FILE_NAME)
    if not os.path.isfile(sav_path):
        raise FileNotFoundError(f"File not found: {sav_path!r}")
    
    df = pd.read_spss(path=sav_path, convert_categoricals=False)

    # rename columns
    mapping = load_gesis_mapping(CONFIG_PATH, os.path.basename(sav_path))
    df.rename(mapping, inplace=True, axis=1)

    # drop all unneeded columns
    cols = list(mapping.values())
    df.drop(df.columns.difference(cols), axis=1, inplace=True)

    # replace all values below/above cutoff with NaN (e.g. encoding for "no answer given")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].mask(df[num_cols] < lower_cutoff)
    df[num_cols] = df[num_cols].mask(df[num_cols] > upper_cutoff)
    if fill:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df = df.fillna(0)  
    
    return df

