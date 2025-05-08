import os
import pandas as pd


def get_party_positions_data(file_dir, file_name, country: str) -> pd.DataFrame:
    """
    Load the party positions dataset and return rows that match a given *country* and *date_code* (e.g. 202109).

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Full path to the MPDS CSV file.
    country : str
        Country name as it appears in the `countryname` column.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe
    """
    csv_path = os.path.join(file_dir, file_name)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter by country and date
    df_filtered = df[df["countryname"] == country]

    # Columns that aren’t needed for most analyses
    cols_to_drop = ["country", "oecdmember", "eumember",
        "party", "partyname", "parfam",
        "candidatename", "coderid", "manual",
        "coderyear", "id_perm", "testresult",
        "testeditsim", "pervote", "voteest",
        "presvote", "absseat", "totseats",
        "progtype", "datasetorigin", "corpusversion",
        "total", "peruncod"]
    
    df_filtered = df_filtered.drop(columns=cols_to_drop).reset_index(drop=True)

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
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found at: {path}")

    df = pd.read_spss(path=path, convert_categoricals=False)

    return df

