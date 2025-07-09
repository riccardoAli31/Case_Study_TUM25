import pandas as pd
import numpy as np
from typing import Union, Dict


def get_age(voter_df: pd.DataFrame, year_of_survey: int) -> pd.DataFrame:
    """Calculates the age from the year of birth of each voter.
    Does not take into account the day and month of birth.

    Parameters
    ----------
    voter_df : pd.DataFrame
        survey data, where each row is one person
    year_of_survey : int
        year when the survey was done, needed to see how old people were at that time

    Returns
    -------
    pd.DataFrame
        voter_df but now with added column "age", which represents the age of the voter in years

    Raises
    ------
    KeyError
        if no column for the year of birth is found in voter_df
    """
    if "year of birth" not in voter_df:
        raise KeyError("'year of birth' not found in the dataframe.")

    def calculate_age(year): return year_of_survey - year
    voter_df["age"] = voter_df["year of birth"].apply(calculate_age)
    voter_df.drop("year of birth", axis=1, inplace=True)
    return voter_df


def get_age_bracket(voter_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns each voter to an age bracket based on the voters age

    Parameters
    ----------
    voter_df : pd.DataFrame
        survey data, where each row is one person

    Returns
    -------
    pd.DataFrame
        voter_df but now with added column "bracket", which represents the bracket the voter has been assigned
    """
    brackets = {(18, 25): 0, (26, 35): 1, (36, 45): 2,
                (46, 55): 3, (56, 65): 4, (66, 200): 5}

    # finds the bracket for each voter
    def find_bracket(age):
        for (start, end), bracket in brackets.items():
            if start <= age <= end:
                return bracket

    voter_df["Age_Bracket"] = voter_df["age"].apply(find_bracket)
    # voter_df.drop("age", axis=1, inplace=True)
    return voter_df


def add_age_col_to_voters(voter_df, year):
    try:
        year = int(year)
    except ValueError:
        raise ValueError(f"'{year}' can't be converted to an integer")
    df = voter_df.copy()
    df["year of birth"] = pd.to_numeric(df["year of birth"], errors="coerce")
    df.dropna(subset=["year of birth"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- compute ages & brackets ---
    df = get_age(df, year)
    df = get_age_bracket(df)   # adds "bracket" column ∈ {0,…,5}
    return df


def get_age_effect(df: pd.DataFrame,
                   translated: bool = False) -> Union[Dict[int, np.ndarray], pd.DataFrame]:
    """
    Compute age‐bracket vote shares by party.

    Parameters
    ----------
    voter_df : pd.DataFrame
      Must contain columns "year of birth" and "second vote".
    year : int or numeric‐string
      The survey year (for age computation).
    translated : bool, default False
      If False, returns a dict mapping party‐codes to arrays of bracket‐shares.
      If True, returns a DataFrame with columns ["party","bracket","share"].
    """
    # --- translation maps ---
    code2party = {
        4:  "SPD",
        1:  "CDU/CSU",
        6:  "90/Greens",
        5:  "FDP",
        322: "AfD",
        7:  "LINKE",
    }
    bracket_labels = {
        0: "18–25",
        1: "26–35",
        2: "36–45",
        3: "46–55",
        4: "56–65",
        5: "66–100",
    }

    # --- raw theta dict: party‐code → array of bracket‐shares ---
    theta: Dict[int, np.ndarray] = {
        party: (
            df.loc[df["second vote"] == party, "Age_Bracket"]
              .value_counts(normalize=True)
              .sort_index()
              .to_numpy()
        )
        for party in df["second vote"].unique()
    }
    theta_named = {code2party.get(
        code, code): shares for code, shares in theta.items()}

    if not translated:
        return theta_named

    # --- build a DataFrame from the theta dict ---
    # rows = party codes, cols = bracket‐ids
    df_theta = (
        pd.DataFrame.from_dict(theta, orient="index")
          .rename_axis("party_code", axis=0)
          .rename_axis("bracket_id", axis=1)
          .reset_index()
    )
    # map codes → names & bracket_ids → bracket_labels
    df_theta["party"] = df_theta["party_code"].map(code2party)
    df_theta = df_theta.rename(columns=bracket_labels)
    # melt into long form
    df_long = df_theta.melt(
        id_vars=["party"],
        value_vars=list(bracket_labels.values()),
        var_name="Age_Bracket",
        value_name="share"
    )
    # drop NaN parties if any codes were unexpected
    df_long = df_long.dropna(subset=["party"]).reset_index(drop=True)
    return df_long


def get_gender_effect(df: pd.DataFrame) -> dict:
    code2party = {
        4:  "SPD",
        1:  "CDU/CSU",
        6:  "90/Greens",
        5:  "FDP",
        322: "AfD",
        7:  "LINKE",
    }
    # dict with party codes as keys and shares of male/female voters as values
    # first entry in array is male share, second is female share
    theta = {
        party: (
            df.loc[df["second vote"] == party, "gender"]
            .value_counts(normalize=True)
            .sort_index()
            .to_numpy()
        )
        for party in df["second vote"].unique()
    }

    theta_named = {code2party.get(
        code, code): shares for code, shares in theta.items()}

    return theta_named
