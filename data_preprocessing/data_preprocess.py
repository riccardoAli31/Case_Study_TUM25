import pandas as pd 
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import data_preprocessing.data_loading as dl
CONFIG_PATH = "data_preprocessing/configs.json"


def aggregate_who_voted_for(voter_df):
    """In the survey they asked multiple non-overlapping times what party the person voted for,
    once for normal voter, then what the <18 age voter would have voted and in 2009 they also asked
    the people who did not vote

    Parameters
    ----------
    voter_df : pd.Dataframe
        survey results from gesis, with NaN decoded as 0

    Returns
    -------
    pd.Dataframe
        where we only have two columns which represent the first and second vote 
    """
    if ("first vote (too young)" in voter_df) and ("second vote (too young)" in voter_df):
        voter_df["who did you vote for:first vote"] += voter_df["first vote (too young)"]
        voter_df["who did you vote for:second vote"] += voter_df["second vote (too young)"]
        voter_df.drop(["first vote (too young)", "second vote (too young)"], axis=1, inplace=True)

    if ("first vote (did not vote)" in voter_df) and ("second vote (did not vote)" in voter_df):
        voter_df["who did you vote for:first vote"] += voter_df["first vote (did not vote)"]
        voter_df["who did you vote for:second vote"] += voter_df["second vote (did not vote)"]
        voter_df.drop(["first vote (did not vote)", "second vote (did not vote)"], axis=1, inplace=True)

    return voter_df


def get_raw_party_voter_data(x_var: str, y_var: str, year: str, file_dir: str = 'data_folder', country: str = 'Germany') -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- Load and filter party data ---
    party_df = dl.get_party_positions_data(file_dir=file_dir, country=country)
    years_available = sorted(party_df['Year'].unique())
    if year not in years_available:
        fallback = years_available[-1]
        warnings.warn(f"Requested year {year!r} not found in party data; "f"falling back to most recent year {fallback!r}.", UserWarning)
        year = fallback
    party_year_filtered = party_df[party_df['Year'] == year].reset_index(drop=True)

    # --- Load voter data ---
    voter_df = dl.get_gesis_data(path=file_dir, year=year)

    # only keep common variables
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
    columns_to_keep = ["bundesland", "who did you vote for:second vote", "do you incline towards a party, if so which one", 
                    "how strongly do you incline towards this party"] + sorted(policy_columns)
    voter_df = voter_df[voter_df.columns.intersection(columns_to_keep)].copy()

    # Create aggregated features for the voters'data 
    # Since for one variable of the party's data, we have a few corresponding ones from the voters'data, we'll have to aggregate the voter's feature into one
    # --- Aggregate voter features for each dimension ---
    for dim in (x_var, y_var):
        vars_to_agg = filtered_mapping[dim]
        if len(vars_to_agg) == 0:
            raise RuntimeError(f"For the variable {dim} there are no features to aggregate")
        weights = np.ones(len(vars_to_agg)) / len(vars_to_agg)
        voter_df[dim] = voter_df[vars_to_agg].dot(weights)

    voter_agg = voter_df[[x_var, y_var, 'who did you vote for:second vote', 'do you incline towards a party, if so which one', 
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
    voter_agg['Party_Name'] = (voter_agg['who did you vote for:second vote'].map(code2party))
    # drop everything else (801, -98, etc.)
    voter_agg = voter_agg.dropna(subset=['Party_Name']).reset_index(drop=True)

    # --- build integer choice 0..p-1 in the same order as party_scaled ---
    party_order = list(voter_agg['Party_Name'].unique())
    name2idx = {name:i for i,name in enumerate(party_order)}
    voter_agg['party_choice'] = voter_agg['Party_Name'].map(name2idx).astype(int)

    return party_year_filtered, voter_agg


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


def get_scaled_party_voter_data(x_var: str, y_var: str, year: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    party_year_filtered, voter_agg = get_raw_party_voter_data(x_var=x_var, y_var=y_var, year=year)

    # --- Standardize all clouds ---
    vot_pts = voter_agg[[x_var, y_var]].copy()
    party_pts = party_year_filtered[[x_var, y_var]].copy()

    # Scale voter data independently
    voter_scaler = StandardScaler()
    voter_scaler.fit(vot_pts)
    v_scaled = voter_scaler.transform(vot_pts)

    # Scale party data independently
    party_scaler = StandardScaler()
    party_scaler.fit(party_pts)
    p_scaled = party_scaler.transform(party_pts)

    voter_df_scaled = voter_agg.copy()
    party_df_scaled = party_year_filtered.copy()
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


def get_valence_from_gesis(politicians: dict, year: str) -> pd.DataFrame:
    """Extract the valences of the parties given in *politicians* as df

    Parameters
    ----------
    politicians : dict
        has the form {name1: party1, name2: party2,...}, e.g. {"soeder": "CSU", "habeck": "GREENS"}
    year: str
        the year for which to get the gesis data and the political leaders of that time
        should be in ["2009", "2013", "2017", "2021"]

    Returns
    -------
    pd.DataFrame
        df with columns for name of politician, party of politician and the valence

    Raises
    ------
    KeyError
        if not all politicians were found in gesis data
    """
    df = dl.get_gesis_data(year=year, fill=False)

    politicians_names = tuple(politicians.keys())
    # filter df to only get the opinion columns, which always have the form
    # "opinion on:name", e.g. "opinion on:soeder"
    cols = [col for col in df.columns if col.endswith(politicians_names)]
    df = df[cols].copy()

    # scale whole dataframe with one standartscaler, so the whole frame has mean=0 and std=1
    mean = np.nanmean(df.values)
    std = np.nanstd(df.values)
    df = (df - mean) / std

    # aggregate over all voters to get one valence value
    df = df.aggregate("mean", axis=0).to_frame("valence")

    # create two new columns for the name of the politicians and the party
    df["politician"] = df.index
    df["politician"] = df["politician"].apply(lambda s: s.split(":")[-1])
    df["Party_Name"] = df["politician"].apply(lambda s: politicians[s])
    df.index = np.arange(len(df))
    
    # if there are multiple top candiates for one party, we take the mean
    df = df.groupby("Party_Name", as_index=False).agg({"politician": ' '.join, "valence": "mean"})

    return df
