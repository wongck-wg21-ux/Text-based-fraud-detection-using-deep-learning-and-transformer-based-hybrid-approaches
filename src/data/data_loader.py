import pandas as pd

def load_and_label(paths):
    df_list = []
    for fp, src in paths:
        # Read each CSV into a DataFrame
        df = pd.read_csv(fp)

        # Track data provenance by adding 'source' column
        df["source"] = src
        df_list.append(df)

    # Combine all DataFrames into one unified table
    return pd.concat(df_list, ignore_index=True)