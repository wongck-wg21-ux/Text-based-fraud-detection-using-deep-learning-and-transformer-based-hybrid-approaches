import pandas as pd
import os

def load_csv(file_path, file_type, file_name):
    if file_type == "subject_body":
        # Datasets that split content into subject + body columns
        df = pd.read_csv(file_path)
        df["text"] = df["subject"].fillna('') + " " + df["body"].fillna('')
        df["source"] = file_name  # Add source tracking
        return df[["text", "label", "source"]]

    elif file_type == "text_combined":
        # Datasets that already have a single 'text_combined' column
        df = pd.read_csv(file_path)
        df = df.rename(columns={"text_combined": "text"})
        df["source"] = file_name
        return df[["text", "label", "source"]]

    elif file_type == "spam_csv":
        # SMS Spam dataset (UCI SMS Spam Collection format)
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df = df.rename(columns={"v1": "label", "v2": "text"})
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        df["source"] = file_name
        return df[["text", "label", "source"]]

    elif file_type == "no_header":
        # Datasets without header row (label,text format)
        df = pd.read_csv(file_path, names=["label", "text"], on_bad_lines='skip')
        df["label"] = df["label"].map({"fraud": 1, "ham": 0})
        df["source"] = file_name
        return df[["text", "label", "source"]]

    else:
        raise ValueError("Unknown file type.")

def merge_datasets(dataset_list, input_dir, output_path):
    all_dfs = []
    # Load each dataset with provenance tracking
    for file_name, file_type in dataset_list:
        path = os.path.join(input_dir, file_name)
        df = load_csv(path, file_type, file_name)  # Pass file_name to track source
        all_dfs.append(df)

    # Concatenate into one unified DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Dataset-level filtering (no text cleaning here)
    combined_df.dropna(subset=["text", "label"], inplace=True)     # remove missing values
    combined_df.drop_duplicates(subset="text", inplace=True)       # remove duplicate messages
    combined_df = combined_df[combined_df["text"].str.len() > 10]  # filter out very short/noisy texts
    combined_df["label"] = combined_df["label"].astype(int)        # ensure integer labels

    # Save merged dataset to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Dataset merged and filtered: {len(combined_df)} records saved.")
    return combined_df
