import pandas as pd  
import os
import smote_variants as sv
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np

# Configurations
curr_dir = os.path.dirname(__file__)

from config import CFG

# Map string labels to numeric indices dynamically
label_mapping = {cls_name: i for i, cls_name in enumerate(CFG.classes)}

# Full video IDs
all_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23, 6, 12, 20]


def dfs_from_ids(ids, get_augmented=True):
    """Load data from CSV files based on IDs and apply augmentation if needed."""
    dfs = []
    for i in ids:
        df = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_original.csv"), index_col=0)
        if get_augmented:
            df_f0 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-vert.csv"), index_col=0)
            df_f1 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-hor.csv"), index_col=0)
            df_f2 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-hor-vert.csv"), index_col=0)
            dfs.extend([df, df_f0, df_f1, df_f2])
        else:
            dfs.append(df)
    return dfs
import ast  # For safely converting string tuples to lis
import numpy as np

import numpy as np
import ast

def preprocess_features(df):
    """Convert string tuples to numeric columns before resampling."""
    numeric_df = pd.DataFrame()

    for col in df.columns:
        if col != "LABEL":
            # Convert string tuples to lists of floats
            xyz_cols = df[col].apply(lambda x: list(map(float, ast.literal_eval(x))) if isinstance(x, str) else x)
            xyz_df = pd.DataFrame(xyz_cols.tolist(), columns=[f"{col}_x", f"{col}_y", f"{col}_z"])
            numeric_df = pd.concat([numeric_df, xyz_df], axis=1)

    numeric_df["LABEL"] = df["LABEL"]

    # Debugging: Print sample values to check conversion
    print("[DEBUG] Converted feature values before resampling:\n", numeric_df.head())

    return numeric_df


def reconstruct_original_format(df_original, df_resampled):
    """Reconstruct the original format after SMOTE resampling."""
    reconstructed_df = pd.DataFrame()
    for col in df_original.columns:
        if col != "LABEL":
            x = df_resampled[f"{col}_x"]
            y = df_resampled[f"{col}_y"]
            z = df_resampled[f"{col}_z"]
            reconstructed_df[col] = x.astype(str) + ", " + y.astype(str) + ", " + z.astype(str)
        else:
            reconstructed_df[col] = df_resampled["LABEL"]
    return reconstructed_df

def apply_dbm_smote(X_train, y_train, X_val, y_val, alpha=0.5):
    """
    Apply SMOTE and Random_SMOTE separately and then concatenate results for DBM resampling.
    """
    # Step 1: Apply SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_smote1, y_smote1 = smote.fit_resample(X_train, y_train)
    X_smote2, y_smote2 = smote.fit_resample(X_val, y_val)

    # Step 2: Apply Random_SMOTE
    random_smote = sv.Random_SMOTE(proportion=alpha, random_state=42)
    X_random_smote1, y_random_smote1 = random_smote.sample(X_train, y_train)
    X_random_smote2, y_random_smote2 = random_smote.sample(X_val, y_val)

    # Step 3: Concatenate original, SMOTE, and Random_SMOTE generated data
    X_train = np.vstack((X_train, X_smote1, X_random_smote1))
    y_train = np.concatenate((y_train, y_smote1, y_random_smote1))
    X_val = np.vstack((X_val, X_smote2, X_random_smote2))
    y_val = np.concatenate((y_val, y_smote2, y_random_smote2))

    print("[INFO] Class distribution after DBM resampling (Training):")
    print(pd.Series(y_train).value_counts())
    print("[INFO] Class distribution after DBM resampling (Validation):")
    print(pd.Series(y_val).value_counts())

    return X_train, y_train, X_val, y_val

def get_train_val_data(all_ids, train_ratio=0.8):
    """Load all data, split into train/validation sets, and apply resampling using DBM."""
    dfs = dfs_from_ids(all_ids, get_augmented=True)
    df_all = pd.concat(dfs)
    df_all.reset_index(drop=True, inplace=True)

    print("[INFO] Combined LABEL distribution before splitting:")
    print(df_all["LABEL"].value_counts())

    # Apply preprocessing
    df_all = preprocess_features(df_all)

    df_train, df_val = train_test_split(
        df_all, test_size=1-train_ratio, random_state=42, stratify=df_all["LABEL"]
    )

    # Convert to NumPy and Ensure Data is Float
    X_train = df_train.drop(columns=["LABEL"]).to_numpy(dtype=np.float32)
    y_train = df_train["LABEL"].to_numpy()
    X_val = df_val.drop(columns=["LABEL"]).to_numpy(dtype=np.float32)
    y_val = df_val["LABEL"].to_numpy()

    print("[INFO] Training LABEL distribution before resampling:")
    print(pd.Series(y_train).value_counts())
    print("[INFO] Validation LABEL distribution before resampling:")
    print(pd.Series(y_val).value_counts())

    X_train_resampled, y_train_resampled, X_val_resampled, y_val_resampled = apply_dbm_smote(X_train, y_train, X_val, y_val)

    return X_train_resampled, y_train_resampled, X_val_resampled, y_val_resampled


if __name__ == "__main__":
    train_ratio = 0.8
    df_train, df_val = get_train_val_data(all_ids, train_ratio=train_ratio)

    print("[INFO] Final Training Dataset Size:", len(df_train))
    print("[INFO] Training Dataset Class Distribution:")
    print(df_train["LABEL"].value_counts())

    print("[INFO] Final Validation Dataset Size:", len(df_val))
    print("[INFO] Validation Dataset Class Distribution:")
    print(df_val["LABEL"].value_counts())