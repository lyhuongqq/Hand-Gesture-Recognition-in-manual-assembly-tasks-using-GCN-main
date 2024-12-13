import pandas as pd
import os
from imblearn.combine import SMOTETomek
import json
from collections import Counter
from sklearn.model_selection import train_test_split

# Configurations
curr_dir = os.path.dirname(__file__)

# Label mapping for readability
label_mapping = {
    0: "Grasp",
    1: "Move",
    2: "Negative",
    3: "Position",
    4: "Reach",
    5: "Release",
}

# Hand landmarks dictionary for renaming columns
hand_landmarks_dict = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}

# Edge index for JSON output
edge_index = [
    [0, 1], [0, 5], [0, 9], [0, 17],
    [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20],
]

# Full video IDs
all_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23, 6, 12, 20, 66]  # Replace with the total number of video IDs available

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

def preprocess_features(df):
    """Convert string tuples to numeric columns."""
    numeric_df = pd.DataFrame()
    for col in df.columns:
        if col != "LABEL":
            xyz_cols = df[col].str.strip("()").str.split(", ", expand=True).astype(float)
            xyz_cols.columns = [f"{col}_x", f"{col}_y", f"{col}_z"]
            numeric_df = pd.concat([numeric_df, xyz_cols], axis=1)
    numeric_df["LABEL"] = df["LABEL"]
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

def apply_multiclass_smote_tomek(df):
    """Apply SMOTETomek to balance all classes in the dataset."""
    df_numeric = preprocess_features(df)

    # Debug: Check class distribution before resampling
    print("[DEBUG] Class distribution before resampling:")
    print(df_numeric["LABEL"].value_counts())

    X = df_numeric.drop(columns=["LABEL"])
    y = df_numeric["LABEL"]

    smote_tomek = SMOTETomek(random_state=42)
    try:
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    except Exception as e:
        print(f"[ERROR] Resampling failed: {e}")
        return df

    # Debug: Check class distribution after resampling
    print("[DEBUG] Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts())

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["LABEL"] = y_resampled
    return reconstruct_original_format(df, df_resampled)

def get_train_val_data(all_ids, train_ratio=0.8):
    """Load all data, apply SMOTETomek resampling, and split into train/validation sets."""
    # Load data from all IDs
    dfs = dfs_from_ids(all_ids, get_augmented=True)
    df_all = pd.concat(dfs)
    df_all.reset_index(drop=True, inplace=True)

    print("[INFO] Combined LABEL distribution before resampling:")
    print(df_all["LABEL"].value_counts())

    # Apply SMOTETomek to balance the dataset
    df_balanced = apply_multiclass_smote_tomek(df_all)

    print("[INFO] Combined LABEL distribution after resampling:")
    print(df_balanced["LABEL"].value_counts())

    # Split into training and validation
    df_train, df_val = train_test_split(
        df_balanced, test_size=1-train_ratio, random_state=42, stratify=df_balanced["LABEL"]
    )

    print("[INFO] Training LABEL distribution before balancing:")
    print(df_train["LABEL"].value_counts())

    print("[INFO] Validation LABEL distribution before balancing:")
    print(df_val["LABEL"].value_counts())

    # Apply SMOTETomek to validation data to balance it
    df_val_balanced = apply_multiclass_smote_tomek(df_val)

    print("[INFO] Validation LABEL distribution after balancing:")
    print(df_val_balanced["LABEL"].value_counts())

    return df_train, df_val_balanced

def get_train_data():
    """Load and resample training data."""
    train_dfs = dfs_from_ids(all_ids)
    df_train = pd.concat(train_dfs)
    print("[DEBUG] Combined train data shape:", df_train.shape)

    # Resample with SMOTETomek
    df_train_resampled = apply_multiclass_smote_tomek(df_train)
    print("[DEBUG] Resampled train data shape:", df_train_resampled.shape)

    return df_train_resampled

def get_val_data():
    """Load validation data without resampling."""
    val_dfs = dfs_from_ids(all_ids)
    df_val = pd.concat(val_dfs)
    print("[DEBUG] Combined validation data shape:", df_val.shape)

    return df_val

if __name__ == "__main__":
    train_ratio = 0.8

    # Get training and validation data
    df_train, df_val = get_train_val_data(all_ids, train_ratio=train_ratio)

    # Export or proceed with the next steps
    print("[INFO] Training data preview:")
    print(df_train.head())

    print("[INFO] Validation data preview:")
    print(df_val.head())
