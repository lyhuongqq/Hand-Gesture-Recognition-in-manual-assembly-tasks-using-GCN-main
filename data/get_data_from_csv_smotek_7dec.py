import pandas as pd
import os
from imblearn.over_sampling import SMOTE
import json
from config import CFG

# Configurations
curr_dir = os.path.dirname(__file__)

# IDs for train and validation
train_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23]  # Train IDs
val_ids = [6, 12, 20]  # Validation IDs

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
            # Convert '(x, y, z)' into three separate numeric columns
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

from imblearn.over_sampling import SMOTE
import pandas as pd

from imblearn.combine import SMOTETomek
from collections import Counter

def apply_multiclass_smote_tomek(df):
    """Apply SMOTETomek to balance all classes in the dataset."""
    # Preprocess features (convert to numeric)
    df_numeric = preprocess_features(df)
    
    # Separate features (X) and labels (y)
    X = df_numeric.drop(columns=["LABEL"])
    y = df_numeric["LABEL"]
    
    # Print original dataset distribution
    print(f"[INFO] Original dataset shape: {Counter(y)}")
    
    # Apply SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    
    # Print resampled dataset distribution
    print(f"[INFO] Resampled dataset shape: {Counter(y_resampled)}")
    
    # Combine resampled features and labels into a DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["LABEL"] = y_resampled

    # Reconstruct original format
    df_reconstructed = reconstruct_original_format(df, df_resampled)
    return df_reconstructed



def print_distribution(df, title):
    """Print class distribution with human-readable labels."""
    class_counts = df["LABEL"].map(label_mapping).value_counts()
    print(f"[INFO] {title} DATA DISTRIBUTION")
    print(class_counts)
    print(f"[INFO] {title.upper()} ON {len(df)} DATAPOINTS")


def export_to_json(df, filename="hand_data.json"):
    """Export the DataFrame to JSON with renamed columns and edge index."""
    df.rename(columns=hand_landmarks_dict, inplace=True)
    df_json = df.to_dict(orient="records")
    data = {"data": df_json, "edges": edge_index}
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] JSON exported to {filename}")


def get_train_data():
    """Prepare training data and balance it using SMOTETomek."""
    train_dfs = dfs_from_ids(train_ids)
    df_train = pd.concat(train_dfs)
    df_train.reset_index(drop=True, inplace=True)
    df_train = df_train.replace("Postion", "Position")
    df_train["LABEL"] = df_train["LABEL"].astype('category').cat.codes
    
    # Print initial distribution
    print_distribution(df_train, "INITIAL TRAIN")
    
    # Apply SMOTETomek to balance the dataset
    df_train_balanced = apply_multiclass_smote_tomek(df_train)
    print_distribution(df_train_balanced, "BALANCED TRAIN")
    
    # Export to JSON
    export_to_json(df_train_balanced, filename="train_data.json")
    return df_train_balanced


def get_val_data():
    """Prepare validation data."""
    # Load validation data
    val_dfs = dfs_from_ids(val_ids)
    df_val = pd.concat(val_dfs)
    df_val.reset_index(drop=True, inplace=True)
    df_val = df_val.replace("Postion", "Position")  # Fix label typo

    # Map string labels to numeric indices
    label_mapping = {cls_name: i for i, cls_name in enumerate(CFG.classes)}
    df_val["LABEL"] = df_val["LABEL"].map(label_mapping)

    # Check for missing or incorrect mappings
    if df_val["LABEL"].isnull().any():
        print("[ERROR] Missing or incorrect labels in validation data:")
        print(df_val[df_val["LABEL"].isnull()])
        df_val = df_val.dropna(subset=["LABEL"])  # Drop rows with missing labels

    # Log validation distribution
    print_distribution(df_val, "VALIDATION")
    return df_val

if __name__ == "__main__":
    # Prepare training data
    df_train = get_train_data()

    # Log distribution after balancing
    print_distribution(df_train, "FINAL TRAIN")

    # Prepare validation data
    df_val = get_val_data()

    # Log validation distribution
    print_distribution(df_val, "FINAL VALIDATION")

