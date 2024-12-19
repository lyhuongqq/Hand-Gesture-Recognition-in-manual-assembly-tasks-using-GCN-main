import pandas as pd
import os
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from config import CFG

# Configurations
curr_dir = os.path.dirname(__file__)

# Map string labels to numeric indices dynamically
label_mapping = {cls_name: i for i, cls_name in enumerate(CFG.classes)}

# Full video IDs
all_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23, 6, 12, 20]

# Load and preprocess features
def preprocess_features(df):
    """Convert string tuples to numeric columns."""
    numeric_df = pd.DataFrame()
    for col in df.columns:
        if col != "LABEL":
            xyz_cols = df[col].str.strip("()").str.split(", ", expand=True).astype(float)
            xyz_cols.columns = [f"{col}_x", f"{col}_y", f"{col}_z"]
            numeric_df = pd.concat([numeric_df, xyz_cols], axis=1)
    numeric_df["LABEL"] = df["LABEL"].map(label_mapping)
    return numeric_df

def reconstruct_original_format(df_original, df_resampled):
    """Reconstruct the original format after resampling."""
    reconstructed_df = pd.DataFrame()
    for col in df_original.columns:
        if col != "LABEL":
            x = df_resampled[f"{col}_x"]
            y = df_resampled[f"{col}_y"]
            z = df_resampled[f"{col}_z"]
            reconstructed_df[col] = x.astype(str) + ", " + y.astype(str) + ", " + z.astype(str)
        else:
            reconstructed_df[col] = df_resampled["LABEL"].map(
                {v: k for k, v in label_mapping.items()}
            )
    return reconstructed_df

def apply_hybrid_resampling(df, target_size=None):
    """
    Apply Borderline-SMOTE + ENN to balance all classes in the dataset.
    """
    df_numeric = preprocess_features(df)

    print("[DEBUG] Class distribution before resampling:")
    print(df_numeric["LABEL"].value_counts())

    X = df_numeric.drop(columns=["LABEL"])
    y = df_numeric["LABEL"]

    # Determine the target size
    if target_size is None:
        target_size = y.value_counts().max()

    # Set the sampling strategy
    sampling_strategy = {cls: target_size for cls in y.unique()}

    # Create a hybrid pipeline
    hybrid_pipeline = Pipeline([
        ("oversample", BorderlineSMOTE(random_state=42, sampling_strategy=sampling_strategy)),
        ("clean", EditedNearestNeighbours())
    ])

    try:
        X_resampled, y_resampled = hybrid_pipeline.fit_resample(X, y)
    except Exception as e:
        print(f"[ERROR] Resampling failed: {e}")
        return df

    print("[DEBUG] Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts())

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["LABEL"] = y_resampled
    return reconstruct_original_format(df, df_resampled)

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

def get_train_val_data(all_ids, train_ratio=0.8):
    """Load all data, split into train/validation sets, and apply resampling to both sets."""
    # Load data from all IDs
    dfs = dfs_from_ids(all_ids, get_augmented=True)
    df_all = pd.concat(dfs)
    df_all.reset_index(drop=True, inplace=True)

    print("[INFO] Combined LABEL distribution before splitting:")
    print(df_all["LABEL"].value_counts())

    # Split into training and validation sets before resampling
    df_train, df_val = train_test_split(
        df_all, test_size=1-train_ratio, random_state=42, stratify=df_all["LABEL"]
    )

    print("[INFO] Training LABEL distribution before resampling:")
    print(df_train["LABEL"].value_counts())

    print("[INFO] Validation LABEL distribution before resampling:")
    print(df_val["LABEL"].value_counts())

    # Resample the training set
    print("[INFO] Applying hybrid resampling to the training set...")
    train_majority_size = df_train["LABEL"].value_counts().max()
    df_train_resampled = apply_hybrid_resampling(df_train, target_size=train_majority_size)

    # Resample the validation set
    print("[INFO] Applying hybrid resampling to the validation set...")
    val_majority_size = df_val["LABEL"].value_counts().max()
    df_val_resampled = apply_hybrid_resampling(df_val, target_size=val_majority_size)

    return df_train_resampled, df_val_resampled

if __name__ == "__main__":
    # Example usage
    all_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23, 6, 12, 20]
    train_ratio = 0.8

    # Get training and validation data
    df_train, df_val = get_train_val_data(all_ids, train_ratio=train_ratio)

    # Export or proceed with the next steps
    print("[INFO] Final Training Dataset Size:", len(df_train))
    print("[INFO] Final Validation Dataset Size:", len(df_val))
    print("[INFO] Training Dataset Class Distribution:")
    print(df_train["LABEL"].value_counts())
    print("[INFO] Validation Dataset Class Distribution:")
    print(df_val["LABEL"].value_counts())
