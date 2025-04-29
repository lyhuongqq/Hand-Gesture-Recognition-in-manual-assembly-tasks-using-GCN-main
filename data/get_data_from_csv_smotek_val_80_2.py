import pandas as pd
import os
import numpy as np
from imblearn.combine import SMOTETomek
from config import CFG

curr_dir = os.path.dirname(__file__)

train_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23, 64]
val_ids = [6, 12, 20, 66]

def dfs_from_ids(ids, get_augmented=True):
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

def apply_smote_tomek(df):
    print("[DEBUG] Applying SMOTETomek resampling...")
    
    # Extract feature columns and label
    feature_columns = df.columns[:-1]  # All columns except LABEL
    label_column = "LABEL"
    
    # Flatten tuple columns into separate x, y, z columns
    features = []
    for col in feature_columns:
        expanded = df[col].apply(lambda t: np.array(eval(t)) if isinstance(t, str) else t).tolist()
        expanded = np.array(expanded)  # Convert to NumPy array
        features.append(expanded)
    
    # Combine features horizontally
    X = np.hstack(features)  # Shape: (num_samples, num_keypoints * 3)
    y = df[label_column]
    
    print("[DEBUG] Parsed features shape:", X.shape)
    print("[DEBUG] Original label distribution:")
    print(y.value_counts())
    
    # Apply SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    
    print("[DEBUG] Resampled features shape:", X_resampled.shape)
    print("[DEBUG] Resampled label distribution:")
    print(pd.Series(y_resampled).value_counts())
    
    # Convert resampled features back into original tuple format
    num_keypoints = len(feature_columns)  # Number of keypoints
    reshaped_features = X_resampled.reshape(X_resampled.shape[0], num_keypoints, 3)
    
    # Create resampled DataFrame
    resampled_data = {
        col: [tuple(row) for row in reshaped_features[:, i, :]] for i, col in enumerate(feature_columns)
    }
    resampled_df = pd.DataFrame(resampled_data)
    resampled_df[label_column] = y_resampled  # Add resampled labels
    
    return resampled_df


def get_train_data():
    train_dfs = dfs_from_ids(train_ids)
    df_train = pd.concat(train_dfs)
    print("[DEBUG] Combined train data shape:", df_train.shape)
    
    # Resample with SMOTETomek
    df_train_resampled = apply_smote_tomek(df_train)
    print("[DEBUG] Resampled train data shape:", df_train_resampled.shape)

    return df_train_resampled

#def get_val_data():
#    val_dfs = dfs_from_ids(val_ids)
#    df_val = pd.concat(val_dfs)
#    print("[DEBUG] Combined validation data shape:", df_val.shape)

    # Validation data is not resampled
#    return df_val

def get_val_data():
    val_dfs = dfs_from_ids(val_ids)
    df_val = pd.concat(val_dfs)
    print("[DEBUG] Combined val data shape:", df_val.shape)
        # Resample with SMOTETomek
    df_val_resampled = apply_smote_tomek(df_val)
    print("[DEBUG] Resampled val data shape:", df_val_resampled.shape)

    return df_val_resampled

if __name__ == "__main__":
    print("[INFO] Loading and processing training data...")
    train_data = get_train_data()
    print("[DEBUG] Train data preview:")
    print(train_data.head())

    print("[INFO] Loading and processing validation data...")
    val_data = get_val_data()
    print("[DEBUG] Validation data preview:")
    print(val_data.head())
