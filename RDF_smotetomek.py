from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import os
import pandas as pd
import numpy as np
import torch  # For device handling (CUDA/CPU)

# Check for device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define train and validation IDs
train_ids = [1, 21, 51, 71, 91, 101, 111, 131, 211, 221, 231, 61, 121, 201]
val_ids = [2, 22, 52, 72, 92, 102, 112, 132, 212, 222, 232, 62, 122, 202]

# Define the current directory path or input directory
curr_dir = '/teamspace/studios/this_studio/Hand-Gesture-Recognition-in-manual-assembly-tasks-using-GCN-main/data'

# Function to load data
def dfs_from_ids(ids, get_augmented=True):
    dfs = []
    for i in ids:
        try:
            df = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_original.csv"), index_col=0)
            if get_augmented:
                df_f0 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-vert.csv"), index_col=0)
                df_f1 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-hor.csv"), index_col=0)
                df_f2 = pd.read_csv(os.path.join(curr_dir, f"graphdata/{i}_mdc04_mtc05_Train_flip-hor-vert.csv"), index_col=0)
                dfs.extend([df, df_f0, df_f1, df_f2])
            else:
                dfs.append(df)
        except FileNotFoundError as e:
            print(f"File not found for ID {i}: {e}")
        except Exception as e:
            print(f"Error loading data for ID {i}: {e}")
    return dfs

# Load and combine train and validation data
train_dfs = dfs_from_ids(train_ids)
val_dfs = dfs_from_ids(val_ids, get_augmented=True)

train_data = pd.concat(train_dfs, ignore_index=True)
val_data = pd.concat(val_dfs, ignore_index=True)

# Extract features and labels
def safe_eval(point):
    if isinstance(point, str):
        try:
            return eval(point)
        except Exception as e:
            print(f"Error evaluating point {point}: {e}")
            return [0, 0, 0]
    elif isinstance(point, (list, tuple)):
        return point
    elif isinstance(point, float):
        return [point, point, point]
    else:
        return [0, 0, 0]

X_train = train_data.iloc[:, :-1].apply(lambda row: [item for point in row for item in safe_eval(point)], axis=1)
X_train = np.array(X_train.tolist())
y_train = train_data['LABEL'].astype('category').cat.codes

X_val = val_data.iloc[:, :-1].apply(lambda row: [item for point in row for item in safe_eval(point)], axis=1)
X_val = np.array(X_val.tolist())
y_val = val_data['LABEL'].astype('category').cat.codes

# Move data to device if CUDA is available
if device.type == "cuda":
    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.long)
    X_val = torch.tensor(X_val, device=device, dtype=torch.float32)
    y_val = torch.tensor(y_val, device=device, dtype=torch.long)

# Apply SMOTETomek for resampling
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(
    X_train.cpu().numpy() if device.type == "cuda" else X_train,
    y_train.cpu().numpy() if device.type == "cuda" else y_train
)

# Move resampled data back to the device if needed
if device.type == "cuda":
    X_train_resampled = torch.tensor(X_train_resampled, device=device, dtype=torch.float32)
    y_train_resampled = torch.tensor(y_train_resampled, device=device, dtype=torch.long)

# Define and train the Random Forest model
parameters = {
    'n_estimators': [10, 100, 1000],
    'max_depth': [3, 6, 9],
    'max_features': ['sqrt', 'log2', None]
}
rf_model = RandomForestClassifier(random_state=42)
model = GridSearchCV(rf_model, parameters, n_jobs=-1, cv=4, scoring='accuracy', verbose=4)

# Fit the model
model.fit(X_train_resampled.cpu().numpy() if device.type == "cuda" else X_train_resampled,
          y_train_resampled.cpu().numpy() if device.type == "cuda" else y_train_resampled)

# Predict on the validation set
y_pred = model.predict(X_val.cpu().numpy() if device.type == "cuda" else X_val)

# Evaluate the model
accuracy = accuracy_score(y_val.cpu().numpy() if device.type == "cuda" else y_val, y_pred)
print('Best Parameters:', model.best_params_)
print(f'Accuracy Score on Validation Set: {accuracy * 100:.2f} %')

# Create a summary of metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_val.cpu().numpy() if device.type == "cuda" else y_val, y_pred, average=None
)
results = []
for idx, label in enumerate(train_data['LABEL'].astype('category').cat.categories):
    results.append({
        "Label": label,
        "Precision": precision[idx],
        "Recall": recall[idx],
        "F1-Score": f1_score[idx],
        "Best Parameters": model.best_params_,
        "Accuracy": accuracy * 100
    })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df['Run'] = 1
results_df.to_csv('random_forest_run_results_tomek.csv', index=False)

# Save detailed classification report to CSV
report = classification_report(
    y_val.cpu().numpy() if device.type == "cuda" else y_val, y_pred,
    target_names=train_data['LABEL'].astype('category').cat.categories, output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report_detailed_tomek.csv', index=True)
