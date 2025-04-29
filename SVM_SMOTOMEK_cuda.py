import os
import pandas as pd
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from imblearn.combine import SMOTETomek  # Import SMOTE-Tomek

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Define train and validation IDs
train_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23]
val_ids = [6, 12, 20]

curr_dir = r"/teamspace/studios/this_studio/Hand-Gesture-Recognition-in-manual-assembly-tasks-using-GCN-main/data"

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

# Load train and validation data
train_dfs = dfs_from_ids(train_ids)
val_dfs = dfs_from_ids(val_ids, get_augmented=False)

# Combine dataframes
train_data = pd.concat(train_dfs, ignore_index=True)
val_data = pd.concat(val_dfs, ignore_index=True)

# Extract features and labels
X_train = train_data.iloc[:, :-1].apply(lambda row: [item for point in row for item in safe_eval(point)], axis=1)
X_train = np.array(X_train.tolist())
y_train = train_data['LABEL'].astype('category').cat.codes

X_val = val_data.iloc[:, :-1].apply(lambda row: [item for point in row for item in safe_eval(point)], axis=1)
X_val = np.array(X_val.tolist())
y_val = val_data['LABEL'].astype('category').cat.codes

# Apply SMOTE-Tomek resampling to training and validation sets
print("Applying SMOTE-Tomek resampling...")
smote_tomek = SMOTETomek(random_state=42)

# Resample training set
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Resample validation set
X_val_resampled, y_val_resampled = smote_tomek.fit_resample(X_val, y_val)

# Move data to GPU if available
X_train_resampled = torch.tensor(X_train_resampled, device=device, dtype=torch.float32)
y_train_resampled = torch.tensor(y_train_resampled, device=device, dtype=torch.long)
X_val_resampled = torch.tensor(X_val_resampled, device=device, dtype=torch.float32)
y_val_resampled = torch.tensor(y_val_resampled, device=device, dtype=torch.long)

# Define and train the SVM model
parameters = [
    {'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]
svm_model = SVC()
model = GridSearchCV(svm_model, parameters, n_jobs=-1, cv=4, verbose=4)

# Convert data back to CPU for SVM compatibility
model.fit(X_train_resampled.cpu().numpy(), y_train_resampled.cpu().numpy())

# Predict on the resampled validation set
y_pred = model.predict(X_val_resampled.cpu().numpy())

# Evaluate the model accuracy
accuracy = accuracy_score(y_val_resampled.cpu().numpy(), y_pred)
print('Best Parameters:', model.best_params_)
print(f'Accuracy Score on Validation Set: {accuracy * 100:.2f} %')

# Save accuracy and best parameters to a CSV file
accuracy_results = pd.DataFrame({
    "Metric": ["Accuracy"],
    "Value": [accuracy * 100]
})
accuracy_results.to_csv('accuracy_results.csv', index=False)

# Save best parameters to a separate CSV
best_params = pd.DataFrame([model.best_params_])
best_params.to_csv('best_parameters.csv', index=False)

# Calculate and save detailed classification report
report = classification_report(y_val_resampled.cpu().numpy(), y_pred, target_names=train_data['LABEL'].astype('category').cat.categories, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report_full.csv', index=True)

# Calculate F1-score, Precision, and Recall for each label
precision, recall, f1_score, _ = precision_recall_fscore_support(y_val_resampled.cpu().numpy(), y_pred, average=None)
results = []
for idx, label in enumerate(train_data['LABEL'].astype('category').cat.categories):
    results.append({
        "Label": label,
        "Precision": precision[idx],
        "Recall": recall[idx],
        "F1-Score": f1_score[idx]
    })

# Save detailed results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('detailed_classification_metrics_full.csv', index=False)
