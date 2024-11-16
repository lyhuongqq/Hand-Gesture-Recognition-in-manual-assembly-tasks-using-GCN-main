import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# Define train and validation IDs
train_ids = [ 0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23] #1 21, 51, 71, 91, 101, 111, 131, 211, 221, 231, 61, 121, 201]
val_ids = [6, 12, 20]#2, 22, 52, 72, 92, 102, 112, 132, 212, 222, 232, 62, 122, 202]

curr_dir = r"/root/Hand-Gesture-Recognition-in-manual-assembly-tasks-using-GCN-main/data"

def dfs_from_ids(ids, get_augmented=True):
    """Load data from specified IDs with optional data augmentation."""
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
    """Safely evaluate and convert point data to avoid float errors."""
    if isinstance(point, str):
        try:
            return eval(point)  # Assumes safe context for eval()
        except Exception as e:
            print(f"Error evaluating point {point}: {e}")
            return [0, 0, 0]  # Default fallback value
    elif isinstance(point, (list, tuple)):
        return point  # Return as-is if already a list/tuple
    elif isinstance(point, float):  # Handle unexpected float types
        return [point, point, point]  # Adjust fallback logic as needed
    else:
        return [0, 0, 0]  # Fallback for other unexpected types

# Load train and validation data
train_dfs = dfs_from_ids(train_ids)
val_dfs = dfs_from_ids(val_ids, get_augmented=False)  # No augmentation for validation

# Combine dataframes
train_data = pd.concat(train_dfs, ignore_index=True)
val_data = pd.concat(val_dfs, ignore_index=True)

# Extract features and labels
X_train = train_data.iloc[:, :-1].apply(lambda row: [item for point in row for item in safe_eval(point)], axis=1)
X_train = np.array(X_train.tolist())  # Convert to NumPy array
y_train = train_data['LABEL'].astype('category').cat.codes  # Convert labels to numeric codes

X_val = val_data.iloc[:, :-1].apply(lambda row: [item for point in row for item in safe_eval(point)], axis=1)
X_val = np.array(X_val.tolist())
y_val = val_data['LABEL'].astype('category').cat.codes

# Define and train the SVM model
parameters = [
    {'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]
svm_model = SVC()
model = GridSearchCV(svm_model, parameters, n_jobs=-1, cv=4, verbose=4)

# Fit the model with training data
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model accuracy
accuracy = accuracy_score(y_val, y_pred)
print('Best Parameters:', model.best_params_)
print(f'Accuracy Score on Validation Set: {accuracy * 100:.2f} %')

# Calculate F1-score, Precision, and Recall for each label
report = classification_report(y_val, y_pred, target_names=train_data['LABEL'].astype('category').cat.categories, output_dict=True)

# Convert the report to a DataFrame and save to CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report_full.csv', index=True)

# Alternatively, print precision, recall, and F1-score separately and save to CSV
precision, recall, f1_score, _ = precision_recall_fscore_support(y_val, y_pred, average=None)
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
