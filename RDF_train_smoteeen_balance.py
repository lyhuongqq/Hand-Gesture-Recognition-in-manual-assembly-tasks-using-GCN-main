from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np

# Define train and validation IDs
train_ids = [0, 2, 5, 7, 9, 10, 11, 13, 21, 22, 23]
val_ids = [6, 12, 20]

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

# Print class distribution before resampling
print("Training set class distribution before SMOTEENN:")
print(pd.Series(y_train).value_counts())

print("\nValidation set class distribution before SMOTEENN:")
print(pd.Series(y_val).value_counts())

# Apply SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
X_val_resampled, y_val_resampled = smote_enn.fit_resample(X_val, y_val)

# Print class distribution after SMOTEENN
print("\nTraining set class distribution after SMOTEENN:")
print(pd.Series(y_train_resampled).value_counts())

print("\nValidation set class distribution after SMOTEENN:")
print(pd.Series(y_val_resampled).value_counts())

# Function to balance classes after SMOTEENN using SMOTE
def balance_with_smote(X, y, target_size):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = X, y
    for class_label in np.unique(y):
        class_count = np.sum(y_balanced == class_label)
        if class_count < target_size:
            X_res, y_res = smote.fit_resample(X_balanced, y_balanced)
            X_balanced, y_balanced = X_res, y_res
    return X_balanced, y_balanced

# Find the maximum class size after SMOTEENN
max_train_size = max(pd.Series(y_train_resampled).value_counts())
max_val_size = max(pd.Series(y_val_resampled).value_counts())

# Balance training and validation sets further using SMOTE
X_train_balanced, y_train_balanced = balance_with_smote(X_train_resampled, y_train_resampled, max_train_size)
X_val_balanced, y_val_balanced = balance_with_smote(X_val_resampled, y_val_resampled, max_val_size)

# Shuffle the datasets to ensure randomness
X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=42)
X_val_balanced, y_val_balanced = shuffle(X_val_balanced, y_val_balanced, random_state=42)

# Print the new class distributions
print("\nTraining set class distribution after balancing with SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

print("\nValidation set class distribution after balancing with SMOTE:")
print(pd.Series(y_val_balanced).value_counts())

# Define and train the Random Forest model
parameters = {
    'n_estimators': [10, 100, 1000],
    'max_depth': [3, 6, 9],
    'max_features': ['sqrt', 'log2', None]
}
rf_model = RandomForestClassifier(random_state=42)
model = GridSearchCV(rf_model, parameters, n_jobs=-1, cv=4, scoring='accuracy', verbose=4)

model.fit(X_train_balanced, y_train_balanced)

# Predict on the resampled validation set
y_pred = model.predict(X_val_balanced)

# Evaluate the model
accuracy = accuracy_score(y_val_balanced, y_pred)
print('\nBest Parameters:', model.best_params_)
print(f'Accuracy Score on Resampled Validation Set: {accuracy * 100:.2f} %')

# Create a summary of metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(y_val_balanced, y_pred, average=None)
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
results_df.to_csv('random_forest_run_results_een_balanced.csv', index=False)

# Save detailed classification report to CSV
report = classification_report(y_val_balanced, y_pred, target_names=train_data['LABEL'].astype('category').cat.categories, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('RDF_classification_report_een_balanced.csv', index=True)
