import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC

# Loading the dataset
# Experiment#1
# data_file = 'training_dataset_per_second.csv'
# test_data_file = 'test-dataset_per_second_70-30.csv'

# Experiment#2
# data_file = 'training_dataset_per_second.csv'
# test_data_file = 'test-dataset_per_second_93-7.csv'

# Experiment#3
# data_file = 'training_dataset_200k.csv'
# test_data_file = 'test-dataset_70-30.csv'

# Experiment#4
data_file = 'training_dataset_200k.csv'
test_data_file = 'test-dataset_93-7.csv'

data = pd.read_csv(data_file)
data_test = pd.read_csv(test_data_file)

# Fill missing values with 'unknown'
data['File Extension'] = data['File Extension'].fillna('cmd')


# Data Pre-Processing
# Experiment to check and consolidate the data per 5 seconds. Not included for Project Experiments
def calculate_requests_every_5s(df):
    # Convert 'Timestamp' to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sort by 'Client ID' and 'Timestamp'
    df = df.sort_values(by=['Client ID', 'Timestamp'])

    # Initialize the new column with zeros
    df['Requests Every 5s'] = 0

    # Group by 'Client ID'
    grouped = df.groupby('Client ID')

    for name, group in grouped:
        # Calculate the number of requests every 5 seconds for each client
        count_5s = group['Timestamp'].diff().dt.total_seconds().fillna(0).cumsum() // 5
        df.loc[group.index, 'Requests Every 5s'] = group.groupby(count_5s).cumcount() + 1

    return df


# Experiment to check and consolidate the data per micro seconds. Not included for Project Experiments
def calculate_requests_and_volume_per_microsecond(df):
    # Convert 'Timestamp' to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sort by 'Client ID' and 'Timestamp'
    df = df.sort_values(by=['Client ID', 'Timestamp'])

    # Round down the timestamp to the nearest microsecond
    # df['Microsecond'] = df['Timestamp'].dt.floor('U')
    df['Microsecond'] = df['Timestamp'].dt.floor('L')  # milliseconds instead of micro.

    # Calculate the number of requests per microsecond for each Client ID
    df['Requests Per Microsecond'] = df.groupby(['Client ID', 'Microsecond'])['Microsecond'].transform('count')

    # Calculate the sum of Data Transfer Volume per microsecond for each Client ID
    df['Volume Per Microsecond'] = df.groupby(['Client ID', 'Microsecond'])['Data Transfer Volume'].transform('sum')

    # Keep only the last record for each microsecond for each Client ID
    df = df.groupby(['Client ID', 'Microsecond']).tail(1)

    # Drop the 'Microsecond' column as it's no longer needed
    df = df.drop(columns=['Microsecond'])

    return df


# Experiment to consolidate the data per second. Finalized for Project Experiments
def calculate_requests_and_volume_per_second(df):
    # Convert 'Timestamp' to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sort by 'Client ID' and 'Timestamp'
    df = df.sort_values(by=['Client ID', 'Timestamp'])

    # Round down the timestamp to the nearest second
    df['Second'] = df['Timestamp'].dt.floor('S')

    # Calculate the number of requests per second for each Client ID
    df['Requests Per Second'] = df.groupby(['Client ID', 'Second'])['Second'].transform('count')

    # Calculate the sum of Data Transfer Volume per second for each Client ID
    df['Volume Per Second'] = df.groupby(['Client ID', 'Second'])['Data Transfer Volume'].transform('sum')

    # Keep only the last record for each second for each Client ID
    df = df.groupby(['Client ID', 'Second']).tail(1)

    # Drop the 'Second' column as it's no longer needed
    df = df.drop(columns=['Second'])

    return df


# data = calculate_requests_and_volume_per_second(data)
# data_test = calculate_requests_and_volume_per_second(data_test)

# Save the updated dataset
# data.to_csv('training_dataset_per_second.csv', index=False)
# data_test.to_csv('test-dataset_per_second_70-30.csv', index=False)

# Display the first few rows of the updated dataset
# print(data.head())
# print(data_test.head())

# Display the last few rows of the updated dataset
# print(data.tail())
# print(data_test.tail())

print(f'Training Dataset Name is: {data_file}')
print(f'Test Dataset Name is: {test_data_file}')
print(f'Number of rows: {len(data)}')
print(f'Number of test rows: {len(data_test)}')

# Check the distribution of the target variable
print(f"Data Intruder count: {data['Intruder Flag'].value_counts()}")
print(f"Test Data Intruder count: {data_test['Intruder Flag'].value_counts()}")

# Feature selection and target variable
features = data[[
    'Client ID',
    'Message Type',
    'Message Size',
    'File Extension',
    'File Size',
    'Connection Duration',
    'Number of Messages',
    'Number of Files',
    'Client IP',
    'Data Transfer Volume',
    'Request Frequency',
    'Error Rate',
    # 'Requests Every 5s',
    # 'Requests Per Second',
    # 'Volume Per Second',
    # 'Requests Per Microsecond',
    # 'Volume Per Microsecond',
]]
target = data['Intruder Flag']

features_test = data_test[[
    'Client ID',
    'Message Type',
    'Message Size',
    'File Extension',
    'File Size',
    'Connection Duration',
    'Number of Messages',
    'Number of Files',
    'Client IP',
    'Data Transfer Volume',
    'Request Frequency',
    'Error Rate',
    # 'Requests Every 5s',
    # 'Requests Per Second',
    # 'Volume Per Second',
    # 'Requests Per Microsecond',
    # 'Volume Per Microsecond',
]]
target_test = data_test['Intruder Flag']

# Handling missing values by imputing with mean for numerical features
numeric_features = [
    # 'Message Type',
    'Message Size',
    'File Size',
    'Connection Duration',
    # 'Number of Messages',
    # 'Number of Files',
    'Data Transfer Volume',
    # 'Request Frequency',
    # 'Error Rate',
    # 'Requests Every 5s',
    # 'Requests Per Second',
    # 'Volume Per Second',
    # 'Requests Per Microsecond',
    # 'Volume Per Microsecond',
]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('minmax', MinMaxScaler()),
])

# Handling categorical features
categorical_features = [
    # 'Client ID',
    # 'File Extension',
    # 'Client IP'
]

print(f'Numeric features: {numeric_features} \nCategorical features: {categorical_features if categorical_features else None}')

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# AI Models used to train for Intrusion Detection System
models = [
    ('RandomForest', RandomForestClassifier(random_state=42, class_weight='balanced')),
    ('LogisticRegression', LogisticRegression(random_state=0, class_weight='balanced', max_iter=5000)),
    ('SupportVector', SVC(kernel='rbf', class_weight='balanced', probability=True)),
    ('NaiveBayes', GaussianNB()),
    ('XGBoost', xgb.XGBClassifier(eval_metric='mlogloss')),
    ('LGBoost', lgb.LGBMClassifier(force_col_wise=True)),
]

# Display model options
print("Select a model:")
for i, (name, model) in enumerate(models):
    print(f"{i} - {name}")

# Get user input
model_index = int(input("Enter the number corresponding to the model: "))

# Get selected model
model_name, model = models[model_index]
print(f"\nModel selected: {model_name}\n")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline with reduced complexity
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

print(f'Model used is: {model}')

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Assigning test sets
X_test, y_test = features_test, target_test


# Perform cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, features, target, cv=skf)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

param_grid = {}

# Hyperparameter tuning
if model_name == 'RandomForest':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
elif model_name == 'LogisticRegression':
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
elif model_name == 'SupportVector':
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['rbf', 'linear']
    }
elif model_name == 'XGBoost':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
elif model_name == 'LGBoost':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__num_leaves': [31, 50, 100],
        'classifier__min_data_in_leaf': [20, 50, 100],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
elif model_name == 'NaiveBayes':
    param_grid = {
        'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
    }

if param_grid:
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
else:
    best_model = pipeline.fit(X_train, y_train)

# Train the best model
best_model.fit(X_train, y_train)

result = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=42, scoring='accuracy')

# Create a DataFrame to map feature names to importances
feature_names = features.columns
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': result['importances_mean'],
    'Importance Std': result['importances_std']
})

print(importances_df)

# Make predictions
# y_pred = pipeline.predict(X_test)
y_pred = best_model.predict(X_val)

# Evaluate the model
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nValidation Classification Report:")
print(classification_report(y_val, y_pred))

print("\nValidation Accuracy Score:")
print(accuracy_score(y_val, y_pred))

# Plotting cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', label='Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title(f'Cross-Validation Accuracy per Fold with model {model_name}')
plt.legend()
plt.show()

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model: {model_name} \n Receiver Operating Characteristic (ROC) Validation Curve')
    plt.legend(loc='lower right')
    plt.show()

# Make predictions
# y_pred = pipeline.predict(X_test)
y_pred_test = best_model.predict(X_test)

# Evaluate the model
print("Test Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test))

print("\nTest Accuracy Score:")
print(accuracy_score(y_test, y_pred_test))

if hasattr(best_model, "predict_proba"):
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model: {model_name} \n Receiver Operating Characteristic (ROC) Test Curve')
    plt.legend(loc='lower right')
    plt.show()

# Inference: Display Ground Truth and Predicted values
results = pd.DataFrame({
    'Client ID': data_test['Client ID'],
    'Ground Truth': y_test,
    'Predicted': y_pred_test
})

print("\nInference Results:")
print(results)

# Display detected intruders
detected_intruders = results[results['Predicted'] == 1]
detected_clients = results[results['Predicted'] == 0]
print("\nDetected intruders:")
print(detected_intruders)

# Matrix for correctly and incorrectly detected intruders and clients
tn, fp, fn, tp = cm.ravel()
matrix = pd.DataFrame({
    'Correctly Detected': [tn, tp],
    'Incorrectly Detected': [fp, fn]
}, index=['clients', 'intruders'])

print("\nDetection Matrix:")
print(matrix)

print("\nConfusion Matrix:")
print(cm)

print(f'Client Detection Success Rate (TNR) for test dataset: {(tn / (tn + fp) * 100):.2f}%')
print(f'Intruder Detection Success Rate (TPR) for test dataset: {(tp / (tp + fn) * 100):.2f}%')

# Plot the matrix
matrix.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title(f'Correctly and Incorrectly Detected clients and intruders for model {model_name}')
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(title='Detection')
plt.show()

# Calculate evaluation metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
fpr = fp / (fp + tn)

# Calculate DR, FPR, and FNR for different thresholds
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr_list, recall_list, threshold_list = roc_curve(y_test, y_proba)

    dr_list = recall_list
    fnr_list = 1 - recall_list

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_list, dr_list, label='Detection Rate (Recall)')
    plt.plot(threshold_list, fpr_list, label='False Positive Rate (FPR)')
    plt.plot(threshold_list, fnr_list, label='False Negative Rate (FNR)')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(f'Model: {model_name} \n Detection Rate, FPR, FNR vs Threshold')
    plt.legend(loc='upper right')
    plt.show()
else:
    print("The model does not support probability predictions. Unable to calculate DR, FPR, and FNR for different "
          "thresholds.")
