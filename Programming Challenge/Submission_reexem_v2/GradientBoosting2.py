import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# Load the data
df = pd.read_csv('TrainOnMe.csv')


# Z-score method for outlier detection
z_scores = np.abs(stats.zscore(df.select_dtypes(['float64', 'int64'])))
df = df[(z_scores < 3).all(axis=1)]

# Capping outliers instead of removing
for col in df.select_dtypes(['float64', 'int64']).columns:
    lower, upper = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], lower, upper)


# Split data into features and target
X = df.drop('y', axis=1)
y = df['y']

X['x1_x2'] = X['x1'] * X['x2'] # Improve

X['x2_x3'] = X['x2'] * X['x3'] # Improvement

X['x5_x11'] = X['x5'] * X['x11'] # Improve

X['x6_x8'] = X['x6'] * X['x8'] # Improve / Worse training data
X['x6_x9'] = X['x6'] * X['x9'] # Improve
X['x6_x11'] = X['x6'] * X['x11'] # Improve

X['x8_x9'] = X['x8'] * X['x9'] # Improve a lot

X['x9_x10'] = X['x9'] * X['x10'] # Improve - Same training data
X['x9_x11'] = X['x9'] * X['x11'] # Improve

X['x10_x11'] = X['x10'] * X['x11'] # Improve
X['x10_x13'] = X['x10'] * X['x13'] # Improve

X['x11_x13'] = X['x11'] * X['x13'] # Improve

# Convert categorical columns
X = pd.get_dummies(X, columns=['x7'])

# Normalize the features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Gradient Boosting classifier
gbm_classifier = GradientBoostingClassifier(random_state=42)

# Define the hyperparameter space for the grid search
param_grid = {
    'n_estimators': [102,],
    'learning_rate': [0.04],
    'max_depth': [3],
    'min_samples_split': [2], 
    'min_samples_leaf': [6],
    'subsample': [0.9]
}


# Best parameters: {'learning_rate': 0.04, 'max_depth': 3, 'min_samples_leaf': 6, 'min_samples_split': 2, 'n_estimators': 102, 'subsample': 0.9}
# Accuracy of Gradient Boosting Machines (optimized) on training data: 0.8271
# Accuracy of Gradient Boosting Machines (optimized) on test data: 0.8567

# Implement GridSearchCV
grid_search = GridSearchCV(gbm_classifier, param_grid, cv=5, verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)
grid_search.fit(X, y)


# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Train and predict on training data using the best model
y_pred_train = grid_search.predict(X)

# Calculate accuracy on training data
accuracy_train = accuracy_score(y, y_pred_train)
print(f"Accuracy of Gradient Boosting Machines (optimized) on training data: {accuracy_train:.4f}")

# Predict using the best model on test data
# y_pred_gbm = grid_search.predict(X_test)

# # Calculate accuracy on test data
# accuracy_test = accuracy_score(y_test, y_pred_gbm)
# print(f"Accuracy of Gradient Boosting Machines (optimized) on test data: {accuracy_test:.4f}")

# ---------------------------------------------------------------------------------------------
# Load the evaluation data

eval_df = pd.read_csv('EvaluateOnMe.csv')
print(eval_df.shape)

# Preprocess the data

# Z-score method for outlier detection
# z_scores_eval = np.abs(stats.zscore(eval_df.select_dtypes(['float64', 'int64'])))
# eval_df = eval_df[(z_scores_eval < 3).all(axis=1)]

for col in df.select_dtypes(['float64', 'int64']).columns:
    lower, upper = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], lower, upper)

print(eval_df.shape)

# Convert categorical columns
eval_df = pd.get_dummies(eval_df, columns=['x7'])

print(eval_df.shape)


# Match columns of training and evaluation datasets
missing_cols = set(X.columns) - set(eval_df.columns)
for c in missing_cols:
    eval_df[c] = 0
eval_df = eval_df[X.columns]

print(eval_df.shape)

# Normalize the features
eval_df = pd.DataFrame(scaler.transform(eval_df), columns=eval_df.columns)

# Predict using the trained model
eval_predictions = grid_search.predict(eval_df)

# Convert predictions back to original labels
eval_predictions_labels = le.inverse_transform(eval_predictions)

print(eval_df.shape)


# Write predictions to a .txt file
with open("predictions.txt", "w") as f:
    for label in eval_predictions_labels:
        f.write(f"{label}\n")
