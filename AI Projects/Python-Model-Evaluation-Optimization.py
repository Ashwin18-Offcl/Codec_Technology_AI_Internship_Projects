import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Display current working directory
# --------------------------------------------------
print("Current Working Directory:", os.getcwd())

# --------------------------------------------------
# Load dataset safely
# --------------------------------------------------
DATA_PATH = r"f:\Codec_Technology_AI_Internship_Projects\AI Projects\dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Validate target column
# --------------------------------------------------
TARGET_COLUMN = "target"

if TARGET_COLUMN not in data.columns:
    raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset")

# --------------------------------------------------
# Prepare features and target
# --------------------------------------------------
X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

# Convert categorical features to numeric
X = pd.get_dummies(X, drop_first=True)

# --------------------------------------------------
# Train-test split (safe split)
# --------------------------------------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # Fallback if stratify fails
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# --------------------------------------------------
# Initialize model
# --------------------------------------------------
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# Hyperparameter tuning
# --------------------------------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# --------------------------------------------------
# Train model
# --------------------------------------------------
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = best_model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
print("\nBest Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
