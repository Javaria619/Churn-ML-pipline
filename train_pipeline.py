import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================
df = pd.read_csv("telco_churn.csv")

# Drop ID
df.drop("customerID", axis=1, inplace=True)

# Target encode
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# Columns
# =========================
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# =========================
# Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# =========================
# Pipeline
# =========================
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# =========================
# GridSearch (CORRECT)
# =========================
param_grid = {
    "classifier__C": [0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=1
)

# =========================
# Train
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# =========================
# Save model
# =========================
joblib.dump(grid.best_estimator_, "churn_pipeline.pkl")
print("Model saved as churn_pipeline.pkl")
