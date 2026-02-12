# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ===============================
# 2. LOAD DATA
# ===============================
df = pd.read_csv("data/heart.csv")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Drop unnecessary column
df = df.drop(columns=["id"])

# Convert target to binary (0 = no disease, >0 = disease)
df["num"] = df["num"].apply(lambda x: 1 if int(x) > 0 else 0)

X = df.drop("num", axis=1)
y = df["num"]


# ===============================
# 3. IDENTIFY COLUMN TYPES
# ===============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool"]).columns


# ===============================
# 4. PREPROCESSING
# ===============================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ===============================
# 5. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 6. DEFINE MODELS
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

results = {}

# ===============================
# 7. TRAIN & VALIDATE
# ===============================
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")


# ===============================
# 8. SELECT BEST MODEL
# ===============================
best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])


# ===============================
# 9. RETRAIN ON FULL DATA
# ===============================
final_pipeline.fit(X, y)


# ===============================
# 10. SAVE MODEL
# ===============================
joblib.dump(final_pipeline, "model/model.pkl")

print("\nFinal model saved successfully!")
