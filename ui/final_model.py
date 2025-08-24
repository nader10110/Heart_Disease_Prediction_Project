import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

PROJECT_DIR = r"D:\DATA_ANALYSIS\ML_\1\Heart_Disease_Project"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "top10_features.csv")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "Final_model.pkl")

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"✅ Model trained and saved at: {MODEL_PATH}")
