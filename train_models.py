import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ------------------ Ensure models folder exists ------------------
os.makedirs("models", exist_ok=True)

# ------------------ Function to Train & Save Random Forest ------------------
def train_save_rf_model(X, y, disease_name):
    """
    Train Random Forest model and save it along with feature names.
    Prints accuracy and classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=200 if disease_name=="Diabetes" else 100)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {disease_name} Random Forest Accuracy: {acc:.4f}")
    print(f"Classification Report for {disease_name}:\n", classification_report(y_test, y_pred))
    pickle.dump((rf_model, X.columns.tolist()), open(f"models/{disease_name.lower()}_rf_model.pkl", "wb"))
    print(f"🎉 {disease_name} model saved in 'models/{disease_name.lower()}_rf_model.pkl'")

# ------------------- HEART DISEASE -------------------
print("\n--- TRAINING HEART DISEASE MODEL ---")
heart_path = "data/heart_cleveland_upload.csv"
if not os.path.exists(heart_path):
    raise FileNotFoundError(f"File not found: {heart_path}")
heart_df = pd.read_csv(heart_path)
if "condition" not in heart_df.columns:
    raise ValueError("Column 'condition' not found in heart dataset.")
X_heart = heart_df.drop("condition", axis=1)
y_heart = heart_df["condition"]
train_save_rf_model(X_heart, y_heart, "Heart")

# ------------------- DIABETES -------------------
print("\n--- TRAINING DIABETES MODEL ---")
diabetes_path = "data/Dataset of Diabetes .csv"
if not os.path.exists(diabetes_path):
    raise FileNotFoundError(f"File not found: {diabetes_path}")
diabetes_df = pd.read_csv(diabetes_path)
print("Diabetes columns:", diabetes_df.columns)

# Clean CLASS column: strip spaces, uppercase, and filter
diabetes_df["CLASS"] = diabetes_df["CLASS"].astype(str).str.strip().str.upper()
print("Diabetes CLASS value counts (before filtering):\n", diabetes_df["CLASS"].value_counts())

# Remove classes with <2 samples
class_counts = diabetes_df["CLASS"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
diabetes_df = diabetes_df[diabetes_df["CLASS"].isin(valid_classes)]

# Optionally, keep only 'Y' and 'N' for binary classification
diabetes_df = diabetes_df[diabetes_df["CLASS"].isin(['Y', 'N'])]
print("Diabetes CLASS value counts (after filtering):\n", diabetes_df["CLASS"].value_counts())

# Map 'Y' to 1, 'N' to 0
diabetes_df["CLASS"] = diabetes_df["CLASS"].map({'Y': 1, 'N': 0})

for col in ["CLASS", "ID", "No_Pation"]:
    if col not in diabetes_df.columns:
        raise ValueError(f"Column '{col}' not found in diabetes dataset.")
X_diabetes = diabetes_df.drop(["CLASS", "ID", "No_Pation"], axis=1)
y_diabetes = diabetes_df["CLASS"]
X_diabetes = pd.get_dummies(X_diabetes, drop_first=True)
print("Diabetes features after get_dummies:\n", X_diabetes.head())
train_save_rf_model(X_diabetes, y_diabetes, "Diabetes")

# ------------------- STROKE -------------------
print("\n--- TRAINING STROKE MODEL ---")
stroke_path = "data/healthcare-dataset-stroke-data.csv"
if not os.path.exists(stroke_path):
    raise FileNotFoundError(f"File not found: {stroke_path}")
stroke_df = pd.read_csv(stroke_path)
print("Stroke columns:", stroke_df.columns)
if "stroke" not in stroke_df.columns:
    raise ValueError("Column 'stroke' not found in stroke dataset.")
if "id" in stroke_df.columns:
    stroke_df = stroke_df.drop("id", axis=1)
print("Stroke target value counts:\n", stroke_df["stroke"].value_counts())
stroke_df = pd.get_dummies(stroke_df, drop_first=True)
X_stroke = stroke_df.drop("stroke", axis=1)
y_stroke = stroke_df["stroke"]
print("Stroke features before imputation nulls:\n", X_stroke.isnull().sum())
imputer = SimpleImputer(strategy="median")
X_stroke = pd.DataFrame(imputer.fit_transform(X_stroke), columns=X_stroke.columns)
print("Stroke features after imputation nulls:\n", X_stroke.isnull().sum())
print("Stroke features dtypes:\n", X_stroke.dtypes)
train_save_rf_model(X_stroke, y_stroke, "Stroke")

print("\n🎉 All Random Forest models trained and saved in 'models/' folder.")