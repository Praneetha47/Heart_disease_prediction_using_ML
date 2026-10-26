"""
=============================================================================
Heart Disease Prediction - Machine Learning Model Training (SVM)
=============================================================================
Dataset: Cleveland Heart Disease Dataset (UCI Machine Learning Repository)
Model: Support Vector Machine (SVM / SVC with RBF Kernel)
Author: Academic ML Project
=============================================================================
"""

import os
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress deprecation warnings for probability parameter
warnings.filterwarnings("ignore", category=FutureWarning)

def train_and_export_model():
    print("=" * 65)
    print("Heart Disease Prediction - SVM Model Training Pipeline")
    print("=" * 65)

    # 1. Locate and load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "heart_disease_cleveland.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset not found at {csv_file}. Please ensure the file is present.")

    print(f"\n[1/6] Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"      Dataset successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Define features (X) and target (y)
    feature_cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    target_col = 'target'

    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # 3. Convert target to binary classification:
    # 0 = No Heart Disease, [1, 2, 3, 4] = Heart Disease (1)
    print("\n[2/6] Converting Cleveland target to binary classification...")
    y = y_raw.apply(lambda val: 0 if val == 0 else 1)
    
    no_disease_count = (y == 0).sum()
    disease_count = (y == 1).sum()
    print(f"      Target Distribution -> 0 (No Heart Disease): {no_disease_count}, 1 (Heart Disease): {disease_count}")

    # 4. Handle missing values using SimpleImputer with median strategy
    print("\n[3/6] Imputing missing values using SimpleImputer(strategy='median')...")
    missing_count = X.isnull().sum().sum()
    print(f"      Total missing values detected before imputation: {missing_count}")
    
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_cols)

    # 5. Split dataset into train and test sets (80/20, stratify=y, random_state=42)
    print("\n[4/6] Splitting dataset (80% train, 20% test, stratify=y, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed_df, 
        y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    print(f"      Training sample size: {X_train.shape[0]} | Testing sample size: {X_test.shape[0]}")

    # 6. Standardize features using StandardScaler
    # Note: SVM is a distance-based classifier, so StandardScaler is critical for optimal hyperplane margin!
    print("\n[5/6] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Train Support Vector Machine Classifier (SVC)
    print("\n[6/6] Training Support Vector Classifier (SVC with RBF Kernel)...")
    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)

    # 8. Evaluate model performance
    y_pred = svm_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print("SVM MODEL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Algorithm: Support Vector Machine (Kernel: Radial Basis Function - RBF)")
    print(f"Accuracy Score: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=["No Heart Disease (0)", "Heart Disease (1)"]
    ))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f" [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"  [FN={cm[1][0]}, TP={cm[1][1]}]]\n")

    print(f"Support Vectors Count: {len(svm_model.support_)} vectors ({svm_model.n_support_} per class)")

    # 9. Save trained SVM model, scaler, imputer, and metadata into heart_disease_model.pkl
    model_payload = {
        "model": svm_model,
        "model_name": "Support Vector Machine (SVM)",
        "kernel": "rbf",
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": feature_cols,
        "accuracy": acc
    }

    output_model_path = os.path.join(script_dir, "heart_disease_model.pkl")
    joblib.dump(model_payload, output_model_path)
    print("\n" + "=" * 65)
    print(f"SVM Model package saved successfully to:\n{output_model_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_and_export_model()
