"""
=============================================================================
Heart Disease Prediction - Flask REST API Backend (with Authentication)
=============================================================================
Provides endpoints for:
1. User Registration (POST /register)
2. User Authentication / Login (POST /login)
3. Health check (GET /)
4. Model inference for 13 clinical biomarkers (POST /predict)
=============================================================================
"""

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) so frontend pages (running locally
# or served on a different port/protocol) can communicate seamlessly with Flask
CORS(app)

# Path to the serialized model artifact and SQLite database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "users.db")

# Global containers for loaded artifacts
model_bundle = None
ml_model = None
scaler = None
imputer = None
feature_names = []


def init_db():
    """Initialize SQLite database for user accounts and seed a demo account."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Prediction History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT,
            age REAL,
            sex REAL,
            cp REAL,
            trestbps REAL,
            chol REAL,
            fbs REAL,
            restecg REAL,
            thalach REAL,
            exang REAL,
            oldpeak REAL,
            slope REAL,
            ca REAL,
            thal REAL,
            prediction INTEGER,
            result TEXT,
            risk TEXT,
            no_disease_probability REAL,
            disease_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Seed default demo account if it doesn't exist
    cursor.execute("SELECT id FROM users WHERE username = 'demo' OR email = 'demo@cardiopredict.com'")
    demo_user = cursor.fetchone()
    if not demo_user:
        demo_pw_hash = generate_password_hash("demo123")
        cursor.execute(
            "INSERT INTO users (fullname, username, email, password_hash) VALUES (?, ?, ?, ?)",
            ("Dr. Demo Clinician", "demo", "demo@cardiopredict.com", demo_pw_hash)
        )
        demo_id = cursor.lastrowid
        print("[*] Seeded default demo account: demo / demo123")
    else:
        demo_id = demo_user[0]

    # Seed initial history sample if table is empty
    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO prediction_history (
                user_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                prediction, result, risk, no_disease_probability, disease_probability, created_at
            ) VALUES 
            (?, 'John Doe (Cardiology Clinic)', 54, 1, 4, 140, 239, 0, 0, 160, 0, 1.2, 1, 0, 3, 0, 'No Heart Disease', 'Lower Risk', 95.7, 4.3, datetime('now', '-2 hours')),
            (?, 'Eleanor Vance (ICU Referral)', 67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 1, 'Heart Disease Detected', 'Higher Risk', 12.3, 87.7, datetime('now', '-1 hours'))
        """, (demo_id, demo_id))
        print("[*] Seeded demo prediction history records.")

    conn.commit()
    conn.close()


def get_db_connection():
    """Return a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_artifacts():
    """Load the trained model, scaler, imputer, and feature names from the .pkl file."""
    global model_bundle, ml_model, scaler, imputer, feature_names
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Artifact file '{MODEL_PATH}' not found! "
            "Please run 'python train_model.py' first to train and generate the model."
        )
    print(f"[*] Loading ML model bundle from: {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    ml_model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    imputer = model_bundle["imputer"]
    feature_names = model_bundle["feature_names"]
    model_name = model_bundle.get("model_name", "Support Vector Machine (SVM)")
    print(f"[OK] Artifacts loaded successfully. Algorithm: {model_name}. Required features: {feature_names}")


# Initialize database and load model artifacts at application startup
init_db()
load_artifacts()


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.route("/register", methods=["POST"])
def register():
    """
    User registration endpoint.
    Expects JSON: { "fullname": str, "username": str, "email": str, "password": str }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid request body. Expected JSON payload."}), 400

        fullname = (data.get("fullname") or "").strip()
        username = (data.get("username") or "").strip().lower()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        # Basic validations
        if not fullname or not username or not email or not password:
            return jsonify({"error": "All fields (Full Name, Username, Email, Password) are required."}), 400

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long."}), 400

        if "@" not in email or "." not in email:
            return jsonify({"error": "Please provide a valid email address."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check for existing username or email
        cursor.execute("SELECT id, username, email FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            conn.close()
            if existing_user["username"] == username:
                return jsonify({"error": f"Username '{username}' is already taken. Please choose another."}), 409
            else:
                return jsonify({"error": f"An account with email '{email}' already exists. Please log in."}), 409

        # Hash password and insert
        pw_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (fullname, username, email, password_hash) VALUES (?, ?, ?, ?)",
            (fullname, username, email, pw_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"[+] New user registered successfully: {username} ({email})")

        return jsonify({
            "message": "Registration successful! You are now logged in.",
            "user": {
                "id": user_id,
                "fullname": fullname,
                "username": username,
                "email": email
            }
        }), 201

    except Exception as e:
        return jsonify({"error": "Failed to complete registration.", "details": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    """
    User login endpoint.
    Expects JSON: { "username_or_email": str, "password": str }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid request body. Expected JSON payload."}), 400

        identifier = (data.get("username_or_email") or "").strip().lower()
        password = data.get("password") or ""

        if not identifier or not password:
            return jsonify({"error": "Username/Email and Password are required."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, fullname, username, email, password_hash FROM users WHERE username = ? OR email = ?",
            (identifier, identifier)
        )
        user = cursor.fetchone()
        conn.close()

        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Invalid username/email or password. Please try again."}), 401

        print(f"[+] User logged in successfully: {user['username']}")

        return jsonify({
            "message": "Login successful!",
            "user": {
                "id": user["id"],
                "fullname": user["fullname"],
                "username": user["username"],
                "email": user["email"]
            }
        }), 200

    except Exception as e:
        return jsonify({"error": "Failed to authenticate.", "details": str(e)}), 500


# =============================================================================
# Model & Health Endpoints
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Health check endpoint.
    Returns status of the API, authentication system, and model information.
    """
    model_name = model_bundle.get("model_name", "Support Vector Machine (SVM)")
    return jsonify({
        "status": "online",
        "service": "Heart Disease Prediction API",
        "model": model_name,
        "features_required": feature_names,
        "auth_enabled": True,
        "endpoints": {
            "GET /": "Health check",
            "POST /register": "Register a new user account",
            "POST /login": "Sign in with username/email and password",
            "POST /predict": "Submit 13 patient biomarkers for heart disease risk prediction"
        }
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.
    Expects a JSON payload containing the 13 clinical features.
    Performs data validation, median imputation, standard scaling, and SVM inference.
    """
    try:
        # 1. Receive JSON payload
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({
                "error": "Invalid request body. Expected a JSON object with patient metrics."
            }), 400

        # 2. Validate presence and types of all 13 required features
        missing_fields = [feat for feat in feature_names if feat not in data or data[feat] is None or data[feat] == ""]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Parse numerical values safely
        cleaned_features = {}
        for feat in feature_names:
            try:
                val = float(data[feat])
                cleaned_features[feat] = val
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid numerical value for field '{feat}': received '{data[feat]}'"
                }), 400

        # 3. Convert input dictionary into pandas DataFrame with exact column ordering
        input_df = pd.DataFrame([cleaned_features], columns=feature_names)

        # 4. Apply saved median imputer
        imputed_array = imputer.transform(input_df)
        imputed_df = pd.DataFrame(imputed_array, columns=feature_names)

        # 5. Apply saved standard scaler
        scaled_array = scaler.transform(imputed_df)

        # 6. Perform inference with trained ML model (SVM)
        prediction_class = int(ml_model.predict(scaled_array)[0])

        # Probability estimates: [P(class 0), P(class 1)]
        probabilities = ml_model.predict_proba(scaled_array)[0]
        no_disease_prob = round(float(probabilities[0]) * 100, 1)
        disease_prob = round(float(probabilities[1]) * 100, 1)

        # 7. Formulate readable risk and result strings
        if prediction_class == 0:
            result_str = "No Heart Disease"
            risk_str = "Lower Risk"
        else:
            result_str = "Heart Disease Detected"
            risk_str = "Higher Risk"

        # 8. Save prediction event to history table in SQLite
        history_id = None
        try:
            user_id = data.get("user_id")
            patient_name = data.get("patient_name") or f"Patient ({int(cleaned_features['age'])}y, {'Male' if cleaned_features['sex'] == 1 else 'Female'})"
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO prediction_history (
                    user_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal,
                    prediction, result, risk, no_disease_probability, disease_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, patient_name,
                cleaned_features['age'], cleaned_features['sex'], cleaned_features['cp'],
                cleaned_features['trestbps'], cleaned_features['chol'], cleaned_features['fbs'],
                cleaned_features['restecg'], cleaned_features['thalach'], cleaned_features['exang'],
                cleaned_features['oldpeak'], cleaned_features['slope'], cleaned_features['ca'],
                cleaned_features['thal'],
                prediction_class, result_str, risk_str, no_disease_prob, disease_prob
            ))
            conn.commit()
            history_id = cursor.lastrowid
            conn.close()
        except Exception as db_err:
            print(f"[!] Warning: Failed to record history to SQLite: {db_err}")

        response_payload = {
            "prediction": prediction_class,
            "result": result_str,
            "risk": risk_str,
            "no_disease_probability": no_disease_prob,
            "disease_probability": disease_prob,
            "history_id": history_id
        }

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "error": "An internal server error occurred during prediction.",
            "details": str(e)
        }), 500


# =============================================================================
# Patient Assessment History Endpoints
# =============================================================================

@app.route("/history", methods=["GET"])
def get_prediction_history():
    """
    Retrieve past patient prediction assessments from SQLite.
    Optionally filters by user_id if provided.
    """
    try:
        user_id = request.args.get("user_id")
        conn = get_db_connection()
        cursor = conn.cursor()

        if user_id:
            cursor.execute(
                "SELECT * FROM prediction_history WHERE user_id = ? OR user_id IS NULL ORDER BY created_at DESC",
                (user_id,)
            )
        else:
            cursor.execute("SELECT * FROM prediction_history ORDER BY created_at DESC")

        rows = cursor.fetchall()
        conn.close()

        records = []
        for r in rows:
            records.append({
                "id": r["id"],
                "user_id": r["user_id"],
                "patient_name": r["patient_name"],
                "age": r["age"],
                "sex": r["sex"],
                "cp": r["cp"],
                "trestbps": r["trestbps"],
                "chol": r["chol"],
                "fbs": r["fbs"],
                "restecg": r["restecg"],
                "thalach": r["thalach"],
                "exang": r["exang"],
                "oldpeak": r["oldpeak"],
                "slope": r["slope"],
                "ca": r["ca"],
                "thal": r["thal"],
                "prediction": r["prediction"],
                "result": r["result"],
                "risk": r["risk"],
                "no_disease_probability": r["no_disease_probability"],
                "disease_probability": r["disease_probability"],
                "created_at": (r["created_at"].replace(" ", "T") + "Z") if r["created_at"] and "Z" not in r["created_at"] and "+" not in r["created_at"] else r["created_at"]
            })

        return jsonify({"history": records, "total": len(records)}), 200

    except Exception as e:
        return jsonify({"error": "Failed to fetch prediction history.", "details": str(e)}), 500


@app.route("/history/clear", methods=["POST"])
def clear_prediction_history():
    """Clear all prediction history records."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_id = data.get("user_id")
        conn = get_db_connection()
        cursor = conn.cursor()

        if user_id:
            cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("DELETE FROM prediction_history")

        conn.commit()
        conn.close()
        return jsonify({"message": "Patient prediction history cleared successfully."}), 200

    except Exception as e:
        return jsonify({"error": "Failed to clear history.", "details": str(e)}), 500


@app.route("/history/<int:history_id>", methods=["DELETE"])
def delete_history_record(history_id):
    """Delete a single history record by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history WHERE id = ?", (history_id,))
        conn.commit()
        conn.close()
        return jsonify({"message": f"Record #{history_id} deleted successfully."}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to delete record #{history_id}.", "details": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask development server on 127.0.0.1:5000
    print("[*] Starting Heart Disease Prediction Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
