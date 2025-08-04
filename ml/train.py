import os
import json
from datetime import datetime, timezone

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

from backend.database import SessionLocal, Transaction
from sqlalchemy.orm import Session
from ml.modeling import NNet, get_random_forest, get_nllr, get_xgboost, grid_search_nn

load_dotenv()
mlflow.set_experiment("CreditCardFraudDetection")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_data_from_db():
    session: Session = SessionLocal()
    transactions = session.query(Transaction).all()
    session.close()
    data = [{
        "Time": t.time,
        **{f"V{i}": getattr(t, f"v{i}") for i in range(1, 29)},
        "Amount": t.amount,
        "Class": t.class_label
    } for t in transactions]
    return pd.DataFrame(data)

def evaluate_model(model, X_val, y_val, is_torch=False):
    if is_torch:
        preds = model.use(X_val)
    else:
        preds = model.predict_proba(X_val)[:, 1]
    preds_binary = (preds >= 0.5).astype(int)
    return {
        "preds": preds,
        "accuracy": accuracy_score(y_val, preds_binary),
        "auprc": average_precision_score(y_val, preds)
    }

def save_model_with_metadata(model, model_path, metadata_path, metadata, is_torch):
    if is_torch:
        torch.save(model, model_path)
    else:
        import joblib
        joblib.dump(model, model_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def train_model(seed: int = 42):
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    with mlflow.start_run():
        df = get_data_from_db()
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        X = df.drop(columns=["Class"])
        y = df["Class"].astype("float64")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

        results = []

        # --- Random Forest ---
        rf = get_random_forest(seed)
        rf.fit(X_train, y_train)
        rf_eval = evaluate_model(rf, X_val, y_val)
        mlflow.log_metrics({
            "RF_AUPRC": rf_eval["auprc"],
            "RF_Accuracy": rf_eval["accuracy"]
        })
        results.append(("random_forest", rf, rf_eval, False))
        print(f"Random Forest AUPRC: {rf_eval['auprc']:.4f}")

        # --- XGBoost ---
        xgb = get_xgboost(seed)
        xgb.fit(X_train, y_train)
        xgb_eval = evaluate_model(xgb, X_val, y_val)
        mlflow.log_metrics({
            "XGB_AUPRC": xgb_eval["auprc"],
            "XGB_Accuracy": xgb_eval["accuracy"]
        })
        results.append(("xgboost", xgb, xgb_eval, False))
        print(f"XGBoost AUPRC: {xgb_eval['auprc']:.4f}")

        # --- Logistic Regression ---
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_val_std = scaler.transform(X_val)
        nllr = get_nllr()
        nllr.fit(X_train_std, y_train)
        nllr_eval = evaluate_model(nllr, X_val_std, y_val)
        mlflow.log_metrics({
            "NLLR_AUPRC": nllr_eval["auprc"],
            "NLLR_Accuracy": nllr_eval["accuracy"]
        })
        results.append(("nllr", nllr, nllr_eval, False))
        print(f"NLLR AUPRC: {nllr_eval['auprc']:.4f}")

        # --- Neural Net (Grid Search) ---
        grid_results = grid_search_nn(X_train.values, y_train.values, X_val.values, y_val.values)
        best_nn_result = grid_results[0]
        nn_model = best_nn_result['model']
        nn_eval = evaluate_model(nn_model, X_val.values, y_val.values.reshape(-1, 1), is_torch=True)
        mlflow.log_metrics({
            "NN_AUPRC": nn_eval["auprc"],
            "NN_Accuracy": nn_eval["accuracy"]
        })
        mlflow.log_params({
            "NN_Layers": best_nn_result['layers'],
            "NN_Dropout": best_nn_result['dropout'],
            "NN_LR": best_nn_result['lr'],
            "NN_Loss": best_nn_result['loss_type'],
            "NN_Optimizer": best_nn_result['optimizer'],
            "NN_Activation": best_nn_result['activation']
        })
        results.append(("neural_net", nn_model, nn_eval, True))
        print(f"Neural Net AUPRC: {nn_eval['auprc']:.4f}")

        # --- Select best model ---
        best_name, best_model, best_eval, is_torch = max(results, key=lambda r: r[2]["auprc"])
        print(f"Best model: {best_name} → AUPRC: {best_eval['auprc']:.4f}")

        # --- Save if better than previous ---
        model_path = os.path.join(MODELS_DIR, f"best_model_{best_name}.pt" if is_torch else f"best_model_{best_name}.pkl")
        metadata_path = model_path.replace(".pt", ".json").replace(".pkl", ".json")

        metadata = {
            "model_type": best_name,
            "auprc": best_eval["auprc"],
            "accuracy": best_eval["accuracy"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hyperparams": {
                k: v for k, v in best_nn_result.items() if k != "model"
            } if best_name == "neural_net" else {}

        }

        should_save = True
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                existing = json.load(f)
            if existing.get("auprc", 0) >= best_eval["auprc"]:
                print("Existing model is better or equal. Skipping overwrite.")
                should_save = False

        if should_save:
            print("Saving new best model.")
            save_model_with_metadata(best_model, model_path, metadata_path, metadata, is_torch)

        mlflow.log_param("best_model_type", best_name)
        mlflow.log_metric("BestModel_AUPRC", best_eval["auprc"])
        print(f"Model path: {model_path}")

if __name__ == "__main__":
    train_model()
