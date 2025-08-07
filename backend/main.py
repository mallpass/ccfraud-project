from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, Transaction
import io, os, json, joblib, torch
from sklearn.metrics import accuracy_score, average_precision_score
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")
if not logger.handlers:
    h = logging.StreamHandler()
    f = logging.Formatter("%(levelname)s: %(message)s")
    h.setFormatter(f)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

app = FastAPI()

# CORS
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables if needed
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    expected_columns = [
        'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class'
    ]

    total_rows = 0
    try:
        for chunk in pd.read_csv(io.StringIO(contents.decode('utf-8')), chunksize=1000):
            if list(chunk.columns) != expected_columns:
                msg = "CSV format mismatch"
                logger.warning(f"/upload rejected file: {msg}. Columns={list(chunk.columns)}")
                return {"error": msg}

            transactions = [
                Transaction(
                    time=float(r['Time']),
                    v1=float(r['V1']),  v2=float(r['V2']),  v3=float(r['V3']),
                    v4=float(r['V4']),  v5=float(r['V5']),  v6=float(r['V6']),
                    v7=float(r['V7']),  v8=float(r['V8']),  v9=float(r['V9']),
                    v10=float(r['V10']), v11=float(r['V11']), v12=float(r['V12']),
                    v13=float(r['V13']), v14=float(r['V14']), v15=float(r['V15']),
                    v16=float(r['V16']), v17=float(r['V17']), v18=float(r['V18']),
                    v19=float(r['V19']), v20=float(r['V20']), v21=float(r['V21']),
                    v22=float(r['V22']), v23=float(r['V23']), v24=float(r['V24']),
                    v25=float(r['V25']), v26=float(r['V26']), v27=float(r['V27']),
                    v28=float(r['V28']), amount=float(r['Amount']), class_label=int(r['Class'])
                )
                for _, r in chunk.iterrows()
            ]

            db.add_all(transactions)
            db.commit()
            total_rows += len(transactions)

        logger.info(f"/upload ingested {total_rows} rows into PostgreSQL.")
    except ValueError as e:
        logger.error(f"/upload ValueError: {e}")
        return {"error": f"Invalid data in CSV: {str(e)}"}
    except pd.errors.EmptyDataError:
        logger.warning("/upload received empty or malformed file.")
        return {"error": "Uploaded file is empty or malformed"}
    except Exception as e:
        logger.exception(f"/upload unexpected error: {e}")
        return {"error": f"Unexpected server error: {str(e)}"}

    return {"message": "CSV uploaded and stored successfully", "rows": total_rows}

@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    expected_columns = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
    ]
    if list(df.columns) != expected_columns:
        raise HTTPException(status_code=400, detail="CSV format mismatch")

    X = df.drop(columns=["Class"])
    y_true = df["Class"].values

    models_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
    
    print(f"Current file: {__file__}")
    print(f"Current directory: {os.path.dirname(__file__)}")
    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {os.path.exists(models_dir)}")
    if os.path.exists(models_dir):
        print(f"Files in models dir: {os.listdir(models_dir)}")

    model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model_") and f.endswith(".json")]
    if not model_files:
        raise HTTPException(status_code=500, detail="No trained model found")
    
    metadata_file = model_files[0]
    model_file = metadata_file.replace(".json", ".pt")
    
    model_path = os.path.join(models_dir, model_file)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"Model file not found: {model_file}")
    
    model = torch.load(model_path, weights_only=False)
    model.eval()
    
    y_pred_proba = model.use(X.values)  # Returns probabilities
    y_pred_binary = (y_pred_proba >= 0.5).astype(int).flatten()  # Convert to 0/1 predictions
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    auprc = average_precision_score(y_true, y_pred_proba.flatten())

    return {
        "message": "Predictions completed successfully",
        "num_rows": len(df),
        "accuracy": float(accuracy),
        "auprc": float(auprc),
        "probability_preview": y_pred_proba.flatten()[:10].tolist(),
        "predictions_preview": y_pred_binary[:10].tolist(),  # First 10 predictions
        "true_labels_preview": y_true[:10].tolist(),  # First 10 true labels
        "model_used": model_file
    }

@app.get("/health")
def health():
    return {"status": "ok"}