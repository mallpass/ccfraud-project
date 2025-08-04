from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, Transaction
import io, os, json, joblib, torch
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
from sklearn.metrics import accuracy_score

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    expected_columns = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
    ]

    db = next(get_db())
    total_rows = 0

    try:
        for chunk in pd.read_csv(io.StringIO(contents.decode('utf-8')), chunksize=1000):
            if list(chunk.columns) != expected_columns:
                return {"error": "CSV format mismatch"}

            transactions = []
            for _, row in chunk.iterrows():
                transaction = Transaction(
                    time=float(row['Time']),
                    v1=float(row['V1']), v2=float(row['V2']), v3=float(row['V3']),
                    v4=float(row['V4']), v5=float(row['V5']), v6=float(row['V6']),
                    v7=float(row['V7']), v8=float(row['V8']), v9=float(row['V9']),
                    v10=float(row['V10']), v11=float(row['V11']), v12=float(row['V12']),
                    v13=float(row['V13']), v14=float(row['V14']), v15=float(row['V15']),
                    v16=float(row['V16']), v17=float(row['V17']), v18=float(row['V18']),
                    v19=float(row['V19']), v20=float(row['V20']), v21=float(row['V21']),
                    v22=float(row['V22']), v23=float(row['V23']), v24=float(row['V24']),
                    v25=float(row['V25']), v26=float(row['V26']), v27=float(row['V27']),
                    v28=float(row['V28']), amount=float(row['Amount']), class_label=int(row['Class'])
                )
                transactions.append(transaction)

            db.add_all(transactions)
            db.commit()
            total_rows += len(transactions)

    except ValueError as e:
        return {"error": f"Invalid data in CSV: {str(e)}"}
    except pd.errors.EmptyDataError:
        return {"error": "Uploaded file is empty or malformed"}
    except Exception as e:
        return {"error": f"Unexpected server error: {str(e)}"}

    return {
        "message": "CSV uploaded and stored successfully",
        "rows": total_rows
    }

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
    
    y_pred = model.use(X.values)  # Returns probabilities
    y_pred_binary = (y_pred >= 0.5).astype(int).flatten()  # Convert to 0/1 predictions
    
    accuracy = accuracy_score(y_true, y_pred_binary)

    return {
        "message": "Predictions completed successfully",
        "num_rows": len(df),
        "accuracy": float(accuracy),
        "predictions_preview": y_pred_binary[:10].tolist(),  # First 10 predictions
        "true_labels_preview": y_true[:10].tolist(),  # First 10 true labels
        "model_used": model_file
    }

@app.get("/health")
def health():
    return {"status": "ok"}
