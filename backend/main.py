from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, Transaction
import io

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
                # Validate and cast all values strictly
                transaction = Transaction(
                    time=float(row['Time']),
                    v1=float(row['V1']),
                    v2=float(row['V2']),
                    v3=float(row['V3']),
                    v4=float(row['V4']),
                    v5=float(row['V5']),
                    v6=float(row['V6']),
                    v7=float(row['V7']),
                    v8=float(row['V8']),
                    v9=float(row['V9']),
                    v10=float(row['V10']),
                    v11=float(row['V11']),
                    v12=float(row['V12']),
                    v13=float(row['V13']),
                    v14=float(row['V14']),
                    v15=float(row['V15']),
                    v16=float(row['V16']),
                    v17=float(row['V17']),
                    v18=float(row['V18']),
                    v19=float(row['V19']),
                    v20=float(row['V20']),
                    v21=float(row['V21']),
                    v22=float(row['V22']),
                    v23=float(row['V23']),
                    v24=float(row['V24']),
                    v25=float(row['V25']),
                    v26=float(row['V26']),
                    v27=float(row['V27']),
                    v28=float(row['V28']),
                    amount=float(row['Amount']),
                    class_label=int(row['Class'])
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

@app.get("/health")
def health():
    return {"status": "ok"}
