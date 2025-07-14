from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base, Transaction
import io

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
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
    total_rows = 0
    expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

    db = next(get_db())
    for chunk in pd.read_csv(io.StringIO(contents.decode('utf-8')), chunksize=1000):
        if list(chunk.columns) != expected_columns:
            return {"error": "CSV format mismatch"}
        for _, row in chunk.iterrows():
            transaction = Transaction(
                time=row['Time'], v1=row['V1'], v2=row['V2'], v3=row['V3'], v4=row['V4'],
                v5=row['V5'], v6=row['V6'], v7=row['V7'], v8=row['V8'], v9=row['V9'],
                v10=row['V10'], v11=row['V11'], v12=row['V12'], v13=row['V13'], v14=row['V14'],
                v15=row['V15'], v16=row['V16'], v17=row['V17'], v18=row['V18'], v19=row['V19'],
                v20=row['V20'], v21=row['V21'], v22=row['V22'], v23=row['V23'], v24=row['V24'],
                v25=row['V25'], v26=row['V26'], v27=row['V27'], v28=row['V28'], amount=row['Amount'],
                class_label=row['Class']
            )
            db.add(transaction)
        db.commit()
        total_rows += len(chunk)
    return {"message": "CSV uploaded and stored successfully", "rows": total_rows}

@app.get("/health")
def health():
    print("Health endpoint accessed")
    return {"status": "ok"}