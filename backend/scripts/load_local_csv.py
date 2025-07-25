import os
import sys
import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database import SessionLocal, Transaction


CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ccx.csv")

EXPECTED_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"
]

def load_csv_to_db(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    if list(df.columns) != EXPECTED_COLUMNS:
        print("CSV columns do not match expected format.")
        print("Expected:", EXPECTED_COLUMNS)
        print("Got:", list(df.columns))
        return

    try:
        transactions = [
            Transaction(
                time=float(row["Time"]),
                v1=float(row["V1"]),
                v2=float(row["V2"]),
                v3=float(row["V3"]),
                v4=float(row["V4"]),
                v5=float(row["V5"]),
                v6=float(row["V6"]),
                v7=float(row["V7"]),
                v8=float(row["V8"]),
                v9=float(row["V9"]),
                v10=float(row["V10"]),
                v11=float(row["V11"]),
                v12=float(row["V12"]),
                v13=float(row["V13"]),
                v14=float(row["V14"]),
                v15=float(row["V15"]),
                v16=float(row["V16"]),
                v17=float(row["V17"]),
                v18=float(row["V18"]),
                v19=float(row["V19"]),
                v20=float(row["V20"]),
                v21=float(row["V21"]),
                v22=float(row["V22"]),
                v23=float(row["V23"]),
                v24=float(row["V24"]),
                v25=float(row["V25"]),
                v26=float(row["V26"]),
                v27=float(row["V27"]),
                v28=float(row["V28"]),
                amount=float(row["Amount"]),
                class_label=int(row["Class"]),
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"CSV row parsing failed: {e}")
        return

    session: Session = SessionLocal()
    try:
        session.add_all(transactions)
        session.commit()
        print(f"Successfully inserted {len(transactions)} rows.")
    except IntegrityError as e:
        session.rollback()
        print(f"Database commit failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    load_csv_to_db(CSV_PATH)
