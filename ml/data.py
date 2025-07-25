import pandas as pd
from backend.database import SessionLocal, Transaction
from sqlalchemy.orm import Session

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
