# test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_csv():
    csv_content = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,"
        "V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,"
        "V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class\n"
        "0,-1.359807,-0.072781,2.536346,1.378155,-0.338321,0.462388,0.239599,"
        "0.098698,0.363787,0.090794,-0.5516,-0.6178,-0.99139,-0.31117,1.468177,"
        "-0.4704,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,0.028724,"
        "0.078721,0.307978,0.005273,0.038909,0.000802,149.62,0\n"
    )

    files = {
        "file": ("test.csv", csv_content, "text/csv")
    }

    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert response.json()["message"] == "CSV uploaded and stored successfully"
    assert response.json()["rows"] == 1

def test_upload_bad_columns():
    bad_csv = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,"
        "V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,"
        "V21,V22,V23,V24,V25,V26,V27,V28,Amount\n"  # Missing 'Class'
        "0,-1.3,-0.07,2.5,1.3,-0.3,0.4,0.2,0.09,0.3,0.09,"
        "-0.55,-0.61,-0.99,-0.31,1.46,-0.47,0.2,0.02,0.4,0.25,"
        "-0.01,0.27,0.02,0.07,0.3,0.005,0.03,0.0008,149.62\n"
    )
    files = {"file": ("bad.csv", bad_csv, "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "error" in response.json()

def test_upload_with_invalid_row_type():
    csv_content = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,"
        "V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,"
        "V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class\n"
        "0,-1.3,-0.07,2.5,1.3,-0.3,0.4,0.2,0.09,0.3,0.09,"
        "-0.55,-0.61,-0.99,-0.31,1.46,-0.47,0.2,0.02,0.4,0.25,"
        "-0.01,0.27,0.02,0.07,0.3,0.005,0.03,0.0008,INVALID,0\n"  # 'Amount' is invalid
    )
    files = {"file": ("invalid_row.csv", csv_content, "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "error" in response.json()

def test_upload_empty_file():
    empty_csv = ""
    files = {"file": ("empty.csv", empty_csv, "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "error" in response.json() or response.json()["rows"] == 0

def test_upload_multiple_rows():
    csv_content = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,"
        "V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,"
        "V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class\n"
    )
    row = "0,-1.3,-0.07,2.5,1.3,-0.3,0.4,0.2,0.09,0.3,0.09," \
          "-0.55,-0.61,-0.99,-0.31,1.46,-0.47,0.2,0.02,0.4,0.25," \
          "-0.01,0.27,0.02,0.07,0.3,0.005,0.03,0.0008,149.62,0\n"
    csv_content += row * 5

    files = {"file": ("multi.csv", csv_content, "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert response.json()["rows"] == 5

def test_predict_valid_csv():
    csv_content = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,"
        "V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,"
        "V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class\n"
        "0,-1.3,-0.07,2.5,1.3,-0.3,0.4,0.2,0.09,0.3,0.09,"
        "-0.55,-0.61,-0.99,-0.31,1.46,-0.47,0.2,0.02,0.4,0.25,"
        "-0.01,0.27,0.02,0.07,0.3,0.005,0.03,0.0008,149.62,0\n"
    )
    files = {"file": ("valid.csv", csv_content, "text/csv")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "CSV validated and parsed successfully."
    assert data["num_rows"] == 1
    assert data["features_shape"] == [1, 30]
    assert data["y_true_preview"] == [0]

def test_predict_missing_class_column():
    bad_csv = (
        "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,"
        "V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,"
        "V21,V22,V23,V24,V25,V26,V27,V28,Amount\n"
        "0,-1.3,-0.07,2.5,1.3,-0.3,0.4,0.2,0.09,0.3,0.09,"
        "-0.55,-0.61,-0.99,-0.31,1.46,-0.47,0.2,0.02,0.4,0.25,"
        "-0.01,0.27,0.02,0.07,0.3,0.005,0.03,0.0008,149.62\n"
    )
    files = {"file": ("bad.csv", bad_csv, "text/csv")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "CSV format mismatch" in response.json()["detail"]

def test_predict_empty_file():
    files = {"file": ("empty.csv", "", "text/csv")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid CSV" in response.json()["detail"]

