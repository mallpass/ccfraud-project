from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    print("Health endpoint accessed")
    return {"status": "ok"}
