from fastapi import FastAPI
from .predict import predict   # <- IMPORTANT (dot)

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict_delay(payload: dict):
    return predict(payload)