from fastapi import FastAPI, HTTPException
from app.schema import HeartData
from app.model_loader import predict, load_model

app = FastAPI(title="Heart Disease Prediction API")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_heart(data: HeartData):

    try:
        input_data = data.dict()
        result = predict(input_data)

        return {
            "model": "Random Forest",
            "prediction": result,
            "result": "Disease Detected" if result == 1 else "No Disease"
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
