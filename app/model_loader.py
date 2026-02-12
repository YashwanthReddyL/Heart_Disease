import joblib
import pandas as pd

model = None

def load_model():
    global model
    model = joblib.load("model/model.pkl")

def predict(data_dict):
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    return int(prediction[0])
