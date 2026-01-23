from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained pipeline
model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict")
def predict(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float
):
    data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    prediction = model.predict(data)[0]

    return {
        "name": "Hadiq C",
        "roll_no": "2022BCS0044",
        "wine_quality": int(round(prediction))
    }
