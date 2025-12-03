from fastapi import FastAPI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Iris Prediction API")

# Create model with same params as train.py (no MLflow needed)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# Minimal training data for demo
model.fit([[5.1,3.5,1.4,0.2], [7.0,3.2,4.7,1.4], [6.3,3.3,6.0,2.5]], 
          ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

@app.get("/")
def root():
    return {"message": "Iris Prediction API - ready"}

@app.get("/predict")
def predict(sepal_length: float = 5.1, sepal_width: float = 3.5, petal_length: float = 1.4, petal_width: float = 0.2):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    df = pd.DataFrame(data, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    return {
        "prediction": prediction,
        "confidence": {str(cls): float(prob) for cls, prob in zip(model.classes_, probability)},
        "input": {"sepal_length": sepal_length, "sepal_width": sepal_width, 
                 "petal_length": petal_length, "petal_width": petal_width}
    }
