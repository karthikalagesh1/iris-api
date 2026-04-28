from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

# Load dataset
iris = load_iris()

# IMPORTANT FIX: ensure deterministic training
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=None
)

model.fit(iris.data, iris.target)

class_names = iris.target_names.tolist()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):

    # IMPORTANT: keep exact feature order
    features = np.array([[sl, sw, pl, pw]])

    pred = model.predict(features)[0]

    return {
        "prediction": int(pred),
        "class_name": class_names[int(pred)]
    }
