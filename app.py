from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
from fastapi import FastAPI

app = FastAPI()

iris = load_iris()

# 🔥 CRITICAL FIX: constrain tree to match expected decision boundaries
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3,              # IMPORTANT: stabilizes splits
    min_samples_split=2,
    min_samples_leaf=1
)

model.fit(iris.data, iris.target)

class_names = iris.target_names.tolist()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):

    features = np.array([[sl, sw, pl, pw]])

    pred = int(model.predict(features)[0])

    return {
        "prediction": pred,
        "class_name": class_names[pred]
    }
