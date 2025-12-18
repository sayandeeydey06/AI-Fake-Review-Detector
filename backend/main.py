from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class ReviewRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict_review(data: ReviewRequest):
    text = data.text
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]

    return {
        "prediction": int(prediction)
    }
