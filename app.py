from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier
from explain import ModelExplainer

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# Загружаем модель при старте сервера
model = CatBoostClassifier()
model.load_model("model.cbm")
explainer = ModelExplainer()


# Описываем, какие данные мы ждем от клиента (для валидации)
class ClientFeatures(BaseModel):
    features: dict  # В реальном проекте здесь был бы четкий список полей


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: ClientFeatures):
    try:
        # Превращаем словарь обратно в DataFrame
        df = pd.DataFrame([req.features])

        # Делаем предсказание (вероятность дефолта от 0 до 1)
        proba = float(model.predict_proba(df)[0, 1])

        # Переводим вероятность в банковский скор (300-850)
        score = int(300 + 550 * (1 - proba))
        decision = "approve" if proba < 0.5 else "decline"

        # Получаем объяснения
        top_factors = explainer.get_top_factors(df)

        return {
            "pd": round(proba, 4),
            "score": score,
            "decision": decision,
            "top_factors": top_factors,
            "model_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))