from flask import Flask, render_template, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
from explain import ModelExplainer

app = Flask(__name__)

# Загружаем модель и SHAP-объяснитель
model = CatBoostClassifier()
model.load_model("model.cbm")
explainer = ModelExplainer()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features')

        # Превращаем словарь в DataFrame (модель ждет таблицу)
        df = pd.DataFrame([features])

        # 1. Модель предсказывает вероятность дефолта (от 0 до 1)
        proba = float(model.predict_proba(df)[0, 1])
        decision = "approve" if proba < 0.5 else "decline"

        # 2. Получаем объяснения (SHAP)
        top_factors = explainer.get_top_factors(df)

        return jsonify({
            "status": "success",
            "decision": decision,
            "probability": round(proba, 4),
            "top_factors": top_factors
        })

    except Exception as e:
        # Если что-то упадет (например, не хватает колонки), мы увидим ошибку
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)