
import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from catboost import CatBoostClassifier

from data import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_features,
)
from explain import ModelExplainer
from counterfactual import find_counterfactual
from llm_explainer import generate_explanation


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.cbm"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

DEFAULT_THRESHOLD = 0.25

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}"
        )

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    print(f"Модель загружена: {MODEL_PATH}")

    return model


def load_threshold():
    if not METADATA_PATH.exists():
        print(
            "Метаданные не найдены. "
            f"Используется порог по умолчанию: {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        threshold = float(
            metadata.get("classification_threshold", DEFAULT_THRESHOLD)
        )

        print(f"Порог классификации загружен: {threshold}")

        return threshold

    except Exception as error:
        print(f"Ошибка загрузки порога: {error}")
        return DEFAULT_THRESHOLD


def prepare_input(features):
    if not isinstance(features, dict):
        raise ValueError("Данные заявки должны быть объектом.")

    missing_features = [
        feature
        for feature in RAW_FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "Не переданы обязательные признаки: "
            + ", ".join(missing_features)
        )

    input_data = {}

    for feature in RAW_FEATURES:
        value = features[feature]

        if value is None or value == "":
            raise ValueError(
                f"Поле '{feature}' не должно быть пустым."
            )

        if feature in NUMERIC_FEATURES:
            try:
                input_data[feature] = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Поле '{feature}' должно быть числовым."
                )
        else:
            input_data[feature] = str(value)

    input_df = pd.DataFrame([input_data])

    return input_df, input_data


def get_default_probability(input_df):
    probabilities = model.predict_proba(input_df)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("В модели отсутствует класс 1.")

    default_index = classes.index(1)

    return float(probabilities[0][default_index])


def convert_to_json_safe(value):
    """Рекурсивно преобразует numpy-типы в обычные Python-типы."""

    if isinstance(value, dict):
        return {
            str(key): convert_to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            convert_to_json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        return value.item()

    return value


# Загружаем модель один раз при запуске Flask.
model = load_model()

CLASSIFICATION_THRESHOLD = load_threshold()

explainer = ModelExplainer(model)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json(silent=True) or {}

        features = request_data.get("features")

        input_df, raw_features = prepare_input(features)

        # 1. Подготовка признаков для CatBoost.
        prepared_df = prepare_features(input_df)

        # 2. Расчёт вероятности дефолта.
        probability = get_default_probability(prepared_df)

        # 3. Решение модели.
        decision = (
            "approve"
            if probability < CLASSIFICATION_THRESHOLD
            else "decline"
        )

        # 4. SHAP-факторы.
        top_factors = explainer.explain(
            prepared_df,
            top_n=5
        )

        top_factors = convert_to_json_safe(top_factors)

        # 5. Контрфактический поиск.
        counterfactual = None

        if decision == "decline":
            counterfactual = find_counterfactual(
                original_features=raw_features,
                model=model,
                threshold=CLASSIFICATION_THRESHOLD,
                max_alternatives=3,
            )

        # 6. Понятное заключение.
        explanation = generate_explanation(
            probability=probability,
            threshold=CLASSIFICATION_THRESHOLD,
            decision=decision,
            factors=top_factors,
            counterfactual=counterfactual,
        )

        response = {
            "status": "success",
            "decision": decision,
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 2),
            "threshold": CLASSIFICATION_THRESHOLD,
            "top_factors": top_factors,
            "counterfactual": counterfactual,
            "llm_explanation": explanation["text"],
            "llm_source": explanation["source"],
        }

        return jsonify(convert_to_json_safe(response))

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error),
        }), 400

    except Exception as error:
        print(f"Ошибка при обработке заявки: {error}")

        return jsonify({
            "status": "error",
            "message": (
                "Произошла ошибка при оценке заявки. "
                "Проверьте данные и повторите попытку."
            ),
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
