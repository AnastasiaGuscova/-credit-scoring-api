
import json
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, jsonify
from catboost import CatBoostClassifier

from data import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_features
)
from explain import ModelExplainer


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.cbm"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# Запасной путь для совместимости со старой структурой проекта
FALLBACK_MODEL_PATH = BASE_DIR / "model.cbm"

DEFAULT_THRESHOLD = 0.25

app = Flask(__name__)

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# ЗАГРУЗКА МОДЕЛИ И МЕТАДАННЫХ
# ============================================================

def load_model():
    """Загружает обученную модель CatBoost."""

    model_path = MODEL_PATH

    if not model_path.exists():
        model_path = FALLBACK_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            "Файл модели не найден. "
            "Проверь наличие models/model.cbm или model.cbm."
        )

    loaded_model = CatBoostClassifier()
    loaded_model.load_model(str(model_path))

    print(f"Модель загружена: {model_path}")

    return loaded_model


def load_threshold():
    """Загружает подобранный порог классификации."""

    if not METADATA_PATH.exists():
        print(
            f"Метаданные не найдены. "
            f"Используется порог по умолчанию: {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        threshold = metadata.get("threshold", DEFAULT_THRESHOLD)

        threshold = float(threshold)

        if not 0 < threshold < 1:
            raise ValueError("Порог должен находиться между 0 и 1.")

        print(f"Порог классификации загружен: {threshold}")

        return threshold

    except (ValueError, TypeError, json.JSONDecodeError) as error:
        print(
            f"Ошибка чтения порога: {error}. "
            f"Используется значение {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD


# Загружаем компоненты один раз при запуске приложения
model = load_model()
explainer = ModelExplainer()
CLASSIFICATION_THRESHOLD = load_threshold()


# ============================================================
# ПОДГОТОВКА ВХОДНЫХ ДАННЫХ
# ============================================================

def prepare_input(features):
    """
    Проверяет входные данные и приводит их к формату,
    который использовался при обучении модели.
    """

    if not isinstance(features, dict):
        raise ValueError(
            "Поле features должно содержать JSON-объект."
        )

    missing_features = [
        feature
        for feature in RAW_FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Не переданы обязательные признаки: "
            f"{', '.join(missing_features)}"
        )

    # Оставляем только исходные признаки датасета
    raw_data = {
        feature: features[feature]
        for feature in RAW_FEATURES
    }

    raw_df = pd.DataFrame([raw_data])

    # Используем ту же обработку и Feature Engineering,
    # что и при обучении модели
    prepared_df = prepare_features(raw_df)

    return prepared_df


# ============================================================
# ПРЕДСКАЗАНИЕ
# ============================================================

def get_default_probability(input_df):
    """Получает вероятность принадлежности к классу bad (1)."""

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "В модели отсутствует класс 1 (bad / дефолт)."
        )

    default_index = classes.index(1)

    return float(probabilities[default_index])


def convert_to_json_safe(value):
    """Преобразует numpy-типы в обычные Python-типы для JSON."""

    if isinstance(value, dict):
        return {
            key: convert_to_json_safe(item)
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


# ============================================================
# МАРШРУТЫ FLASK
# ============================================================

@app.route("/")
def index():
    """Главная страница приложения."""

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Обрабатывает заявку и возвращает результат скоринга."""

    try:
        # 1. Проверяем формат запроса
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Некорректный JSON-запрос."
            }), 400

        features = data.get("features")

        if features is None:
            return jsonify({
                "status": "error",
                "message": "В запросе отсутствует поле features."
            }), 400

        # 2. Подготавливаем признаки
        input_df = prepare_input(features)

        # 3. Рассчитываем вероятность дефолта
        probability = get_default_probability(input_df)

        # 4. Принимаем решение с использованием подобранного порога
        decision = (
            "approve"
            if probability < CLASSIFICATION_THRESHOLD
            else "decline"
        )

        # 5. Получаем SHAP-объяснение
        top_factors = explainer.get_top_factors(input_df)

        # 6. Формируем ответ
        response = {
            "status": "success",
            "decision": decision,
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 2),
            "threshold": CLASSIFICATION_THRESHOLD,
            "top_factors": convert_to_json_safe(top_factors)
        }

        return jsonify(response), 200

    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except Exception as error:
        app.logger.exception("Ошибка при обработке заявки")

        return jsonify({
            "status": "error",
            "message": "Не удалось обработать заявку."
        }), 500


# ============================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
