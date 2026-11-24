
"""
Обучение модели кредитного скоринга на CatBoost.

Этапы:
1. Загрузка подготовленных данных.
2. Обучение двух конфигураций CatBoost:
   - без балансировки классов;
   - с автоматической балансировкой классов.
3. Сравнение моделей на validation.
4. Выбор лучшей модели.
5. Подбор порога классификации на validation.
6. Финальная оценка на test.
7. Логирование эксперимента в MLflow.
8. Сохранение модели и метаданных.

Целевой класс:
    0 — хороший заёмщик;
    1 — плохой заёмщик (дефолт).
"""

import json
from pathlib import Path

import mlflow
import mlflow.catboost

from catboost import CatBoostClassifier

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from data import (
    load_and_split,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES
)


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

RANDOM_STATE = 42

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.cbm"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Параметры CatBoost
MODEL_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "allow_writing_files": False
}

# Диапазон порогов для подбора на validation
THRESHOLDS = [
    round(value, 2)
    for value in [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80
    ]
]


# ============================================================
# 2. ПОДГОТОВКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ============================================================

def get_categorical_features(X):
    """
    Определяет категориальные признаки для CatBoost.

    CatBoost может работать с категориальными признаками
    напрямую, поэтому One-Hot Encoding здесь не нужен.
    """

    categorical_features = (
        CATEGORICAL_FEATURES
        + ENGINEERED_CATEGORICAL_FEATURES
    )

    # Проверяем, что все ожидаемые категориальные признаки есть
    missing = [
        column
        for column in categorical_features
        if column not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Отсутствуют категориальные признаки: {missing}"
        )

    return categorical_features


# ============================================================
# 3. ПОЛУЧЕНИЕ ВЕРОЯТНОСТИ ДЕФОЛТА
# ============================================================

def get_default_probability(model, X):
    """
    Возвращает вероятность класса 1 (bad/default).

    Не предполагаем, что класс 1 обязательно находится
    во втором столбце predict_proba: проверяем model.classes_.
    """

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Обученная модель не содержит класс 1 (default)."
        )

    class_index = classes.index(1)

    probabilities = model.predict_proba(X)[:, class_index]

    return probabilities


# ============================================================
# 4. РАСЧЁТ МЕТРИК
# ============================================================

def calculate_metrics(y_true, probabilities, threshold=0.5):
    """
    Рассчитывает основные метрики бинарной классификации.

    ROC-AUC оценивает качество ранжирования объектов.
    Precision показывает точность обнаруженных дефолтов.
    Recall показывает полноту обнаружения дефолтов.
    F1 объединяет Precision и Recall.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    auc = roc_auc_score(y_true, probabilities)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    gini = 2 * auc - 1

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    return {
        "roc_auc": float(auc),
        "gini": float(gini),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist()
    }


# ============================================================
# 5. ПОДБОР ПОРОГА КЛАССИФИКАЦИИ
# ============================================================

def find_best_threshold(y_true, probabilities):
    """
    Подбирает порог классификации по максимальному F1-score.

    Подбор выполняется только на validation.
    Тестовая выборка при этом не используется.
    """

    best_threshold = 0.5
    best_f1 = -1.0

    print("\nПодбор порога классификации...")

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"F1-score: {current_f1:.4f}"
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print("\nЛучший порог:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_f1:.4f}")

    return best_threshold


# ============================================================
# 6. ОБУЧЕНИЕ ОДНОЙ МОДЕЛИ
# ============================================================

def train_candidate(
    X_train,
    y_train,
    X_val,
    y_val,
    categorical_features,
    class_weight_mode
):
    """
    Обучает одну конфигурацию CatBoost.

    class_weight_mode:
        None       — без балансировки;
        "Balanced" — автоматическая балансировка классов.
    """

    params = MODEL_PARAMS.copy()

    if class_weight_mode is not None:
        params["auto_class_weights"] = class_weight_mode

    model = CatBoostClassifier(
        **params,
        cat_features=categorical_features
    )

    print("\n" + "=" * 60)

    if class_weight_mode is None:
        print("ОБУЧЕНИЕ БЕЗ БАЛАНСИРОВКИ КЛАССОВ")
    else:
        print("ОБУЧЕНИЕ С БАЛАНСИРОВКОЙ КЛАССОВ")

    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=100
    )

    # Оцениваем модель на validation
    val_probabilities = get_default_probability(
        model,
        X_val
    )

    val_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    print(f"\nValidation ROC-AUC: {val_auc:.4f}")
    print(f"Лучшее количество итераций: {model.best_iteration_}")

    return model, val_auc, params


# ============================================================
# 7. ПЕЧАТЬ РЕЗУЛЬТАТОВ
# ============================================================

def print_metrics(metrics, title):
    """
    Выводит метрики модели в удобном формате.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Gini:     {metrics['gini']:.4f}")
    print(f"Precision:{metrics['precision']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


# ============================================================
# 8. ОСНОВНОЙ ПРОЦЕСС ОБУЧЕНИЯ
# ============================================================

def train_model():

    print("\nЗАГРУЗКА ДАННЫХ")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = load_and_split()

    categorical_features = get_categorical_features(
        X_train
    )

    print("\nКоличество обучающих признаков:", X_train.shape[1])
    print("Количество категориальных признаков:", len(categorical_features))

    print("\nРаспределение целевых классов в train:")
    print(y_train.value_counts().sort_index())

    # Создаём папку для сохранения модели
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Настраиваем MLflow
    mlflow.set_experiment("Credit Scoring")

    with mlflow.start_run(run_name="CatBoost comparison"):

        # Логируем общие параметры эксперимента
        mlflow.log_params({
            "iterations": MODEL_PARAMS["iterations"],
            "learning_rate": MODEL_PARAMS["learning_rate"],
            "depth": MODEL_PARAMS["depth"],
            "random_seed": RANDOM_STATE,
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "test_size": len(X_test),
            "features_count": X_train.shape[1],
            "categorical_features_count": len(categorical_features)
        })

        # ----------------------------------------------------
        # СРАВНЕНИЕ ДВУХ КОНФИГУРАЦИЙ
        # ----------------------------------------------------

        candidates = []

        configurations = [
            ("without_balancing", None),
            ("balanced", "Balanced")
        ]

        for candidate_name, class_weight_mode in configurations:

            with mlflow.start_run(
                run_name=candidate_name,
                nested=True
            ):

                model, val_auc, params = train_candidate(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    categorical_features,
                    class_weight_mode
                )

                # Логируем параметры модели
                mlflow.log_params({
                    "candidate": candidate_name,
                    "class_weight_mode": (
                        class_weight_mode
                        if class_weight_mode is not None
                        else "None"
                    ),
                    "best_iteration": model.best_iteration_
                })

                mlflow.log_metric(
                    "validation_roc_auc",
                    val_auc
                )

                # Сохраняем результат для выбора лучшей модели
                candidates.append({
                    "name": candidate_name,
                    "model": model,
                    "val_auc": val_auc,
                    "params": params,
                    "class_weight_mode": class_weight_mode
                })

        # ----------------------------------------------------
        # ВЫБОР ЛУЧШЕЙ МОДЕЛИ
        # ----------------------------------------------------

        best_candidate = max(
            candidates,
            key=lambda item: item["val_auc"]
        )

        best_model = best_candidate["model"]

        print("\n" + "=" * 60)
        print("ВЫБРАНА ЛУЧШАЯ МОДЕЛЬ")
        print("=" * 60)

        print("Конфигурация:", best_candidate["name"])
        print(
            f"Validation ROC-AUC: "
            f"{best_candidate['val_auc']:.4f}"
        )

        # ----------------------------------------------------
        # ПОДБОР ПОРОГА НА VALIDATION
        # ----------------------------------------------------

        val_probabilities = get_default_probability(
            best_model,
            X_val
        )

        best_threshold = find_best_threshold(
            y_val,
            val_probabilities
        )

        # Считаем метрики выбранной модели на validation
        validation_metrics = calculate_metrics(
            y_val,
            val_probabilities,
            best_threshold
        )

        print_metrics(
            validation_metrics,
            "РЕЗУЛЬТАТЫ НА VALIDATION"
        )

        # ----------------------------------------------------
        # ФИНАЛЬНАЯ ОЦЕНКА НА TEST
        # ----------------------------------------------------

        # Тестовая выборка используется только сейчас:
        # модель и порог уже выбраны.
        test_probabilities = get_default_probability(
            best_model,
            X_test
        )

        test_metrics = calculate_metrics(
            y_test,
            test_probabilities,
            best_threshold
        )

        print_metrics(
            test_metrics,
            "ИТОГОВЫЕ РЕЗУЛЬТАТЫ НА TEST"
        )

        test_predictions = (
            test_probabilities >= best_threshold
        ).astype(int)

        print("\nПодробный отчёт классификации:")
        print(
            classification_report(
                y_test,
                test_predictions,
                labels=[0, 1],
                target_names=[
                    "good (хороший заёмщик)",
                    "bad (дефолт)"
                ],
                zero_division=0
            )
        )

        # ----------------------------------------------------
        # ЛОГИРОВАНИЕ ИТОГОВЫХ МЕТРИК В MLFLOW
        # ----------------------------------------------------

        mlflow.log_metrics({
            f"validation_{key}": value
            for key, value in validation_metrics.items()
            if isinstance(value, (int, float))
        })

        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_metrics.items()
            if isinstance(value, (int, float))
        })

        mlflow.log_param(
            "selected_model",
            best_candidate["name"]
        )

        mlflow.log_param(
            "decision_threshold",
            best_threshold
        )

        # ----------------------------------------------------
        # СОХРАНЕНИЕ МОДЕЛИ
        # ----------------------------------------------------

        best_model.save_model(
            str(MODEL_PATH)
        )

        # Дополнительно сохраняем копию в корне проекта,
        # чтобы текущий flask_app.py продолжал находить model.cbm.
        best_model.save_model("model.cbm")

        # ----------------------------------------------------
        # СОХРАНЕНИЕ МЕТАДАННЫХ
        # ----------------------------------------------------

        metadata = {
            "model_name": "CatBoost Credit Scoring",
            "model_version": "1.0.0",
            "dataset": "German Credit (credit-g)",
            "target_mapping": {
                "good": 0,
                "bad": 1
            },
            "selected_configuration": best_candidate["name"],
            "class_weight_mode": best_candidate["class_weight_mode"],
            "decision_threshold": best_threshold,
            "validation_roc_auc": validation_metrics["roc_auc"],
            "test_metrics": test_metrics,
            "numeric_features": (
                NUMERIC_FEATURES
                + ENGINEERED_NUMERIC_FEATURES
            ),
            "categorical_features": categorical_features,
            "all_features": list(X_train.columns),
            "random_state": RANDOM_STATE
        }

        with open(
            METADATA_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=4
            )

        # ----------------------------------------------------
        # ЛОГИРОВАНИЕ МОДЕЛИ В MLFLOW
        # ----------------------------------------------------

        mlflow.catboost.log_model(
            best_model,
            artifact_path="best_model"
        )

        mlflow.log_artifact(
            str(METADATA_PATH)
        )

        # Сохраняем конфигурацию выбранной модели
        print("\n" + "=" * 60)
        print("СОХРАНЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)

        print(f"Модель: {MODEL_PATH}")
        print(f"Совместимая копия: model.cbm")
        print(f"Метаданные: {METADATA_PATH}")

        print("\nПорог классификации:", best_threshold)
        print("Выбранная конфигурация:", best_candidate["name"])

        print("\nОбучение успешно завершено.")


# ============================================================
# 9. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    train_model()
