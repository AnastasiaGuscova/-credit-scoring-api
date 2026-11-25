
from copy import deepcopy

import pandas as pd

from data import prepare_features


# Диапазоны поиска соответствуют диапазонам обучающего German Credit Dataset.
MIN_DURATION = 4
MAX_DURATION = 72

# Проверяем уменьшение суммы кредита от 5% до 50% с шагом 5%.
MIN_AMOUNT_REDUCTION = 0.05
MAX_AMOUNT_REDUCTION = 0.50
AMOUNT_STEP = 0.05


FEATURE_LABELS = {
    "credit_amount": "Сумма кредита",
    "duration": "Срок кредита",
}


def _get_default_probability(model, prepared_df):
    """Получает вероятность класса 1 — дефолта."""
    probabilities = model.predict_proba(prepared_df)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("В модели не найден класс 1 (default).")

    default_index = classes.index(1)

    return float(probabilities[0][default_index])


def _build_candidates(original_features):
    """Формирует допустимые варианты изменения суммы и срока."""
    original_amount = float(original_features["credit_amount"])
    original_duration = int(original_features["duration"])

    if original_amount <= 0:
        raise ValueError("Сумма кредита должна быть больше нуля.")

    candidates = []
    seen = set()

    def add_candidate(amount, duration):
        amount = int(round(amount))
        duration = int(round(duration))

        if amount <= 0:
            return

        if duration < MIN_DURATION or duration > MAX_DURATION:
            return

        key = (amount, duration)

        if key in seen:
            return

        if amount == round(original_amount) and duration == original_duration:
            return

        seen.add(key)

        candidate = deepcopy(original_features)
        candidate["credit_amount"] = amount
        candidate["duration"] = duration

        amount_change = abs(amount - original_amount) / original_amount
        duration_change = (
            abs(duration - original_duration) /
            max(abs(original_duration), 1)
        )

        # Чем меньше относительное изменение, тем ниже стоимость сценария.
        change_cost = amount_change + duration_change

        changed_fields = []

        if amount != round(original_amount):
            changed_fields.append({
                "feature": "credit_amount",
                "label": FEATURE_LABELS["credit_amount"],
                "old_value": round(original_amount),
                "new_value": amount,
            })

        if duration != original_duration:
            changed_fields.append({
                "feature": "duration",
                "label": FEATURE_LABELS["duration"],
                "old_value": original_duration,
                "new_value": duration,
            })

        candidates.append({
            "features": candidate,
            "amount": amount,
            "duration": duration,
            "change_cost": change_cost,
            "change_count": len(changed_fields),
            "changes": changed_fields,
        })

    # 1. Варианты с уменьшением только суммы кредита.
    reduction = MIN_AMOUNT_REDUCTION

    while reduction <= MAX_AMOUNT_REDUCTION + 1e-9:
        new_amount = original_amount * (1 - reduction)
        add_candidate(new_amount, original_duration)
        reduction += AMOUNT_STEP

    # 2. Варианты с изменением только срока кредита.
    for duration in range(MIN_DURATION, MAX_DURATION + 1):
        add_candidate(original_amount, duration)

    # 3. Сочетания уменьшения суммы и изменения срока.
    reduction = MIN_AMOUNT_REDUCTION

    while reduction <= MAX_AMOUNT_REDUCTION + 1e-9:
        new_amount = original_amount * (1 - reduction)

        for duration in range(MIN_DURATION, MAX_DURATION + 1):
            add_candidate(new_amount, duration)

        reduction += AMOUNT_STEP

    return candidates


def find_counterfactual(
    original_features,
    model,
    threshold,
    max_alternatives=3
):
    """
    Ищет варианты заявки, при которых вероятность дефолта
    становится строго меньше порога классификации.

    Исходные данные не изменяются.
    """

    original_features = deepcopy(original_features)

    original_df = pd.DataFrame([original_features])
    original_prepared = prepare_features(original_df)

    original_probability = _get_default_probability(
        model,
        original_prepared
    )

    result = {
        "found": False,
        "original_probability": round(original_probability, 4),
        "original_probability_percent": round(
            original_probability * 100, 2
        ),
        "threshold": threshold,
        "alternatives": [],
        "best": None,
        "search_limits": {
            "amount_reduction_min_percent": 5,
            "amount_reduction_max_percent": 50,
            "amount_reduction_step_percent": 5,
            "duration_min_months": MIN_DURATION,
            "duration_max_months": MAX_DURATION,
        },
    }

    # Если модель уже относит заявку к классу 0,
    # искать альтернативный сценарий не требуется.
    if original_probability < threshold:
        result["reason"] = "Заявка уже находится ниже порога классификации."
        return result

    candidates = _build_candidates(original_features)

    if not candidates:
        result["reason"] = "Не удалось сформировать варианты заявки."
        return result

    # Готовим все варианты одним DataFrame.
    candidates_df = pd.DataFrame([
        candidate["features"] for candidate in candidates
    ])

    prepared_candidates = prepare_features(candidates_df)

    # Один пакетный вызов модели вместо отдельного вызова для каждого варианта.
    probabilities = model.predict_proba(prepared_candidates)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("В модели не найден класс 1 (default).")

    default_index = classes.index(1)

    successful = []

    for candidate, probability_row in zip(candidates, probabilities):
        probability = float(probability_row[default_index])

        # Важно: логика должна совпадать с Flask:
        # approve только при probability < threshold.
        if probability < threshold:
            successful.append({
                "probability": round(probability, 4),
                "probability_percent": round(probability * 100, 2),
                "change_cost": round(candidate["change_cost"], 6),
                "change_count": candidate["change_count"],
                "changes": candidate["changes"],
            })

    if not successful:
        result["reason"] = (
            "В проверенном диапазоне не найден вариант, "
            "при котором вероятность дефолта становится ниже порога."
        )
        return result

    # Сначала минимальное изменение, затем меньшее число изменений,
    # затем меньшая вероятность дефолта.
    successful.sort(
        key=lambda item: (
            item["change_cost"],
            item["change_count"],
            item["probability"],
        )
    )

    alternatives = successful[:max_alternatives]

    result["found"] = True
    result["alternatives"] = alternatives
    result["best"] = alternatives[0]

    result["reason"] = (
        "Найдены варианты, при которых прогнозируемая "
        "вероятность дефолта ниже порога классификации."
    )

    return result
