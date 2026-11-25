
import json
import re
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKER_PATH = BASE_DIR / "ollama_worker.py"

MODEL_NAME = "qwen3.5:4b"

# Максимальное время ожидания ответа от отдельного worker-процесса.
MAX_GENERATION_TIME = 22

MAX_TOKENS = 65

FEATURE_LABELS = {
    "duration": "срок кредита",
    "credit_amount": "сумма кредита",
    "installment_commitment": "доля дохода на выплаты",
    "residence_since": "срок проживания по текущему адресу",
    "age": "возраст",
    "existing_credits": "количество существующих кредитов",
    "num_dependents": "количество иждивенцев",
    "checking_status": "состояние расчётного счёта",
    "credit_history": "кредитная история",
    "purpose": "цель кредита",
    "savings_status": "объём накоплений",
    "employment": "стаж работы",
    "personal_status": "семейное положение",
    "other_parties": "другие участники кредита",
    "property_magnitude": "имущество",
    "other_payment_plans": "другие планы платежей",
    "housing": "жилищные условия",
    "job": "профессия",
    "own_telephone": "наличие телефона",
    "foreign_worker": "статус иностранного работника",
}

DIRECTION_LABELS = {
    "increases_risk": "связан с повышением оценки риска",
    "decreases_risk": "связан со снижением оценки риска",
    "neutral": "имеет нейтральное влияние",
}


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_intro(probability, threshold, decision):
    probability_percent = probability * 100
    threshold_percent = threshold * 100

    if decision == "approve":
        return (
            f"Модель оценила вероятность дефолта в "
            f"{probability_percent:.2f}%. Это ниже порога "
            f"{threshold_percent:.2f}%, поэтому заявка отнесена "
            f"к классу 0 — низкий риск по классификации модели."
        )

    return (
        f"Модель оценила вероятность дефолта в "
        f"{probability_percent:.2f}%. Это не ниже порога "
        f"{threshold_percent:.2f}%, поэтому заявка отнесена "
        f"к классу 1 — повышенный риск по классификации модели."
    )


def build_factor_text(factors, limit=3):
    if not isinstance(factors, list) or not factors:
        return "Модель не предоставила список факторов для пояснения."

    descriptions = []

    for factor in factors[:limit]:
        feature = factor.get("feature", "неизвестный признак")
        label = FEATURE_LABELS.get(feature, feature)

        direction = factor.get("direction", "neutral")
        direction_text = DIRECTION_LABELS.get(
            direction,
            DIRECTION_LABELS["neutral"],
        )

        value = factor.get("value")

        if value is None:
            descriptions.append(
                f"{label} {direction_text}"
            )
        else:
            descriptions.append(
                f"{label} (значение: {value}) {direction_text}"
            )

    return "Среди факторов, связанных с прогнозом: " + "; ".join(
        descriptions
    ) + "."


def build_counterfactual_text(counterfactual):
    if not counterfactual:
        return ""

    if not counterfactual.get("found"):
        reason = counterfactual.get(
            "reason",
            "В проверенных сценариях подходящий вариант не найден.",
        )

        return (
            "\n\nВарианты изменения заявки: "
            + reason
            + " Это не означает, что другие варианты невозможны."
        )

    best = counterfactual.get("best")

    if not best:
        return ""

    changes = best.get("changes", [])

    if not changes:
        return ""

    change_descriptions = []

    for change in changes:
        label = change.get("label", change.get("feature", "Параметр"))
        old_value = change.get("old_value")
        new_value = change.get("new_value")

        if change.get("feature") == "duration":
            change_descriptions.append(
                f"{label}: {old_value} → {new_value} месяцев"
            )
        elif change.get("feature") == "credit_amount":
            change_descriptions.append(
                f"{label}: {old_value} → {new_value}"
            )

    if not change_descriptions:
        return ""

    probability_percent = best["probability_percent"]

    return (
        "\n\nПроверенный альтернативный сценарий: "
        + "; ".join(change_descriptions)
        + f". В этом сценарии модель оценивает вероятность дефолта "
        f"в {probability_percent:.2f}%, что ниже установленного порога. "
        "Это результат симуляции модели, а не гарантия одобрения кредита."
    )


def build_fallback(
    probability,
    threshold,
    decision,
    factors,
    counterfactual,
):
    intro = build_intro(probability, threshold, decision)
    factor_text = build_factor_text(factors)
    alternative_text = build_counterfactual_text(counterfactual)

    return intro + "\n\n" + factor_text + alternative_text


def generate_qwen_factor_text(factors):
    if not WORKER_PATH.exists():
        print("Файл ollama_worker.py не найден.")
        return None

    # Передаём модели только ограниченный список фактов.
    safe_factors = []

    for factor in (factors or [])[:3]:
        safe_factors.append({
            "feature": FEATURE_LABELS.get(
                factor.get("feature"),
                factor.get("feature", "неизвестный признак"),
            ),
            "value": str(factor.get("value", "не указано")),
            "direction": DIRECTION_LABELS.get(
                factor.get("direction", "neutral"),
                DIRECTION_LABELS["neutral"],
            ),
        })

    system_prompt = (
        "Ты помощник демонстрационной системы кредитного скоринга. "
        "Твоя задача — объяснить факторы прогноза простым русским языком. "
        "Используй только факты из сообщения пользователя. "
        "Не придумывай сведения о заёмщике. "
        "Не делай выводов о причинах или причинно-следственных связях. "
        "Не меняй решение модели, вероятность и порог. "
        "Не предлагай самостоятельно способы улучшить заявку. "
        "Не называй SHAP процентами или процентными пунктами. "
        "Напиши не более двух коротких предложений."
    )

    user_prompt = (
        "Объясни простым языком следующие факторы прогноза. "
        "Не добавляй никаких других сведений:\n"
        + json.dumps(safe_factors, ensure_ascii=False)
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": MAX_TOKENS,
        },
        "think": False,
        "stream": True,
        "keep_alive": "10m",
    }

    start_time = time.time()

    try:
        process = subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=MAX_GENERATION_TIME,
            cwd=str(BASE_DIR),
        )

        elapsed = time.time() - start_time

        if process.stderr:
            print("Ollama worker:", process.stderr.strip())

        if process.returncode != 0:
            print(
                "Worker завершился с ошибкой:",
                process.returncode,
            )
            return None

        if not process.stdout.strip():
            print("Worker вернул пустой ответ.")
            return None

        result = json.loads(process.stdout)

        if not result.get("success"):
            print(
                "Ошибка генерации Qwen:",
                result.get("error", "Неизвестная ошибка"),
            )
            return None

        answer = clean_text(result.get("text", ""))

        if len(answer.split()) < 4:
            print("Ответ Qwen слишком короткий.")
            return None

        print(
            f"Qwen сформировала пояснение за {elapsed:.2f} сек."
        )

        return answer

    except subprocess.TimeoutExpired:
        print(
            f"Превышено время генерации: "
            f"{MAX_GENERATION_TIME} секунд."
        )
        return None

    except json.JSONDecodeError:
        print("Worker вернул некорректный JSON.")
        return None

    except Exception as error:
        print(f"Ошибка при обращении к Qwen: {error}")
        return None


def generate_explanation(
    probability,
    threshold,
    decision,
    factors,
    counterfactual=None,
):
    """
    Возвращает:
    {
        "text": "...",
        "source": "qwen" или "fallback"
    }
    """

    fallback = build_fallback(
        probability=probability,
        threshold=threshold,
        decision=decision,
        factors=factors,
        counterfactual=counterfactual,
    )

    qwen_text = generate_qwen_factor_text(factors)

    if qwen_text:
        # Точные значения и проверенные рекомендации всегда формирует Python.
        intro = build_intro(probability, threshold, decision)
        alternative_text = build_counterfactual_text(counterfactual)

        final_text = (
            intro
            + "\n\n"
            + qwen_text
            + alternative_text
        )

        return {
            "text": final_text,
            "source": "qwen",
        }

    return {
        "text": fallback,
        "source": "fallback",
    }
