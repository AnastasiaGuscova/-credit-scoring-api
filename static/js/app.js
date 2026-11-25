
document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("creditForm");
    const resultCard = document.getElementById("resultCard");
    const errorMessage = document.getElementById("errorMessage");
    const submitButton = document.getElementById("submitButton");

    const numericFields = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents"
    ];

    const categoricalFields = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker"
    ];

    // Получение данных из формы
    function getFormData() {
        const features = {};

        numericFields.forEach(field => {
            const input = document.getElementById(field);

            features[field] = Number(input.value);
        });

        categoricalFields.forEach(field => {
            const input = document.getElementById(field);

            features[field] = input.value;
        });

        return features;
    }

    // Отображение ошибки
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove("hidden");
    }

    // Отображение факторов SHAP
    function renderFactors(factors) {
        const factorList = document.getElementById("factorList");

        factorList.replaceChildren();

        if (!Array.isArray(factors) || factors.length === 0) {
            const item = document.createElement("li");

            item.textContent = "Факторы не найдены.";

            factorList.appendChild(item);
            return;
        }

        const directionLabels = {
            increases_risk: "Повышает риск",
            decreases_risk: "Снижает риск",
            neutral: "Нейтральное влияние"
        };

        factors.forEach(factor => {
            const item = document.createElement("li");
            item.className = "factor-item";

            const name = document.createElement("div");
            name.className = "factor-name";
            name.textContent = factor.feature ?? "Признак";

            const details = document.createElement("div");
            details.className = "factor-details";

            const impact = Number(factor.impact);

            const impactText = Number.isFinite(impact)
                ? impact.toFixed(4)
                : "Не рассчитан";

            const direction = factor.direction ?? "neutral";

            const directionElement = document.createElement("span");

            if (direction === "increases_risk") {
                directionElement.className = "factor-risk";
            } else if (direction === "decreases_risk") {
                directionElement.className = "factor-safe";
            }

            directionElement.textContent =
                directionLabels[direction] ?? direction;

            details.textContent =
                `Значение: ${factor.value ?? "Не указано"} | ` +
                `SHAP: ${impactText} | `;

            details.appendChild(directionElement);

            item.appendChild(name);
            item.appendChild(details);

            factorList.appendChild(item);
        });
    }

    // Отображение контрфактических сценариев
    function renderCounterfactual(counterfactual) {
        const box = document.getElementById("counterfactualBox");
        const description = document.getElementById(
            "counterfactualDescription"
        );
        const scenarioList = document.getElementById("scenarioList");

        scenarioList.replaceChildren();

        if (!counterfactual) {
            box.classList.add("hidden");
            return;
        }

        box.classList.remove("hidden");

        if (!counterfactual.found) {
            description.textContent =
                counterfactual.reason ||
                "В проверенном диапазоне не найден подходящий вариант.";

            return;
        }

        description.textContent =
            "Ниже представлены найденные варианты изменения параметров заявки, " +
            "при которых прогнозируемая вероятность дефолта становится ниже порога.";

        const alternatives = counterfactual.alternatives || [];

        if (alternatives.length === 0) {
            description.textContent =
                "Альтернативные сценарии не найдены.";
            return;
        }

        alternatives.forEach((alternative, index) => {
            const item = document.createElement("li");
            item.className = "scenario-item";

            const title = document.createElement("div");
            title.className = "scenario-title";
            title.textContent = `Вариант ${index + 1}`;

            const details = document.createElement("div");
            details.className = "scenario-details";

            const changes = Array.isArray(alternative.changes)
                ? alternative.changes
                : [];

            if (changes.length === 0) {
                details.textContent = "Изменения не указаны.";
            } else {
                changes.forEach(change => {
                    const line = document.createElement("div");

                    line.textContent =
                        `${change.label}: ${change.old_value} → ${change.new_value}`;

                    details.appendChild(line);
                });
            }

            const probability = document.createElement("div");
            probability.className = "scenario-probability";

            const probabilityPercent = Number(
                alternative.probability_percent
            );

            probability.textContent = Number.isFinite(probabilityPercent)
                ? `Вероятность дефолта: ${probabilityPercent.toFixed(2)}%`
                : "Вероятность не указана";

            item.appendChild(title);
            item.appendChild(details);
            item.appendChild(probability);

            scenarioList.appendChild(item);
        });
    }

    // Отображение итогового результата
    function renderResult(data) {
        const decision = document.getElementById("decision");
        const probability = document.getElementById("probability");
        const progressBar = document.getElementById("progressBar");
        const thresholdInfo = document.getElementById("thresholdInfo");

        const probabilityPercent = Number(data.probability_percent);

        decision.textContent = data.decision === "approve"
            ? "Ниже порога риска"
            : "Порог риска достигнут";

        decision.className =
            "decision " +
            (data.decision === "approve" ? "approve" : "decline");

        probability.textContent =
            Number.isFinite(probabilityPercent)
                ? `${probabilityPercent.toFixed(2)}%`
                : "Не рассчитана";

        progressBar.style.width =
            `${Math.min(100, Math.max(0, probabilityPercent || 0))}%`;

        thresholdInfo.textContent =
            `Порог классификации: ${(Number(data.threshold) * 100).toFixed(0)}%. ` +
            "Если вероятность дефолта ниже порога, модель относит заявку " +
            "к классу 0. В противном случае — к классу 1.";

        renderFactors(data.top_factors);

        // Аналитическое заключение
        const llmExplanation = document.getElementById("llmExplanation");
        const llmStatus = document.getElementById("llmStatus");

        llmExplanation.textContent =
            data.llm_explanation ||
            "Аналитическое заключение не сформировано.";

        if (data.llm_source === "qwen") {
            llmStatus.textContent =
                "Сформировано языковой моделью Qwen 3.5 4B.";
        } else {
            llmStatus.textContent =
                "Использовано резервное программное объяснение.";
        }

        // Альтернативные сценарии
        renderCounterfactual(data.counterfactual);

        resultCard.classList.remove("hidden");

        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    // Отправка формы на Flask
    form.addEventListener("submit", async function(event) {
        event.preventDefault();

        errorMessage.classList.add("hidden");
        resultCard.classList.add("hidden");

        if (!form.reportValidity()) {
            return;
        }

        const features = getFormData();

        submitButton.disabled = true;
        submitButton.textContent =
            "Оценка кредитного риска и подготовка заключения...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    features: features
                })
            });

            const data = await response.json();

            if (!response.ok || data.status !== "success") {
                throw new Error(
                    data.message || "Не удалось выполнить оценку."
                );
            }

            renderResult(data);

        } catch (error) {
            showError(error.message);
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = "Оценить кредитный риск";
        }
    });

});
