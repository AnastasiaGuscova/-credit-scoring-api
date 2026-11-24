import shap
import pandas as pd
from catboost import CatBoostClassifier


class ModelExplainer:
    def __init__(self, model_path="model.cbm"):
        # Загружаем обученную модель
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        # Создаем SHAP-объяснитель для деревьев (CatBoost — это деревья)
        self.explainer = shap.TreeExplainer(self.model)

    def get_top_factors(self, input_df: pd.DataFrame, top_n: int = 3):
        """
        Возвращает топ-N признаков, которые сильнее всего повлияли на решение.
        """
        # Считаем SHAP-значения для одного клиента
        shap_values = self.explainer.shap_values(input_df)

        # Берем значения для класса 1 (дефолт)
        vals = shap_values[0]

        # Сортируем признаки по модулю влияния (от большего к меньшему)
        factors = sorted(
            zip(input_df.columns, vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        # Формируем красивый ответ
        result = []
        for feature, impact in factors:
            result.append({
                "feature": feature,
                "impact": round(float(impact), 4),
                "direction": "increases_risk" if impact > 0 else "decreases_risk"
            })
        return result