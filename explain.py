
"""
SHAP-объяснение предсказаний модели кредитного скоринга.

Модуль:
1. Загружает обученную модель CatBoost.
2. Подготавливает входные данные в соответствии с data.py.
3. Вычисляет SHAP-значения для конкретного заёмщика.
4. Определяет направление влияния каждого признака.
5. Возвращает наиболее значимые факторы.

Положительное SHAP-значение:
признак увеличивает предсказание в сторону класса 1 (дефолт).

Отрицательное SHAP-значение:
признак увеличивает предсказание в сторону класса 0 (хороший заёмщик).

Важно:
SHAP показывает вклад признака в предсказание модели,
а не доказывает причинно-следственную связь.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool

from data import (
    prepare_features,
    CATEGORICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
)


# ============================================================
# КОНСТАНТЫ
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    ROOT_DIR / "models" / "model.cbm",
    ROOT_DIR / "model.cbm",
]

DEFAULT_TOP_N = 5

RISK_CLASS = 1
SAFE_CLASS = 0


# ============================================================
# КЛАСС ОБЪЯСНЕНИЯ МОДЕЛИ
# ============================================================

class ModelExplainer:
    """
    Класс для получения SHAP-объяснений модели CatBoost.

    Parameters
    ----------
    model : CatBoostClassifier | str | Path | None
        Обученная модель или путь к файлу модели.
        Если аргумент не передан, модель загружается автоматически.
    """

    def __init__(
        self,
        model: Union[CatBoostClassifier, str, Path, None] = None
    ):
        self.model = self._load_model(model)

        self.feature_names = list(self.model.feature_names_)

        if not self.feature_names:
            raise ValueError(
                "У модели отсутствуют названия признаков. "
                "Проверьте, что модель обучалась на pandas DataFrame."
            )

        self.categorical_features = (
            CATEGORICAL_FEATURES
            + ENGINEERED_CATEGORICAL_FEATURES
        )

        self._validate_model_features()

    # --------------------------------------------------------
    # ЗАГРУЗКА МОДЕЛИ
    # --------------------------------------------------------

    @staticmethod
    def _load_model(
        model: Union[CatBoostClassifier, str, Path, None]
    ) -> CatBoostClassifier:

        # Если передан уже загруженный объект модели
        if isinstance(model, CatBoostClassifier):
            return model

        # Если передан путь к модели
        if isinstance(model, (str, Path)):
            model_path = Path(model)

        # Если модель не передана, ищем её автоматически
        else:
            model_path = next(
                (
                    path
                    for path in MODEL_PATHS
                    if path.exists()
                ),
                None
            )

            if model_path is None:
                raise FileNotFoundError(
                    "Не найдена обученная модель CatBoost. "
                    "Сначала запустите train.py."
                )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_path}"
            )

        loaded_model = CatBoostClassifier()
        loaded_model.load_model(str(model_path))

        print(f"Модель для SHAP загружена: {model_path}")

        return loaded_model

    # --------------------------------------------------------
    # ПРОВЕРКА ПРИЗНАКОВ
    # --------------------------------------------------------

    def _validate_model_features(self) -> None:
        """
        Проверяет, что названия признаков модели
        соответствуют признакам, подготовленным в data.py.
        """

        expected_categorical = set(self.categorical_features)

        actual_categorical = (
            expected_categorical.intersection(
                set(self.feature_names)
            )
        )

        if actual_categorical != expected_categorical:
            missing = (
                expected_categorical - actual_categorical
            )

            raise ValueError(
                "Не удалось сопоставить категориальные признаки "
                f"модели. Отсутствуют: {sorted(missing)}"
            )

    # --------------------------------------------------------
    # ПОДГОТОВКА ВХОДНЫХ ДАННЫХ
    # --------------------------------------------------------

    def _prepare_input(
        self,
        input_data: Union[pd.DataFrame, dict]
    ) -> pd.DataFrame:
        """
        Приводит входные данные к формату,
        который использовался при обучении модели.

        Принимает:
        - исходные признаки (20 столбцов);
        - уже подготовленные признаки (22 столбца).
        """

        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])

        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()

        else:
            raise TypeError(
                "Входные данные должны быть pandas DataFrame или dict."
            )

        if input_df.empty:
            raise ValueError(
                "Переданы пустые входные данные."
            )

        if len(input_df) != 1:
            raise ValueError(
                "Для объяснения необходимо передать данные "
                "ровно одного заёмщика."
            )

        # Если уже переданы все признаки модели,
        # повторная инженерия признаков не требуется.
        if set(self.feature_names).issubset(input_df.columns):

            prepared_df = input_df[self.feature_names].copy()

        else:
            # Если переданы исходные признаки,
            # используем ту же подготовку, что и при обучении.
            prepared_df = prepare_features(input_df)

            missing_features = (
                set(self.feature_names) - set(prepared_df.columns)
            )

            if missing_features:
                raise ValueError(
                    "После подготовки данных отсутствуют признаки: "
                    f"{sorted(missing_features)}"
                )

            prepared_df = prepared_df[self.feature_names].copy()

        # Приводим категориальные признаки к строковому типу
        for feature in self.categorical_features:
            if feature in prepared_df.columns:
                prepared_df[feature] = (
                    prepared_df[feature]
                    .fillna("Unknown")
                    .astype(str)
                )

        # Приводим числовые признаки к числовому типу
        numeric_features = [
            feature
            for feature in self.feature_names
            if feature not in self.categorical_features
        ]

        for feature in numeric_features:
            prepared_df[feature] = pd.to_numeric(
                prepared_df[feature],
                errors="coerce"
            )

        return prepared_df

    # --------------------------------------------------------
    # ВЫЧИСЛЕНИЕ SHAP-ЗНАЧЕНИЙ
    # --------------------------------------------------------

    def _calculate_shap_values(
        self,
        input_df: pd.DataFrame
    ) -> tuple:
        """
        Вычисляет SHAP-значения с помощью CatBoost.

        Последний элемент массива SHAP — базовое значение
        предсказания модели (expected value).

        Остальные элементы соответствуют признакам.
        """

        categorical_features = [
            feature
            for feature in self.categorical_features
            if feature in input_df.columns
        ]

        pool = Pool(
            data=input_df,
            cat_features=categorical_features,
            feature_names=self.feature_names
        )

        shap_matrix = self.model.get_feature_importance(
            data=pool,
            type="ShapValues"
        )

        shap_matrix = np.asarray(shap_matrix)

        if shap_matrix.ndim != 2 or shap_matrix.shape[0] != 1:
            raise ValueError(
                "Неожиданный формат SHAP-значений. "
                f"Получена форма массива: {shap_matrix.shape}"
            )

        if shap_matrix.shape[1] != len(self.feature_names) + 1:
            raise ValueError(
                "Количество SHAP-значений не совпадает "
                "с количеством признаков модели."
            )

        feature_shap_values = shap_matrix[0, :-1]
        base_value = float(shap_matrix[0, -1])

        return feature_shap_values, base_value

    # --------------------------------------------------------
    # ФОРМИРОВАНИЕ ОБЪЯСНЕНИЙ
    # --------------------------------------------------------

    def explain(
        self,
        input_data: Union[pd.DataFrame, dict],
        top_n: int = DEFAULT_TOP_N
    ) -> list:
        """
        Возвращает наиболее значимые факторы
        для предсказания конкретного заёмщика.

        Parameters
        ----------
        input_data : pandas DataFrame | dict
            Данные одного заёмщика.

        top_n : int
            Количество факторов, которые необходимо вернуть.

        Returns
        -------
        list[dict]
            Список факторов с их значениями,
            SHAP-вкладом и направлением влияния.
        """

        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError(
                "Параметр top_n должен быть положительным целым числом."
            )

        prepared_df = self._prepare_input(input_data)

        shap_values, _ = self._calculate_shap_values(
            prepared_df
        )

        factors = []

        for index, feature in enumerate(self.feature_names):

            shap_value = float(shap_values[index])

            feature_value = prepared_df.iloc[0][feature]

            # Определяем направление влияния на риск дефолта
            if shap_value > 0:
                direction = "increases_risk"

            elif shap_value < 0:
                direction = "decreases_risk"

            else:
                direction = "neutral"

            # Формируем описание одного признака
            factors.append({
                "feature": feature,
                "value": self._format_feature_value(
                    feature_value
                ),
                "impact": round(abs(shap_value), 6),
                "shap_value": round(shap_value, 6),
                "direction": direction
            })

        # Сортируем признаки по абсолютной величине SHAP
        factors.sort(
            key=lambda item: item["impact"],
            reverse=True
        )

        return factors[:top_n]

    # --------------------------------------------------------
    # ПОЛНОЕ ОБЪЯСНЕНИЕ
    # --------------------------------------------------------

    def explain_detailed(
        self,
        input_data: Union[pd.DataFrame, dict],
        top_n: int = DEFAULT_TOP_N
    ) -> dict:
        """
        Возвращает расширенную информацию:
        SHAP-факторы, базовое значение и итоговую вероятность
        дефолта, рассчитанную моделью.

        Вероятность берётся непосредственно из predict_proba,
        а не вычисляется из SHAP самостоятельно.
        """

        prepared_df = self._prepare_input(input_data)

        shap_values, base_value = self._calculate_shap_values(
            prepared_df
        )

        probabilities = self.model.predict_proba(prepared_df)

        classes = list(self.model.classes_)

        if RISK_CLASS not in classes:
            raise ValueError(
                "В модели отсутствует класс 1 (дефолт)."
            )

        risk_class_index = classes.index(RISK_CLASS)

        default_probability = float(
            probabilities[0, risk_class_index]
        )

        factors = self.explain(
            prepared_df,
            top_n=top_n
        )

        return {
            "default_probability": round(
                default_probability, 6
            ),
            "base_value": round(base_value, 6),
            "top_factors": factors
        }

    # --------------------------------------------------------
    # ФОРМАТИРОВАНИЕ ЗНАЧЕНИЙ ПРИЗНАКОВ
    # --------------------------------------------------------

    @staticmethod
    def _format_feature_value(value):
        """
        Преобразует значения признаков
        в удобный для API формат.
        """

        if pd.isna(value):
            return None

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        if isinstance(value, (int, float)):
            return value

        return str(value)


# ============================================================
# ПРОВЕРКА РАБОТЫ МОДУЛЯ
# ============================================================

if __name__ == "__main__":

    print("\nПРОВЕРКА SHAP-ОБЪЯСНЕНИЙ")
    print("=" * 60)

    explainer = ModelExplainer()

    # Пример исходных данных одного заёмщика.
    # Значения взяты из категорий German Credit Dataset.
    example_client = {
        "checking_status": "<0",
        "duration": 12,
        "credit_history": "existing paid",
        "purpose": "radio/TV",
        "credit_amount": 2500,
        "savings_status": "500<=X<1000",
        "employment": "1<=X<4",
        "installment_commitment": 2,
        "personal_status": "male single",
        "other_parties": "none",
        "residence_since": 3,
        "property_magnitude": "car",
        "age": 30,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 1,
        "job": "skilled",
        "num_dependents": 1,
        "own_telephone": "yes",
        "foreign_worker": "yes"
    }

    result = explainer.explain_detailed(
        example_client,
        top_n=5
    )

    print(
        "\nВероятность дефолта:",
        f"{result['default_probability']:.2%}"
    )

    print("\nНаиболее значимые факторы:")

    for factor in result["top_factors"]:

        direction = {
            "increases_risk": "Повышает риск",
            "decreases_risk": "Снижает риск",
            "neutral": "Не влияет"
        }.get(factor["direction"], "Неизвестно")

        print(
            f"- {factor['feature']}: "
            f"{factor['value']} | "
            f"SHAP = {factor['shap_value']:+.4f} | "
            f"{direction}"
        )

    print("\nПроверка завершена.")

