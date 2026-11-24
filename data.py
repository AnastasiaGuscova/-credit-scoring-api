
"""
Подготовка данных для модели кредитного скоринга.

Источник данных: German Credit (credit-g), OpenML.

Целевой класс:
    0 — good: хороший заёмщик;
    1 — bad: плохой заёмщик (дефолт).

Разделение данных:
    60% — train;
    20% — validation;
    20% — test.
"""

import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

TARGET_MAPPING = {
    "good": 0,
    "bad": 1
}

# Числовые признаки исходного датасета
NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

# Категориальные признаки исходного датасета
CATEGORICAL_FEATURES = [
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
]

# Признаки, которые будут созданы дополнительно
ENGINEERED_NUMERIC_FEATURES = [
    "credit_amount_per_month_proxy"
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "age_group"
]

MISSING_CATEGORY = "Unknown"


# ============================================================
# 2. ЗАГРУЗКА ДАТАСЕТА
# ============================================================

def load_dataset():
    """
    Загружает German Credit Dataset из OpenML.

    Возвращает:
        X — исходные признаки;
        y — целевая переменная.
    """

    print("Загрузка German Credit Dataset...")

    dataset = fetch_openml(
        name="credit-g",
        version=1,
        as_frame=True,
        return_X_y=True
    )

    X, y = dataset

    # Приводим названия признаков к единому формату
    X.columns = X.columns.astype(str)

    # Убираем строки, в которых отсутствует целевой класс
    valid_target = y.notna()

    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    # Нормализуем значения целевого класса
    y = y.astype(str).str.strip().str.lower()

    # Преобразуем good/bad в 0/1
    y = y.map(TARGET_MAPPING)

    if y.isna().any():
        raise ValueError(
            "В целевой переменной обнаружены неизвестные значения."
        )

    y = y.astype(int)

    print(f"Датасет загружен: {X.shape[0]} объектов.")
    print(f"Количество исходных признаков: {X.shape[1]}")

    return X, y


# ============================================================
# 3. ПРОВЕРКА И ОЧИСТКА ДАННЫХ
# ============================================================

def clean_data(X):
    """
    Проверяет структуру данных и обрабатывает пропуски.

    Числовые пропуски оставляются как NaN:
    CatBoost умеет обрабатывать их самостоятельно.

    Категориальные пропуски заменяются на Unknown,
    поскольку CatBoost ожидает строковые значения
    в категориальных признаках.
    """

    X = X.copy()

    # Проверяем наличие всех ожидаемых исходных признаков
    expected_features = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    missing_columns = [
        column
        for column in expected_features
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"В датасете отсутствуют признаки: {missing_columns}"
        )

    # Оставляем признаки в заранее определённом порядке
    X = X[expected_features].copy()

    # Обрабатываем числовые признаки
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Обрабатываем категориальные признаки
    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna(MISSING_CATEGORY)
            .astype(str)
        )

    # Выводим информацию о пропусках
    missing_count = int(X.isna().sum().sum())

    print(f"Пропусков после обработки категорий: {missing_count}")

    return X


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def create_features(X):
    """
    Создаёт дополнительные признаки на основе исходных данных.

    Все преобразования детерминированные:
    они не используют статистики всей выборки,
    поэтому не создают утечку данных между train/test.
    """

    X = X.copy()

    # --------------------------------------------------------
    # Признак 1. Отношение суммы кредита к сроку
    # --------------------------------------------------------
    #
    # Показывает, какая часть суммы кредита приходится
    # на один месяц срока кредитования.
    #
    # Это приблизительный показатель, а не реальный
    # ежемесячный платёж: проценты и комиссии не учитываются.

    X["credit_amount_per_month_proxy"] = (
        X["credit_amount"] / X["duration"].where(
            X["duration"] > 0
        )
    )

    # --------------------------------------------------------
    # Признак 2. Возрастная группа
    # --------------------------------------------------------
    #
    # Дополнительное категориальное представление возраста.

    X["age_group"] = pd.cut(
        X["age"],
        bins=[0, 25, 35, 50, float("inf")],
        labels=[
            "young",
            "adult",
            "mature",
            "senior"
        ],
        right=True
    )

    X["age_group"] = (
        X["age_group"]
        .astype("string")
        .fillna(MISSING_CATEGORY)
        .astype(str)
    )

    return X


# ============================================================
# 5. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================

def prepare_features(X):
    """
    Выполняет очистку и Feature Engineering.

    Возвращает подготовленную таблицу признаков.
    """

    X = clean_data(X)

    X = create_features(X)

    return X


# ============================================================
# 6. РАЗДЕЛЕНИЕ НА TRAIN / VALIDATION / TEST
# ============================================================

def load_and_split():
    """
    Загружает, очищает и разделяет данные.

    Итоговое соотношение:
        Train:      60%
        Validation: 20%
        Test:       20%

    Стратификация сохраняет пропорции целевых классов
    в каждой выборке.

    Возвращает:
        X_train, X_val, X_test,
        y_train, y_val, y_test
    """

    # Загружаем исходные данные
    X, y = load_dataset()

    # Проверяем и создаём признаки
    X = prepare_features(X)

    # Сначала отделяем тестовую выборку (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Из оставшихся 80% выделяем 25% на validation.
    # Получаем итоговое распределение 60/20/20.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )

    # Сбрасываем индексы для удобства дальнейшей работы
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Выводим информацию о разделении
    print("\n" + "=" * 50)
    print("ИНФОРМАЦИЯ О ВЫБОРКАХ")
    print("=" * 50)

    print(f"Train:      {X_train.shape[0]} объектов")
    print(f"Validation: {X_val.shape[0]} объектов")
    print(f"Test:       {X_test.shape[0]} объектов")

    print("\nРаспределение классов:")

    for name, target in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test)
    ]:
        distribution = (
            target.value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

        print(f"\n{name}:")
        print(f"Хорошие заёмщики (0): {distribution.get(0, 0)}%")
        print(f"Плохие заёмщики (1): {distribution.get(1, 0)}%")

    print("\nКоличество признаков после Feature Engineering:")
    print(X_train.shape[1])

    print("\nКатегориальные признаки:")
    print(
        CATEGORICAL_FEATURES
        + ENGINEERED_CATEGORICAL_FEATURES
    )

    print("\nПодготовка данных завершена.")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# 7. ПРОВЕРКА ПРИ ЗАПУСКЕ ФАЙЛА
# ============================================================

if __name__ == "__main__":

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = load_and_split()

    print("\nПроверка типов данных:")
    print(X_train.dtypes)

    print("\nПервые пять строк обучающей выборки:")
    print(X_train.head())
