import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from data import load_and_split

def train_model():
    print("Загрузка данных...")
    X_train, X_test, y_train, y_test = load_and_split()

    # Находим категориальные признаки (текстовые колонки, например, 'purpose', 'housing')
    # CatBoost умеет работать с ними напрямую, без One-Hot Encoding
    cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Запускаем трекинг в MLflow (он создаст папку mlruns в проекте)
    with mlflow.start_run():
        print("Обучение модели CatBoost...")

        # Инициализируем модель
        model = CatBoostClassifier(
            iterations=300,       # Количество деревьев
            learning_rate=0.05,   # Скорость обучения
            verbose=50,           # Печатать прогресс каждые 50 итераций
            cat_features=cat_features,
            random_seed=42
        )

        # Обучаем
        model.fit(X_train, y_train)

        # Делаем предсказания на тестовой (экзаменационной) выборке
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Берем вероятность дефолта (класс 1)

        # Считаем банковские метрики
        auc = roc_auc_score(y_test, y_pred_proba)
        gini = 2 * auc - 1

        print(f"\n--- Результаты ---")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Gini: {gini:.4f}")

        # Логируем метрики и модель в MLflow
        mlflow.log_param("iterations", 300)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("gini", gini)
        mlflow.catboost.log_model(model, "model")

        # Сохраняем модель локально в файл, чтобы наш API мог её использовать
        model.save_model("model.cbm")
        print("Модель успешно сохранена в model.cbm")

if __name__ == "__main__":
    train_model()