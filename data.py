from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_split():
    print("Загрузка данных...")
    X, y = fetch_openml("credit-g", version=1, as_frame=True, return_X_y=True)
    y = y.map({"good": 0, "bad": 1})
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Данные загружены. Размер train: {X_train.shape}, test: {y_test.shape}")