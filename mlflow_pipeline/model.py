import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_model():
    # Dummy data with binary classification
    X = np.array([
        [3000, 1000],
        [5000, 2000],
        [7000, 3000],
        [4000, 1500],
        [2000, 800]
    ])
    y = np.array([1, 0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log model and metrics to MLflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    return model