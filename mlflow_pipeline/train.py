import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    mlflow.set_experiment("loan-mlops")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model")
