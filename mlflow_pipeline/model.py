import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow_pipeline.data import load_data

def train_model():
    data = load_data()
    X = data[['income', 'current_loan']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_model()
