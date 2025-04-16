import mlflow.sklearn
import numpy as np

def predict(income, current_loan):
    model = mlflow.sklearn.load_model("models:/loan_model/1")
    prediction = model.predict(np.array([[income, current_loan]]))
    return prediction[0]
