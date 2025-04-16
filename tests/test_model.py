from mlflow_pipeline.model import train_model

def test_training_runs():
    try:
        train_model()
    except Exception as e:
        assert False, f"Training failed: {e}"
