import pandas as pd

def load_data():
    # Simulated dataset
    return pd.DataFrame({
        'income': [3000, 4000, 5000],
        'current_loan': [1000, 1500, 2000],
        'label': [1, 0, 1]
    })
