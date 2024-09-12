import pandas as pd

class MedicalDataAgent:
    def load_medical_data(self, filepath):
        # Load medical data from CSV or other sources
        self.data = pd.read_csv(filepath)
        return self.data

    def preprocess_data(self):
        # Example preprocessing
        self.data = self.data.fillna(method='ffill')  # Forward fill missing values
        return self.data
