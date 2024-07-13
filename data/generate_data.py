# data/generate_data.py

import numpy as np
import pandas as pd

def generate_math_data(num_samples=1000):
    np.random.seed(42)
    X = np.random.rand(num_samples, 10) * 10  # 10 features
    y = np.sum(X, axis=1)  # Simple sum of all features as the target
    return X, y

def save_data(X, y, filepath='data/math_data.csv'):
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    data['target'] = y
    data.to_csv(filepath, index=False)

if __name__ == "__main__":
    X, y = generate_math_data()
    save_data(X, y)
