# src/evaluate.py

from sklearn.metrics import mean_squared_error
import sys
sys.path.append('..\\src')

from model import CoTModel
from utils import load_data
import torch

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs, cot = model(X_test)
        loss = mean_squared_error(y_test.numpy(), outputs.numpy())
        print(f'Test MSE: {loss:.4f}')
    return outputs, cot

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = CoTModel()
    evaluate_model(model, X_test, y_test)
