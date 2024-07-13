# main.py

import os
from src.model import CoTModel
from src.train import train_model
from src.evaluate import evaluate_model
from src.analyze import plot_predictions, plot_loss
from src.utils import load_data

# Create results directories
os.makedirs('results/analysis_plots', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)

# Load data
X_train, y_train, X_test, y_test = load_data()

# Initialize and train model
model = CoTModel()
losses = train_model(model, X_train, y_train)

# Evaluate model
outputs, cot = evaluate_model(model, X_test, y_test)
print("Chain-of-Thought Reasoning:")
for step in cot[:3]:  # Print first 3 CoT steps as an example
    print(step)

# Analyze results
plot_predictions(y_test.numpy(), outputs.numpy())
plot_loss(losses)
