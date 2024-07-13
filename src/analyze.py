# src/analyze.py

import os
import matplotlib.pyplot as plt

def plot_predictions(y_test, predictions, output_path='results/analysis_plots/predictions_vs_actuals.png'):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig(output_path)
    plt.close()

def plot_loss(losses, output_path='results/analysis_plots/training_loss.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    plt.savefig(output_path)
    plt.close()
