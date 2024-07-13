# src/train.py

import os
import torch.optim as optim
import sys
sys.path.append('..\\src')
from model import CoTModel
from utils import load_data
import torch.nn as nn

def train_model(model, X_train, y_train, num_epochs=1000, learning_rate=0.01, log_file='results\\logs\\training_log.txt'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs, cot = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            log_entry = f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n'
            f.write(log_entry)

            if (epoch+1) % 100 == 0:
                print(log_entry.strip())
    
    return losses

if __name__ == "__main__":
    os.makedirs('results\\logs', exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()
    model = CoTModel()
    train_model(model, X_train, y_train)
