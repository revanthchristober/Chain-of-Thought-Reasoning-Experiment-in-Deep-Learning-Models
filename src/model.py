# src/model.py

import torch
import torch.nn as nn

class CoTModel(nn.Module):
    def __init__(self):
        super(CoTModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        output = self.output_layer(x)
        return output, self.generate_cot(x)

    def generate_cot(self, x):
        cot_steps = []
        for step in range(3):
            cot_steps.append(f"Step {step+1}: reasoning with input {x}")
        return cot_steps
