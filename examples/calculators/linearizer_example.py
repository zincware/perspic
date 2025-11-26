import torch
import torch.nn as nn
import torch.nn.functional as F

from perspic.calculator.linearizer import Linearizer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleModel()
criterion = nn.CrossEntropyLoss()

x = torch.randn(4, 10)
y = torch.randint(0, 5, (4,))

eta_array = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

linearizer = Linearizer(eta_array=eta_array)

results = linearizer.probe_train_step(
    model=model,
    criterion=criterion,
    x=x,
    y=y,
)
print(results)
