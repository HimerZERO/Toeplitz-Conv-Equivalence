import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ToeplitzLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.diagonals = nn.Parameter(torch.randn(in_features + out_features - 1))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        W = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(self.in_features):
                W[i, j] = self.diagonals[i + j]
        return x @ W.T

class ToeplitzNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = ToeplitzLinear(28*28, 512)
        self.fc2 = ToeplitzLinear(512, 128)
        self.fc3 = ToeplitzLinear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = ToeplitzNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
