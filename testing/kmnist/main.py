import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ToeplitzLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.diagonals = nn.Parameter(torch.empty(in_features + out_features - 1))
        self.in_features = in_features
        self.out_features = out_features
        idx_matrix = torch.LongTensor(
            [[i+j for j in range(in_features)] for i in range(out_features)]
        )
        self.reset_parameters()
        self.register_buffer("idx_matrix", idx_matrix)

    def reset_parameters(self):
        init.uniform_(self.diagonals, a=-1/10, b=1/10)


    def forward(self, x):
        W = self.diagonals[self.idx_matrix.to(x.device)]
        return x @ W.T

class ToeplitzNet(nn.Module):
    def __init__(self, Linear: type):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = Linear(28*28, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = Linear(256, 64)
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

def train(Linear: type, device, epochs = 10) -> tuple[list[float], list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToeplitzNet(Linear).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []


    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            running_loss += loss.item()


        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return train_losses, train_accuracies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses0, train_accuracies0 = train(ToeplitzLinear, device, epochs = 15)
train_losses1, train_accuracies1 = train(nn.Linear, device, epochs = 15)

plt.plot(train_losses0, label = "ToeplitzLinear")
plt.plot(train_losses1, label = "Linear")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(train_accuracies0, label = "ToeplitzLinear")
plt.plot(train_accuracies1, label = "Linear")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()