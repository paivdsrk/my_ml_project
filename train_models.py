
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim

def train_sklearn_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Scikit-learn Model Accuracy: {accuracy}")

def train_pytorch_model():
    X = np.random.rand(100, 4)
    y = (X.sum(axis=1) > 2).astype(int)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y).float().mean()
        print(f"PyTorch Model Accuracy: {accuracy}")

if __name__ == "__main__":
    train_sklearn_model()
    train_pytorch_model()
