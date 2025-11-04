import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from model import MLP
from dataset import ImbalancedMoonsDataset


# Initialize dataset
dataset: ImbalancedMoonsDataset = ImbalancedMoonsDataset(n_samples=20000, noise=0.2, imbalance_ratio=0.01)

# Train-test split with stratification
indices: np.ndarray = np.arange(len(dataset))
y_labels: np.ndarray = dataset.y.numpy()
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y_labels, random_state=42)

# Create train and test loaders
train_dataset: Subset = Subset(dataset, train_idx)
test_dataset: Subset = Subset(dataset, test_idx)
train_loader: DataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader: DataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, optimizer, and loss function
model: MLP = MLP(input_size=2, hidden_size=64)
optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=0.001)
criterion: nn.BCELoss = nn.BCELoss()

# Training loop
losses: list[float] = []
num_epochs: int = 50

for epoch in range(num_epochs):
    model.train()
    total_loss: float = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs: torch.Tensor = model(batch_x)
        loss: torch.Tensor = criterion(outputs.squeeze(), batch_y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss: float = total_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Validation
model.eval()
all_preds: list[int] = []
all_labels: list[int] = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs: torch.Tensor = model(batch_x)
        preds: torch.Tensor = (outputs.squeeze() > 0.5).int()
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy().astype(int))

# Calculate metrics
accuracy: float = accuracy_score(all_labels, all_preds)
precision: float = precision_score(all_labels, all_preds)
recall: float = recall_score(all_labels, all_preds)
f1: float = f1_score(all_labels, all_preds)
cm: np.ndarray = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("Validation Metrics:")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Save the trained model
# torch.save(model.state_dict(), "model.pth")
# print("Model saved to model.pth")
