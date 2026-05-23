import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

# ─────────────────────────────────────
# DATA
# ─────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training batches:   {len(train_loader)}")
print(f"Test batches:       {len(test_loader)}")

# ─────────────────────────────────────
# MODEL
# ─────────────────────────────────────
class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)

model   = MNISTNet().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────
# TRAINING
# ─────────────────────────────────────
train_losses = []
test_losses  = []
test_accs    = []

for epoch in range(EPOCHS):

    # ── TRAIN ──
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        preds = model(images)           # forward pass
        loss  = loss_fn(preds, labels)  # calculate loss

        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # calculate new gradients
        optimizer.step()       # update weights

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── EVALUATE ──
    model.eval()
    running_test_loss = 0
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(images)
            loss  = loss_fn(preds, labels)
            running_test_loss += loss.item()

            predicted = preds.argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    avg_test_loss = running_test_loss / len(test_loader)
    accuracy      = correct / total

    test_losses.append(avg_test_loss)
    test_accs.append(accuracy)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train: {avg_train_loss:.4f} | "
          f"Test: {avg_test_loss:.4f} | "
          f"Acc: {accuracy:.2%}")

# ─────────────────────────────────────
# RESULTS
# ─────────────────────────────────────
print(f"\nFinal Test Accuracy: {test_accs[-1]:.2%}")


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/mnist_model.pth")
print("Model saved to mnist_model.pth")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train")
ax1.plot(test_losses,  label="Test")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(test_accs)
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
