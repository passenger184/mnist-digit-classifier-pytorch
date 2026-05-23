import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ─────────────────────────────
# DEVICE
# ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────
# MODEL (SAME ARCHITECTURE)
# ─────────────────────────────
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

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
model = MNISTNet().to(DEVICE)
model.load_state_dict(torch.load("models/mnist_model.pth", map_location=DEVICE))
model.eval()

print("Model loaded successfully!")

# ─────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ─────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)

    return prediction.item()

# ─────────────────────────────
# TEST RUN
# ─────────────────────────────
if __name__ == "__main__":
    img_path = "test.png"  
    result = predict(img_path)
    print(f"Predicted Digit: {result}")
