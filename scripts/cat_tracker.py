import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image
import os

# ==== CONFIG ============
IMAGE_SIZE = 224
MODEL_PATH = "models/cat_activity_model.pth"
DATA_DIR = "data"  # For class names from ImageFolder
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruct label names
class_names = sorted(os.listdir(DATA_DIR))

# Define transforms (same as training!)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Open webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL and preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()
        label = class_names[pred]

    # Show prediction
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Cat Activity Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
