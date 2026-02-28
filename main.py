from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
import base64
import cv2
from src.models.cnn import create_model
from src.models.explain import GradCAM, overlay_heatmap

app = FastAPI(title="SynthDerm API")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('resnet18', num_classes=3, pretrained=False)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device)
model.eval()

# Define target layer for Grad-CAM (last convolutional layer of resnet18)
target_layer = model.layer4[-1].conv2

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0).to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess_image(image)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, pred_class)

    # Convert heatmap to overlay
    original_np = np.array(image.resize((224,224)))
    overlayed = overlay_heatmap(original_np, heatmap)

    # Encode overlay to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "class": int(pred_class),
        "class_name": ["benign", "melanoma", "keratosis"][pred_class],
        "confidence": confidence,
        "heatmap_overlay": img_base64
    })

@app.get("/health")
def health():
    return {"status": "ok"}
