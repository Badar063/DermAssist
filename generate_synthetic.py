import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm

def generate_benign(size=(224,224)):
    img = np.random.randint(200, 255, size=(*size,3), dtype=np.uint8)  # skin base
    # Draw a symmetric oval
    center = (np.random.randint(70,154), np.random.randint(70,154))
    axes = (np.random.randint(30,50), np.random.randint(30,50))
    angle = np.random.randint(0,180)
    color = (np.random.randint(100,180), np.random.randint(60,120), np.random.randint(30,80))  # brownish
    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
    # Add slight texture
    noise = np.random.normal(0,5, img.shape).astype(np.uint8)
    img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)
    return img

def generate_melanoma(size=(224,224)):
    img = np.random.randint(200, 255, size=(*size,3), dtype=np.uint8)
    # Irregular shape: random polygon
    pts = []
    for _ in range(6,10):
        x = np.random.randint(50,174)
        y = np.random.randint(50,174)
        pts.append([x,y])
    pts = np.array(pts, np.int32).reshape((-1,1,2))
    color1 = (np.random.randint(0,50), np.random.randint(0,50), np.random.randint(0,50))  # dark
    color2 = (np.random.randint(100,200), np.random.randint(0,100), np.random.randint(0,100))  # red/blue
    cv2.fillPoly(img, [pts], color1)
    # Add a second color patch
    pts2 = pts + np.random.randint(-20,20, pts.shape)
    cv2.fillPoly(img, [pts2], color2)
    # Irregular border: add small protrusions
    return img

def generate_keratosis(size=(224,224)):
    img = np.random.randint(200, 255, size=(*size,3), dtype=np.uint8)
    # Warty surface: multiple small circles
    center = (np.random.randint(70,154), np.random.randint(70,154))
    for _ in range(30,50):
        x = center[0] + np.random.randint(-40,40)
        y = center[1] + np.random.randint(-40,40)
        r = np.random.randint(3,8)
        color = (np.random.randint(150,200),)*3  # grayish
        cv2.circle(img, (x,y), r, color, -1)
    return img

# Generate dataset
if __name__ == "__main__":
    os.makedirs("data/synthetic", exist_ok=True)
    metadata = []
    for cls, func in enumerate([generate_benign, generate_melanoma, generate_keratosis]):
        for i in tqdm(range(500), desc=f"Class {cls}"):
            img = func()
            fname = f"cls{cls}_{i:04d}.png"
            cv2.imwrite(f"data/synthetic/{fname}", img)
            metadata.append([fname, cls])
    pd.DataFrame(metadata, columns=["filename", "label"]).to_csv("data/metadata.csv", index=False)
