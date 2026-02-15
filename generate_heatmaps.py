import torch
import os
import cv2
import glob
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "./Offroad_Segmentation_testImages/Color_Images"
OUTPUT_DIR = "./Heatmap_Audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find latest model (v2)
list_of_files = glob.glob('segformer_v2_epoch_*.pth')
latest_model = max(list_of_files, key=os.path.getctime)
print(f"🔥 generating Heatmaps using: {latest_model}")

# Load Model (Ensure mit_b1 matches your v2 training)
model = smp.Unet(encoder_name="mit_b1", classes=5, activation=None).to(DEVICE)
model.load_state_dict(torch.load(latest_model))
model.eval()

# Colors for Class Overlay
# 0:Back(Black), 1:Ground(Green), 2:Sky(Blue), 3:Rocks(Yellow), 4:Bushes(Red)
COLORS = np.array([
    [0, 0, 0],       # Background
    [0, 255, 0],     # Ground
    [139, 0, 0],     # Sky
    [255, 255, 0],   # Rocks
    [0, 0, 255]      # Bushes
], dtype=np.uint8)

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print(f"🚀 Processing images...")

# Process first 20 images for the report
images = os.listdir(TEST_DIR)[:20]

for img_name in images:
    img_path = os.path.join(TEST_DIR, img_name)
    original_img = cv2.imread(img_path)
    if original_img is None: continue

    # 1. PREDICT
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_tensor)
        
        # A. Hard Prediction (Class ID)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        
        # B. Soft Prediction (Rock Probability - Class 3)
        # We extract the probability map purely for "Rocks"
        rock_prob = probs[0, 3, :, :].cpu().numpy()

    # --- VISUALIZATION 1: CLASS OVERLAY ---
    color_mask = COLORS[mask]
    original_resized = cv2.resize(original_img, (512, 512))
    
    # Blend: 60% Image + 40% Mask
    overlay = cv2.addWeighted(original_resized, 0.6, color_mask, 0.4, 0)

    # --- VISUALIZATION 2: ROCK HEATMAP ---
    # Convert probability (0-1) to Heatmap Color (Blue->Red)
    heatmap = cv2.applyColorMap(np.uint8(255 * rock_prob), cv2.COLORMAP_JET)
    
    # Overlay Heatmap on original
    heatmap_overlay = cv2.addWeighted(original_resized, 0.5, heatmap, 0.5, 0)

    # --- COMBINE & SAVE ---
    # Layout: [ Original | Segmentation | Rock Heatmap ]
    combined = np.hstack([original_resized, overlay, heatmap_overlay])
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"HEATMAP_{img_name}"), combined)
    print(f"   Saved: HEATMAP_{img_name}")

print(f"✅ Heatmaps ready in '{OUTPUT_DIR}'")