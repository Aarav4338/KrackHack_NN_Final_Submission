import torch
import os
import cv2
import glob
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "./Offroad_Segmentation_testImages/Color_Images"
OUTPUT_DIR = "./Audit_Failures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find latest v2 model
list_of_files = glob.glob('segformer_v2_epoch_*.pth')
latest_model = max(list_of_files, key=os.path.getctime)
print(f"🕵️‍♀️ Auditing failures using: {latest_model}")

model = smp.Unet(encoder_name="mit_b1", classes=5, activation=None).to(DEVICE)
model.load_state_dict(torch.load(latest_model))
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

results = []

print(f"🚀 Scanning dataset for confusion (Entropy Calculation)...")
files = os.listdir(TEST_DIR)

for img_name in tqdm(files):
    img_path = os.path.join(TEST_DIR, img_name)
    original_img = cv2.imread(img_path)
    if original_img is None: continue

    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # --- CALCULATE ENTROPY (Confusion Metric) ---
        # Formula: -sum(p * log(p))
        # High Entropy = Model doesn't know what it's looking at
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        mean_entropy = entropy.mean().item()
        
        results.append((img_name, mean_entropy, original_img, entropy.squeeze().cpu().numpy()))

# Sort by HIGHEST confusion first
results.sort(key=lambda x: x[1], reverse=True)

print(f"🔥 Saving Top 20 Most Confusing Images...")

# Define Heatmap Colors (Blue=Sure, Red=Confused)
for i in range(20):
    img_name, score, original, entropy_map = results[i]
    
    # Normalize entropy for visualization (0-255)
    entropy_vis = (entropy_map / np.log(5)) * 255 # log(5) is max entropy for 5 classes
    entropy_vis = entropy_vis.astype(np.uint8)
    heatmap = cv2.applyColorMap(entropy_vis, cv2.COLORMAP_JET)
    
    # Overlay
    original_resized = cv2.resize(original, (512, 512))
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
    
    # Stack: [Original | Confusion Heatmap]
    final = np.hstack([original_resized, overlay])
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"CONFUSED_{i+1}_{img_name}"), final)

print(f"✅ Audit Complete. Check '{OUTPUT_DIR}' for the hardest images.")