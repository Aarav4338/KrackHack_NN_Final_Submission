import torch
import os
import cv2
import glob
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./Offroad_Segmentation_Training_Dataset" # Point to your data
SAVE_DIR = "./Report_Visuals"
os.makedirs(SAVE_DIR, exist_ok=True)
FALCON_MAP = {0:0, 1:1, 3:2, 27:3, 39:4} # Map specific to your dataset

# --- LOAD MODEL ---
list_of_files = glob.glob('segformer_v2_epoch_*.pth')
latest_model = max(list_of_files, key=os.path.getctime)
model = smp.Unet(encoder_name="mit_b1", classes=5, activation=None).to(DEVICE)
model.load_state_dict(torch.load(latest_model))
model.eval()

# --- GET ONE IMAGE ---
# We grab a validation image to test
val_img_dir = os.path.join(DATA_DIR, "val", "color_images")
val_mask_dir = os.path.join(DATA_DIR, "val", "segmentation")
img_name = os.listdir(val_img_dir)[0] # Just take the first one

# Load Image & Mask
image = cv2.imread(os.path.join(val_img_dir, img_name))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(os.path.join(val_mask_dir, img_name), cv2.IMREAD_GRAYSCALE)

# Remap Mask
remapped_mask = np.zeros_like(mask)
for f_id, t_id in FALCON_MAP.items():
    remapped_mask[mask == f_id] = t_id

# Predict
input_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# --- CREATE NEON AUDIT ---
# Logic: If Pred == Truth, pixel is BLACK.
# If Pred != Truth, pixel is NEON RED.
h, w, _ = image.shape
audit_image = np.zeros((h, w, 3), dtype=np.uint8)

# Find errors
errors = (pred_mask != remapped_mask)

# Make errors RED (BGR format for OpenCV)
audit_image[errors] = [0, 0, 255] 

# Add the original image faintly so you can see context
overlay = cv2.addWeighted(image, 0.3, audit_image, 0.7, 0)

# Save
save_path = os.path.join(SAVE_DIR, "Figure_6_3_Neon_Audit.png")
cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"✅ Generated Figure 6.3 at: {save_path}")