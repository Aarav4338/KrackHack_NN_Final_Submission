import torch
import torch.nn as nn
import os
import glob
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./Offroad_Segmentation_Training_Dataset"
VAL_SUBFOLDER = "val" # Ensure this folder exists inside DATA_DIR!
BATCH_SIZE = 4
FALCON_MAP = {0:0, 1:1, 3:2, 27:3, 39:4}

# --- DATASET (Same as Train) ---
class FalconDataset(Dataset):
    def __init__(self, root, mode="val"):
        self.img_dir = os.path.join(root, mode, "color_images") # Check folder names!
        self.mask_dir = os.path.join(root, mode, "segmentation")
        if not os.path.exists(self.img_dir):
             # Fallback if user doesn't have 'val', use 'train' for demo
             print(f"⚠️ Warning: '{mode}' folder not found. Using 'train' as validation proxy.")
             self.img_dir = os.path.join(root, "train", "color_images")
             self.mask_dir = os.path.join(root, "train", "segmentation")
        self.images = os.listdir(self.img_dir)
        self.map_dict = FALCON_MAP

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        remapped_mask = np.zeros_like(mask)
        for f_id, t_id in self.map_dict.items():
            remapped_mask[mask == f_id] = t_id
            
        # Resize to 512x512
        image = cv2.resize(image, (512, 512))
        remapped_mask = cv2.resize(remapped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Norm & Tensor
        image = np.transpose(image, (2, 0, 1)).astype('float32') / 255.0
        return torch.tensor(image), torch.tensor(remapped_mask).long()

# --- METRIC CALCULATOR ---
def calculate_metrics():
    # 1. Setup Data
    val_dataset = FalconDataset(DATA_DIR, mode="val") # Tries to find 'val' folder
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Define Losses
    dice_fn = smp.losses.DiceLoss(mode='multiclass')
    # Weighted Cross Entropy (Penalize Sky/Background less, Obstacles more)
    # Weights: [Back, Ground, Sky, Rocks, Bushes]
    class_weights = torch.tensor([0.5, 1.0, 0.5, 2.0, 2.0]).to(DEVICE)
    ce_fn = nn.CrossEntropyLoss(weight=class_weights)

    val_losses_dice = []
    val_losses_ce = []
    epochs = []

    # 3. Loop through saved checkpoints
    checkpoints = sorted(glob.glob("segformer_v2_epoch_*.pth"), key=os.path.getctime)
    print(f"🔍 Found {len(checkpoints)} checkpoints. Analyzing history...")

    for i, ckpt in enumerate(checkpoints):
        print(f"   Processing Epoch {i+1}...")
        model = smp.Unet(encoder_name="mit_b1", classes=5, activation=None).to(DEVICE)
        model.load_state_dict(torch.load(ckpt))
        model.eval()

        epoch_dice = 0
        epoch_ce = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                
                epoch_dice += dice_fn(outputs, masks).item()
                epoch_ce += ce_fn(outputs, masks).item()
        
        avg_dice = epoch_dice / len(val_loader)
        avg_ce = epoch_ce / len(val_loader)
        
        val_losses_dice.append(avg_dice)
        val_losses_ce.append(avg_ce)
        epochs.append(i+1)

    # 4. PLOTTING
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, val_losses_dice, label='Validation Dice Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses_ce, label='Val Weighted Entropy Loss', marker='s', color='red')
    plt.title('Validation Loss Breakdown per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss_Analysis_Curve.png')
    print("✅ Analysis Complete! Check 'Loss_Analysis_Curve.png'")

if __name__ == "__main__":
    calculate_metrics()