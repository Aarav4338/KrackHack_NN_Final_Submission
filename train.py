import os
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = "./Offroad_Segmentation_Training_Dataset" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 
EPOCHS = 15             # Increased to 15 to let the Scheduler work its magic
LEARNING_RATE = 1e-4

# Falcon IDs -> Sequential Training IDs
FALCON_MAP = {0:0, 1:1, 3:2, 27:3, 39:4} 

# --- 2. DATA AUGMENTATION (Anti-Overfitting) ---
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    
    # Randomly removes small squares (forces context learning)
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
    
    # Geometric warping (prevents memorizing rock shapes)
    A.GridDistortion(p=0.3),
    
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 3. DATASET DEFINITION ---
class FalconDataset(Dataset):
    def __init__(self, root, transform=None):
        # Pointing to the specific subfolders you confirmed earlier
        self.img_dir = os.path.join(root, "train", "color_images")
        self.mask_dir = os.path.join(root, "train", "segmentation")
        
        print(f"--- Checking Paths ---")
        print(f"Looking for Images in: {os.path.abspath(self.img_dir)}")
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"CRITICAL: The folder '{self.img_dir}' was not found!")
            
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        # Store map for faster access
        self.map_dict = FALCON_MAP

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Load Image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Remap Falcon IDs (0, 1, 3, 27, 39) -> (0, 1, 2, 3, 4)
        remapped_mask = np.zeros_like(mask)
        for f_id, t_id in self.map_dict.items():
            remapped_mask[mask == f_id] = t_id

        if self.transform:
            augmented = self.transform(image=image, mask=remapped_mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.long()

# --- 4. INITIALIZE TRAINING ---
dataset = FalconDataset(DATA_DIR, transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# UPGRADE: Using mit_b1 (Slightly larger, better texture understanding)
print("🚀 Initializing SegFormer (mit_b1)...")
model = smp.Unet(
    encoder_name="mit_b1",      # <--- UPGRADE 1
    encoder_weights="imagenet", 
    classes=len(FALCON_MAP), 
    activation=None
).to(DEVICE)

# UPGRADE: Combo Loss (Dice for IoU + Focal for Hard Examples)
dice_loss = smp.losses.DiceLoss(mode='multiclass')
focal_loss = smp.losses.FocalLoss(mode='multiclass') # <--- UPGRADE 2

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# UPGRADE: Learning Rate Scheduler (Cosine Annealing)
# Slowly lowers LR to fine-tune the model perfectly at the end
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6) # <--- UPGRADE 3

# --- 5. TRAINING LOOP ---
print(f"Starting Training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate Combined Loss
        d_loss = dice_loss(outputs, masks)
        f_loss = focal_loss(outputs, masks)
        loss = d_loss + f_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    # Step the scheduler at the end of the epoch
    scheduler.step()
    
    # Save the model
    # We add _v2 so you don't overwrite your previous work
    save_name = f"segformer_v2_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), save_name)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(loader):.4f} | LR: {current_lr:.6f}")