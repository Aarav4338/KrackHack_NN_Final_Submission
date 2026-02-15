import torch
import os
import glob
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
DATA_DIR = "./Offroad_Segmentation_Training_Dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
# Your Falcon IDs mapped to 0-4
FALCON_MAP = {0:0, 1:1, 3:2, 27:3, 39:4}
CLASS_NAMES = ["Background", "Ground", "Sky", "Rocks", "Bushes"]

# --- DATASET (Standard) ---
class FalconDataset(Dataset):
    def __init__(self, root, mode="val"):
        # Try to find validation folder first
        self.img_dir = os.path.join(root, mode, "color_images")
        self.mask_dir = os.path.join(root, mode, "segmentation")
        
        # Fallback to train if val doesn't exist (for testing purposes)
        if not os.path.exists(self.img_dir):
            print(f"⚠️  '{mode}' folder not found. Testing on TRAIN data instead.")
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
            
        image = cv2.resize(image, (512, 512))
        remapped_mask = cv2.resize(remapped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        image = np.transpose(image, (2, 0, 1)).astype('float32') / 255.0
        return torch.tensor(image), torch.tensor(remapped_mask).long()

# --- METRIC CALCULATOR ---
def evaluate():
    # 1. Find the latest V2 Model
    list_of_files = glob.glob('segformer_v2_epoch_*.pth')
    if not list_of_files:
        print("❌ No 'segformer_v2' models found! Did you run train_v2.py?")
        return
    
    latest_model = max(list_of_files, key=os.path.getctime)
    print(f"🏆 Evaluating Model: {latest_model}")

    # 2. Load Model (mit_b1)
    model = smp.Unet(encoder_name="mit_b1", classes=5, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(latest_model))
    model.eval()

    # 3. Setup Data
    dataset = FalconDataset(DATA_DIR, mode="val")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Metrics Storage
    # We use a global confusion matrix to calculate everything
    num_classes = 5
    total_conf_matrix = np.zeros((num_classes, num_classes))

    print("🚀 Calculating Accuracy & IoU...")
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # Convert logits to class IDs
            
            # Flatten to 1D arrays for confusion matrix
            preds_flat = preds.cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            # Update Confusion Matrix
            total_conf_matrix += confusion_matrix(masks_flat, preds_flat, labels=range(num_classes))

    # --- 5. COMPUTE FINAL STATS ---
    print("\n" + "="*40)
    print("      🔍  FINAL ACCURACY REPORT")
    print("="*40)
    
    # Per-Class IoU Calculation
    # IoU = TP / (TP + FP + FN)
    iou_scores = []
    
    print(f"{'CLASS':<15} | {'IoU':<10} | {'ACCURACY':<10}")
    print("-" * 40)
    
    for i in range(num_classes):
        tp = total_conf_matrix[i, i]
        fp = total_conf_matrix[:, i].sum() - tp
        fn = total_conf_matrix[i, :].sum() - tp
        
        iou = tp / (tp + fp + fn + 1e-10) # Avoid div by zero
        acc = tp / (tp + fn + 1e-10)      # Pixel accuracy for this class
        
        iou_scores.append(iou)
        print(f"{CLASS_NAMES[i]:<15} | {iou:.2%}     | {acc:.2%}")

    mIoU = sum(iou_scores) / num_classes
    pixel_acc = np.diag(total_conf_matrix).sum() / total_conf_matrix.sum()

    print("-" * 40)
    print(f"✅ Global Pixel Accuracy: {pixel_acc:.2%}")
    print(f"🏆 Mean IoU (mIoU):      {mIoU:.2%}")
    print("="*40)

if __name__ == "__main__":
    evaluate()