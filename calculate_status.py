import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

# --- 1. 基本設定 (請確保訓練路徑正確) ---
# ‼️ 請確保此路徑指向您存放訓練圖片的根目錄
TRAIN_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\train'
# TRAIN_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\test-renamed_images'

IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 8 # 可根據您的 CPU 核心數調整


# --- 2. 需要用到的 Dataset 類別 ---
class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []

        if not os.path.isdir(data_dir):
            print(f"錯誤：資料夾不存在於 '{data_dir}'")
            sys.exit(1)
            
        # 遍歷所有子資料夾，收集所有圖片的路徑
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 在這裡我們只需要圖片本身，所以回傳一個假標籤 0
        return image, 0


# --- 3. 計算 Mean 和 Std 的核心函式 ---
def calculate_mean_std(dataset, batch_size, num_workers):
    """
    計算資料集的像素平均值和標準差。
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    sum_channels = torch.zeros(3)
    sum_sq_channels = torch.zeros(3)
    pixel_count = 0

    print("開始計算資料集的 Mean 和 Std... 這可能會需要幾分鐘時間。")
    
    for images, _ in tqdm(loader, desc="Calculating Stats"):
        # images 的 shape: [batch, channels, height, width]
        b, c, h, w = images.shape
        num_pixels_in_batch = b * h * w
        
        sum_channels += images.sum(axis=[0, 2, 3])
        sum_sq_channels += (images ** 2).sum(axis=[0, 2, 3])
        pixel_count += num_pixels_in_batch

    mean = sum_channels / pixel_count
    std = torch.sqrt((sum_sq_channels / pixel_count) - mean ** 2)
    
    return mean, std

# --- 4. 主程式執行區塊 ---

if __name__ == "__main__":
    # 建立一個"乾淨"的 transform，只做必要的尺寸調整和格式轉換
    # ‼️ 這裡絕對不能有任何隨機擴充或 Normalize
    stats_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    # 建立 Dataset
    stats_dataset = ImageFolderDataset(TRAIN_PATH, transform=stats_transform)
    
    if len(stats_dataset) == 0:
        print(f"錯誤：在 '{TRAIN_PATH}' 中找不到任何圖片。請檢查路徑。")
    else:
        print(f"在訓練資料夾中找到 {len(stats_dataset)} 張圖片。")
        # 執行計算
        dataset_mean, dataset_std = calculate_mean_std(stats_dataset, BATCH_SIZE, NUM_WORKERS)
        
        # --- 顯示結果 ---
        print("\n" + "="*50)
        print("計算完成！")
        print(f"\n資料集 Mean (平均值):")
        print(dataset_mean)
        print(f"\n資料集 Std (標準差):")
        print(dataset_std)
        print("\n" + "="*50)
        print("\n請將以下程式碼複製到您的主要訓練腳本中：\n")
        
        # 格式化輸出，方便使用者複製
        mean_list = [round(val, 4) for val in dataset_mean.tolist()]
        std_list = [round(val, 4) for val in dataset_std.tolist()]
        
        print(f"dataset_mean = {mean_list}")
        print(f"dataset_std = {std_list}")
        print("\n# 然後在 transforms.Normalize 中使用這些變數：")
        print("# transforms.Normalize(mean=dataset_mean, std=dataset_std)")
        print("\n" + "="*50)