# calculate_stats.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import config
from data_loader import SimpsonsDataset

def calculate_and_save_stats():
    """Calculates mean and std of the training dataset and saves them to a file."""
    print("Calculating dataset mean and std...")
    
    stats_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    dataset = SimpsonsDataset(config.TRAIN_PATH, transform=stats_transform)
    if not dataset:
        print(f"Error: No images found in '{config.TRAIN_PATH}'. Please check the path.")
        return None, None

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)
    
    sum_channels, sum_sq_channels, pixel_count = torch.zeros(3), torch.zeros(3), 0
    
    for images, _ in tqdm(loader, desc="Calculating Stats"):
        b, c, h, w = images.shape
        num_pixels_in_batch = b * h * w
        sum_channels += images.sum(axis=[0, 2, 3])
        sum_sq_channels += (images ** 2).sum(axis=[0, 2, 3])
        pixel_count += num_pixels_in_batch
        
    mean = sum_channels / pixel_count
    std = torch.sqrt((sum_sq_channels / pixel_count) - mean ** 2)
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    # Write stats to file
    stats = {'mean': mean_list, 'std': std_list}
    
    with open(config.STATS_PATH, 'w') as f:
        json.dump(stats, f)
        
    print(f"Stats saved to {config.STATS_PATH}")
    print(f"Dataset Mean: {mean_list}")
    print(f"Dataset Std: {std_list}")
    
    return mean_list, std_list

if __name__ == "__main__":
    calculate_and_save_stats()