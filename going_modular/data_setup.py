import os
import zipfile 
import requests
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


NUM_WORKERS = 2
data_path = Path('data')
image_path = data_path / "pizza_steak_sushi"

def get_data():
    if image_path.is_dir():
        print("Data already downloaded!")

    else:
         print(f"Did not find {image_path} directory, creating one...")
         image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)

    os.remove(data_path / "pizza_steak_sushi.zip")


def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int=NUM_WORKERS):
    """

    Returns a tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.

    """
    train_data = datasets.ImageFolder(root = train_dir, transform=transform)
    test_data = datasets.ImageFolder(root = test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names











