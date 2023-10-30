import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import requests
from pathlib import Path
import zipfile
import os



device = "mps"



# Download data
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

def download_data():
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
    
        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...") 
            zip_ref.extractall(image_path)


def walk_through_dirs(dir_path):
    """
    Walks through dir_path returning its contents.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def prepare_data():
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transform.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root = train_dir,
                                      transform = data_transform,
                                      target_transform =None)

    test_data = datasets.ImageFolder(root = test_dir,
                                     transform = data_transform)

    train_dataloader = datasets.DataLoader(dataset = train_data,
                                           batch_size = 1,
                                           num_workers = 2,
                                           shuffle = True)

    test_dataloader = datasets.DataLoader(dataset = test_data,
                                          batch_size = 1,
                                          num_workers = 2,
                                          shuffle  = False)




def __main__():
    download_data()
    walk_through_dirs(image_path)
    prepare_data()
    

if __name__ == "__main__":
    __main__()
















































