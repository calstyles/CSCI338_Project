import os
#import pandas as pd
import torch
#import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io

class LinesPiesBars(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        #could figure out a way to not hardcode this, but this is sufficient for the moment
        return 11965

    def __getitem__(self, index):
        if (index < 6970):
            img_path = os.path.join(self.root_dir, f"GraphBar_{str(index)}.png")
            y_label = torch.tensor(0)
        elif (index < 10042):
            img_path = os.path.join(self.root_dir, f"GraphLine_{str(index)}.png")
            y_label = torch.tensor(1)
        elif (index < 11965):
            img_path = os.path.join(self.root_dir, f"GraphPie_{str(index)}.png")
            y_label = torch.tensor(2)
        #if someone has another way to "read" an image, we don't have to use the scikit-image package
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)

        return (image, y_label)
