import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

class FamilyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=to_pil_image, target_transform=None):
        """
        :param annotations_file: The csv file with filenames and descriptions
        :param img_dir: The directory containing the image
        :param height: The image height in pixels
        :param width: The image width in pixels
        """
        self.data = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform=transform
        self.target_transform=target_transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = read_image(img_path)
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data.index)
