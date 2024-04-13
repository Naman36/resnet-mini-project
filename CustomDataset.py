import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_lists, transform=None):
        """
        data_lists: List of lists, where each sublist is 3072 elements representing an image.
        transform: PyTorch transforms to apply to the data.
        """
        self.data_lists = data_lists
        self.transform = transform

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, idx):
        # Convert list to tensor and reshape to 3x32x32
        image = torch.FloatTensor(self.data_lists[idx]).view(3, 32, 32) / 255.0
        if self.transform:
            image = self.transform(image)
        return image
