from torchvision import io
from torch.utils.data import Dataset
import functools


class TripletFaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    @functools.lru_cache(maxsize=128)
    def get_image(self, image_path):
        image = io.read_image(image_path)
        return self.transform(image)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        anchor = self.get_image(data['anchor'])
        positive = self.get_image(data['positive'])
        negative = self.get_image(data['negative'])
        return anchor, positive, negative
