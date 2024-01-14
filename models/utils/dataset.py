from PIL import Image
from torch.utils.data import Dataset
import functools
from models.mtcnn import MTCNN


class TripletFaceDataset(Dataset):
    """
        Custom PyTorch dataset class for loading triplets of images for face recognition tasks.

        Args:
            triplets_dataframe (pandas.DataFrame): Pandas dataframe containing the paths of triplets.
            transform (callable, optional): A function/transform to be applied to each image triplet.
                                           Default is None.

        Methods:
            __len__(): Returns the number of triplets in the dataset.
            get_image(image_path): Reads an image from the specified path and applies the specified
                                   transformation (if any).
            __getitem__(idx): Retrieves and returns the anchor, positive, and negative images for
                              the triplet at the specified index.

        Attributes:
            dataframe (pandas.DataFrame): The input dataframe containing triplet information.
            transform (callable): The transformation to be applied on image retrieval.

        Note:
            This dataset assumes that the input dataframe has columns named 'anchor', 'positive',
            and 'negative' containing file paths for the anchor, positive, and negative images
            respectively.
    """

    def __init__(self, triplets_dataframe, image_size, transform=None):
        self.dataframe = triplets_dataframe
        self.transform = transform
        self.mtcnn = MTCNN(image_size=image_size)

    def __len__(self):
        return self.dataframe.shape[0]

    def get_image(self, image_path):
        image = self.mtcnn(Image.open(image_path))
        if image is None:
            print('Image is none.')
        return self.transform(image)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        print(idx)
        anchor = self.get_image(data['anchor'])
        positive = self.get_image(data['positive'])
        negative = self.get_image(data['negative'])
        return anchor, positive, negative
