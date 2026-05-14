from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np



class Mnist(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.len = len(self.label)
        self.transforms = transforms.Compose([transforms.ToTensor() , transforms.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img = self.dataset[item]
        img_id = self.label[item]

        img = np.transpose(img,(1,2,0))
        img = self.transforms(img)

        return img, img_id


