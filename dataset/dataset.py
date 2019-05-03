from torch.utils.data import Dataset
import pandas as pd 
import os 
import random
from PIL import Image
import torch

class ocrDataset(Dataset):
    def __init__(self, root, label, transform=None, target_transform=None):
        (self.path, self.label) = self.parse_path(root, label)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform 
        
    def parse_path(self, root, label):
        
        df = pd.read_json(os.path.join(root, label), typ='series')
        df = pd.DataFrame(df)
        df = df.reset_index()
        df.columns = ['index', 'label']
        return (df['index'].tolist(), df['label'].tolist())

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        try:
            filename = os.path.join(self.root, 'words')
            filename = os.path.join(filename, self.path[index])
            img = Image.open(filename).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.transform is not None:
            img = self.transform(img)
        label = self.label[index].encode()
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, label)

class alignCollate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        images, labels = zip(*batch)
        c = images[0].size(0)
        h = max([p.size(1) for p in images])
        w = max([p.size(2) for p in images])
        batch_images = torch.zeros(len(images), c, h, w).fill_(1)
        for i, image in enumerate(images):
            started_h = max(0, random.randint(0, h-image.size(1)))
            started_w = max(0, random.randint(0, w-image.size(2)))
            batch_images[i,:, started_h:started_h+image.size(1), started_w:started_w+image.size(2)] = image
        return batch_images, labels