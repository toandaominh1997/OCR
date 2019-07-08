from torch.utils.data import Dataset
import pandas as pd 
import os 
import random
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

class ocrDataset(Dataset):
    def __init__(self, args, root, label, train, transform=None, target_transform=None):
        (self.path, self.target) = self.read_data(args, root, label, train)
        self.root = root
        self.label = label
        self.transform = transform
        self.target_transform = target_transform
    def read_typefile(self, filename, typefile):
        list_path = list()
        list_target = list()

        if typefile == 'json':
            df = pd.read_json(filename, typ='series')
            df = pd.DataFrame(df)
            df = df.reset_index()
            df.columns = ['index', 'target']
            list_path = df['index'].tolist()
            list_target = df['target'].tolist()
        elif typefile == 'txt':
            lines = list(open(filename))
            for line in lines:
                path, target = line.split('|')[0], line.split('|')[1]
                target = target.replace('\n', '')
                list_path.append(path)
                list_target.append(target)
        return list_path, list_target           
    def read_data(self, args, root, label, train):
        labels = label.split('+')
        list_path = list()
        list_target = list()
        for lab in labels:
            typefile = lab.split('.')[-1]
            list_path, list_target = self.read_typefile(os.path.join(root, lab), typefile)
        return list_path, list_target
    def __len__(self):
        return len(self.path)
    def __getitem__(self, index):
        try:
            root = os.path.join(self.root, '/'.join(self.label.split('/')[:-1]))
            filename = os.path.join(root, self.path[index])
            image = Image.open(filename).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.transform is not None:
            image = self.transform(image)
        target = self.target[index].encode()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)

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
