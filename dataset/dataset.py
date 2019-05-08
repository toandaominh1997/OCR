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
    def read_data(self, args, root, label, train):
        typefile = label.split('.')[-1]
        list_path = list()
        list_target = list()
        if(typefile=='json'):
            df = pd.read_json(os.path.join(root, label), typ='series')
            df = pd.DataFrame(df)
            df = df.reset_index()
            df.columns = ['index', 'target']
            list_path = df['index'].tolist()
            list_target = df['target'].tolist()
            if(train==False):
                list_path_fake = list()
                list_target_fake = list()
                for path, target in zip(list_path, list_target):
                    check = False
                    for l in target:
                        if(l not in args.alphabet):
                            check=True
                    if(check==False):
                        list_path_fake.append(path)
                        list_target_fake.append(target)
                list_path=list_path_fake
                list_target=list_target_fake
        elif(typefile=='txt'):
            lines = list(open(os.path.join(root, label)))
            for line in lines:
                path, target = line.split('|')[0], line.split('|')[1]
                target = target.replace('\n', '')
                check=False
                if(train==False):
                    for l in target:
                        if(l not in args.alphabet):
                            check=True
                if(check==False):
                    list_path.append(path)
                    list_target.append(target)
        # if(train):
        #     list_path, _, list_target, _ = train_test_split(list_path, list_target, test_size=0.9998)
        return (list_path, list_target)
    def __len__(self):
        return len(self.path)
    def __getitem__(self, index):
        try:
            root = os.path.join(self.root, '/'.join(self.label.split('/')[:-1]))
            filename = os.path.join(root, self.path[index])
            img = Image.open(filename).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]
        if self.transform is not None:
            img = self.transform(img)
        target = self.target[index].encode()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

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