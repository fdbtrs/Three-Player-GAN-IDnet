import numbers
import os
import queue as Queue
import threading
from pathlib import Path
import logging

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

from utils.rand_augment import RandAugment

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
    



class AugmentationFaceDatasetFolder(Dataset):
    
    def __init__(self, root_dir, local_rank, sample_file):
        super(AugmentationFaceDatasetFolder, self).__init__()

        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            RandAugment(num_ops=4, magnitude=16),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ])

        self.root_dir = root_dir
        self.local_rank = local_rank
        
        self._image_names, self._image_labels = self.load_available_samples(
                                                    root_dir,
                                                    sample_file
                                                )
        self.num_imgs = len(self._image_names)
        
        
    def load_available_samples(self, root_dir, sample_file):
        
        image_names     = []
        image_labels    = []
        
        logging.info(f"Dataset root: {root_dir}")
        logging.info(f"Sample file: {sample_file}")
        logging.info(f"Amount folders in root dir: {len([f for f in Path(root_dir).iterdir() if f.is_dir()])}")
        
        with open(f"{root_dir}/{sample_file}", "r") as f:
            for line in f:
                if len(line) < 3:
                    continue
                
                path, label = line.strip().split("\t")
                
                p = Path(f"{root_dir}/{path}").resolve()
                
                if not p.exists():
                    logging.warn(f"Image {p} does not exist in dataset!")
                
                image_names.append(p)
                image_labels.append(int(label))
                
        return image_names, image_labels
        
    
    def __getitem__(self, index):
        
        img_name    = self._image_names[index]
        img_label   = self._image_labels[index]
        
        img_label = torch.tensor(img_label, dtype=torch.long)
                
        img = cv2.imread(str(img_name))
        
        if img is None:
            logging.error(f"Image {img_name} could not be loaded!")
            raise ValueError(f"Image {img_name} could not be found.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, img_label


    def __len__(self):
        return self.num_imgs
    
    
class FaceDatasetFolder(Dataset):
    
    def __init__(self, root_dir, local_rank, sample_file):
        super(FaceDatasetFolder, self).__init__()

        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ])

        self.root_dir = root_dir
        self.local_rank = local_rank
        
        self._image_names, self._image_labels = self.load_available_samples(
                                                    root_dir,
                                                    sample_file
                                                )
        self.num_imgs = len(self._image_names)
        
        
    def load_available_samples(self, root_dir, sample_file):
        
        image_names     = []
        image_labels    = []
        
        logging.info(f"Dataset root: {root_dir}")
        logging.info(f"Sample file: {sample_file}")
        logging.info(f"Amount folders in root dir: {len([f for f in Path(root_dir).iterdir() if f.is_dir()])}")
        
        with open(f"{root_dir}/{sample_file}", "r") as f:
            for line in f:
                if len(line) < 3:
                    continue
                
                path, label = line.strip().split("\t")
                
                p = Path(f"{root_dir}/{path}").resolve()
                
                if not p.exists():
                    logging.warn(f"Image {p} does not exist in dataset!")
                
                image_names.append(p)
                image_labels.append(int(label))
                
        return image_names, image_labels
        
    
    def __getitem__(self, index):
        
        img_name    = self._image_names[index]
        img_label   = self._image_labels[index]
        
        img_label = torch.tensor(img_label, dtype=torch.long)
                
        img = cv2.imread(str(img_name))
        
        if img is None:
            logging.error(f"Image {img_name} could not be loaded!")
            raise ValueError(f"Image {img_name} could not be found.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, img_label


    def __len__(self):
        return self.num_imgs