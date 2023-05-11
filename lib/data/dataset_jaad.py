import numpy as np
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

class JAADDataset(Dataset):
    def __init__(self,data_path):
        dataset = read_pkl(data_path)
        self.bboxs = []
        self.labels = []
        for vid in dataset.keys():
            for ped in dataset[vid]['ped_annotations'].keys():
                for sample in dataset[vid]['ped_annotations'][ped]:
                    bbox = sample['bbox']
                    label = sample['cross']

                    if bbox.shape != (60,4):##temporarily
                        continue
                    if type(label) != np.int64:#temporarily
                        continue
                    
                    self.bboxs.append(bbox.astype(np.float32))
                    self.labels.append(label)

        self.bboxs = np.array(self.bboxs)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.bboxs)
    
    def __getitem__(self, idx):
        motion, label = self.bboxs[idx], self.labels[idx]
        return motion, label