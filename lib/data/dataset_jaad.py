import numpy as np
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

class JAADDataset(Dataset):
    def __init__(self,data_path):
        dataset = read_pkl("data/jaad/jaad_database.pkl")
        self.bboxs = []
        self.labels = []
        for vid in dataset.keys():
            for ped in dataset[vid]['ped_annotations'].keys():
                for sample in dataset[vid]['ped_annotations'][ped]:
                    coords = []
                    for bbox in sample['bbox']:
                        coord = [bbox[0:2],bbox[2:4]]
                        coords.append(coord)
                    
                    label = sample['cross']
                    self.bboxs.append([np.asarray(coords).astype(np.float32)])
                    self.labels.append(label)

        self.bboxs = np.array(self.bboxs)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.bboxs)
    
    def __getitem__(self, idx):
        motion, label = self.bboxs[idx], self.labels[idx]
        return motion, label