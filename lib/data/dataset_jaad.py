import numpy as np
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

class JAADDataset(Dataset):
    def __init__(self,data_path):
        dataset = read_pkl(data_path)
        self.motions = []
        self.labels = []
        for vid in dataset.keys():
            for ped in dataset[vid]['ped_annotations'].keys():
                bbox = dataset[vid]['ped_annotations'][ped]['bbox']
                occ = dataset[vid]['ped_annotations'][ped]['occlusion']
                motion = np.concatenate((bbox,occ),axis=-1)
                self.motions.append(motion)
                self.labels.append(dataset[vid]['ped_annotations'][ped]['behavior']['cross'])
        self.motions = np.array(self.motions)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        motion, label = self.motions[idx], self.labels[idx]
        return motion, label