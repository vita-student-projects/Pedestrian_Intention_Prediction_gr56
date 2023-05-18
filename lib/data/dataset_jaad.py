import numpy as np
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

class JAADDataset(Dataset):
    def __init__(self,data_path,is_train = True):
        dataset = read_pkl(data_path)
        self.bboxs = []
        self.labels = []
        i = 0
        for vid in dataset.keys():
            i+=1
            if is_train and (i > 253):
                break
            elif not is_train and (i <= 253):
                continue
            h = dataset[vid]['height']
            w = dataset[vid]['width']
            for ped in dataset[vid]['ped_annotations'].keys():
                for sample in dataset[vid]['ped_annotations'][ped]:
                    coords = []
                    for bbox in sample['bbox']:
                        coord = [bbox[0:2],bbox[2:4]]
                        coords.append(coord)
                    
                    coords = make_cam(np.asarray(coords),(h,w))
                    self.bboxs.append([coords.astype(np.float32)]) #here in a list of 1d to keep the M size for the network
                    label = sample['cross']
                    self.labels.append(label)

        self.bboxs = np.array(self.bboxs)
        #self.bboxs = self.bboxs[0:2] #test overfitting
        self.labels = np.array(self.labels)
        #self.labels = self.labels[0:2] #test overfitting

    def __len__(self):
        return len(self.bboxs)
    
    def __getitem__(self, idx):
        motion, label = self.bboxs[idx], self.labels[idx]
        return motion, label
    
def make_cam(x, img_shape):
    '''
        Normalize coordinate between -1 and 1 and keeping the ration between the y and x axis
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam