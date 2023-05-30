import numpy as np
from torch.utils.data import Dataset
from lib.utils.tools import read_pkl

def make_cam(x, img_shape):
    '''
        Normalize coordinate between -1 and 1 and keeping the ration between the y and x axis
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape

    #get the confidence score
    x_conf = x[:,:,2]

    #normalize the coordinates on the greater axis to keep the ratio
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1

    #restore the confidence score
    x_cam[:,:,2] = x_conf

    return x_cam

class JAADDataset(Dataset):
    '''
    Dataset class for JAAD dataset With only the bboxs
    Input: data_path (path to the pkl file)
           is_train (True for training, False for testing)
    '''
    def __init__(self,data_path,is_train = True):

        #import the dataset
        dataset = read_pkl(data_path)

        #get the split depending on if we want train or testing
        split = dataset['split']['train_ID' if is_train else 'test_ID']
        annotations = dataset['annotations']
        self.bboxs = []
        self.labels = []

        for vid in split:
            h = annotations[vid]['height']
            w = annotations[vid]['width']
            for ped in annotations[vid]['ped_annotations'].keys():
                for sample in annotations[vid]['ped_annotations'][ped]:
                    coords = []

                    #get the bboxs in the format [x1,y1,x2,y2] and transform to format [[x1,y1],[x2,y2]]
                    for bbox in sample['bbox']:
                        coord = [bbox[0:2],bbox[2:4]]
                        coords.append(coord)
                    
                    #normalize the coordinates of the bboxs
                    coords = make_cam(np.asarray(coords),(h,w)) 
                    self.bboxs.append([coords.astype(np.float32)]) #here in a list of 1d to keep the M size for the network

                    label = sample['cross']
                    self.labels.append(label)

        #transform the list of bboxs and labels to numpy array for the dataloader
        self.bboxs = np.array(self.bboxs)
        self.labels = np.array(self.labels)

    def __len__(self):
        '''
        Return the number of samples in the dataset, needed for the dataloader
        '''
        return len(self.bboxs)
    
    def __getitem__(self, idx):
        '''
        Return the sample at the index idx, needed for the dataloader
        '''
        motion, label = self.bboxs[idx], self.labels[idx]
        return motion, label

class KPJAADDataset(Dataset):
    '''
    Dataset class for JAAD dataset with keypoints, confidence score, bboxs and occlusion
    Input: data_path (path to the pkl file)
           is_train (True for training, False for testing)
    '''
    def __init__(self,data_path,is_train = True):

        #import the dataset
        dataset = read_pkl(data_path)

        #get the split depending on if we want train or testing
        split = dataset['split']['train_ID' if is_train else 'test_ID']
        annotations = dataset['annotations']
        self.motions = []
        self.labels = []

        for vid in split:
            h = annotations[vid]['height']
            w = annotations[vid]['width']
            for ped in annotations[vid]['ped_annotations'].keys():
                for sample in annotations[vid]['ped_annotations'][ped]:

                    #get the bboxs in the format [x1,y1,x2,y2] and transform to format [[x1,y1,o1],[x2,y2,o2]] andding occlusion
                    bboxs=[]
                    for bb, occlu in zip(sample['bbox'], sample['occlusion']):
                        bbox = [np.append(bb[0:2],occlu),np.append(bb[2:4],occlu)]
                        bboxs.append(bbox)
                    bboxs = np.asarray(bboxs)

                    #concatenate the keypoints and the bboxs
                    coords = np.concatenate((sample['2dkp'],bboxs),axis=1)

                    #normalize the coordinates of the keypoints and bboxs
                    coords = make_cam(np.asarray(coords),(h,w))
                    self.motions.append([coords.astype(np.float32)]) #here in a list of 1d to keep the M size for the network
                    label = sample['cross']
                    self.labels.append(label)

        #transform the list of keypoints and labels to numpy array for the dataloader
        self.motions = np.array(self.motions)
        self.labels = np.array(self.labels)

    def __len__(self):
        '''
        Return the number of samples in the dataset, needed for the dataloader
        '''
        return len(self.bboxs)
        
    def __getitem__(self, idx):
        '''
        Return the sample at the index idx, needed for the dataloader
        '''
        motion, label = self.bboxs[idx], self.labels[idx]
        return motion, label