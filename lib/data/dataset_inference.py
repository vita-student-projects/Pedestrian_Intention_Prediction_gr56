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
    x_conf = x[:,:,2]
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    x_cam[:,:,2] = x_conf
    return x_cam

class KPInfDataset(Dataset):
    def __init__(self,data_path):
        annotations = read_pkl(data_path)
        self.motions = []
        self.bb_ped = []
        self.pedsNseq = []
        self.pedFrames = []
        self.num_frames = annotations['total_frame_vid']
        h = annotations['height']
        w = annotations['width']
        for ped in annotations['ped_annotations']:
            self.pedsNseq.append(len(ped))
            ped_frames = []
            for sample in ped:
                ped_frames.append(sample['frames'])
                bboxs=[]
                bbox_ped = []
                for bb, occlu in zip(sample['bbox'], sample['occlusion']):
                    bbox = [np.append(bb[0:2],occlu),np.append(bb[2:4],occlu)]
                    bb_ped = [bb[0:2],bb[2:4]]
                    bboxs.append(bbox)
                    bbox_ped.append(bb_ped)
                bboxs = np.asarray(bboxs)
                coords = np.concatenate((sample['2dkp'],bboxs),axis=1)
                coords = make_cam(np.asarray(coords),(h,w))
                self.motions.append([coords[:30].astype(np.float32)]) #here in a list of 1d to keep the M size for the network
                self.bb_ped.append(bbox_ped)
                
            self.pedFrames.append(ped_frames)
        self.motions = np.array(self.motions)
        print(self.motions.shape)

    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        return self.motions[idx]

    def get_seqs(self):
        return self.pedsNseq
    
    def get_frames(self):
        return self.num_frames, self.pedFrames
    
    def get_bboxs(self):
        return self.bb_ped
