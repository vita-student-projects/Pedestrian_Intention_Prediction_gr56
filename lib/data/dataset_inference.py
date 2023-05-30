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

class KPInfDataset(Dataset):
    def __init__(self,data_path):
        annotations = read_pkl(data_path)

        self.motions = []
        self.bb_ped = [] #list of bbox to keep the position of the ped in the image after the prediction 
        self.pedsNseq = [] #list of number of sequences per ped
        self.pedFrames = [] #list of list of frames per ped
        self.num_frames = annotations['total_frame_vid']
        h = annotations['height']
        w = annotations['width']

        for ped in annotations['ped_annotations']:
            self.pedsNseq.append(len(ped)) #get the number of sequences for this ped
            ped_frames = []
            for sample in ped:
                ped_frames.append(sample['frames']) #get the frames for this ped

                #get the bboxs in the format [x1,y1,x2,y2] and transform to format [[x1,y1,o1],[x2,y2,o2]] andding occlusion
                bboxs=[]
                bbox_ped = []
                for bb, occlu in zip(sample['bbox'], sample['occlusion']):
                    bbox = [np.append(bb[0:2],occlu),np.append(bb[2:4],occlu)] #add the occlusion to the bbox
                    bb_ped = [bb[0:2],bb[2:4]] #keep the bbox without the occlusion for the ped position
                    bboxs.append(bbox)
                    bbox_ped.append(bb_ped)
                bboxs = np.asarray(bboxs)
                self.bb_ped.append(bbox_ped)

                #concatenate the bboxs with the keypoints
                coords = np.concatenate((sample['2dkp'],bboxs),axis=1)

                #normalize the coordinates of the keypoints and bboxs
                coords = make_cam(np.asarray(coords),(h,w))
                self.motions.append([coords.astype(np.float32)]) #here in a list of 1d to keep the M size for the network    
            self.pedFrames.append(ped_frames)#add the list of list of frames for this ped
        self.motions = np.array(self.motions)

    def __len__(self):
        '''
        Return the number of samples in the dataset, needed for the dataloader
        '''
        return len(self.motions)
    
    def __getitem__(self, idx):
        '''
        Return the sample at the index idx, needed for the dataloader
        '''
        return self.motions[idx]

    def get_seqs(self):
        '''
        Return the number of sequences per ped
        '''
        return self.pedsNseq
    
    def get_frames(self):
        '''
        Return the total number of frame in the video and the list of list of frames per ped
        '''
        return self.num_frames, self.pedFrames
    
    def get_bboxs(self):
        '''
        Return the position of pedestrian grouped by ped then by seq, all flattened
        '''
        return self.bb_ped
