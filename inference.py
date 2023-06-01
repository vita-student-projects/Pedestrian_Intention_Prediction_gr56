import pickle5 as pickle
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join, abspath, exists, dirname, isfile
import cv2
import PIL
import json
import openpifpaf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.dataset_inference import KPInfDataset
from lib.model.model_action import ActionNet




def parse_args():
    '''
    Function used to parse the launch arguments
    Input : None
    Output : opts (launch arguments)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=dirname(abspath('__file__')), help="Path to the folder of the repository")
    parser.add_argument('--filename', type=str, help="Name of the file used for interference (without extension)")
    parser.add_argument('--config', type=str, default='configs/inference.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    return opts

def infer_model(opts,args):
    '''
    Function used to perform inference on the JAAD dataset
    Input : args (config arguments)
    Output : None
    '''
    print('INFO: Inference mode')
    print('INFO: Loading model')
    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()

    #move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 2,
          'pin_memory': True,
          'persistent_workers': True,
          'drop_last': False
    }

    #load the dataset on which we perform the inference and the model
    jaad_ts = KPInfDataset(data_path=join('datagen/data/' + opts.filename + "_PIP.pkl"))
    test_loader = DataLoader(jaad_ts, **testloader_params)
    model.load_state_dict(torch.load(args.chk_path, map_location=lambda storage, loc: storage)['model'], strict=True)

    print('INFO: Inference on {} sequences'.format(len(test_loader)))
    
    model.eval() #put the model in eval mode
    
    preds = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader)):

            #move the batch to the GPU if available
            if torch.cuda.is_available():
                batch = batch.cuda()

            #perform the inference
            output = model(batch)
            prob = F.softmax(output, dim=1)
            p_max, pred = torch.max(prob,dim=1) #get confidence score and the prediction

            #append the prediction and the confidence score to the list
            preds.append([pred,p_max])
    
    print('INFO: Inference done')
    
    #put the predictions back to the right sequence of the right pedestrian, same for the bboxs
    prediction = []
    bb_order = []
    bboxs = jaad_ts.get_bboxs()
    seqs = jaad_ts.get_seqs()
    for i in range(len(seqs)):
        seq = seqs[i]
        prediction.append(preds[:seq])
        preds = preds[seq:]
        bb_order.append(bboxs[:seq])
        bboxs = bboxs[seq:]
        
    #get the frames for each pedestrian and the total number of frames
    num_frames, pedFrames = jaad_ts.get_frames()
    
    #create a dictionary with the predictions in the right format
    output = []
    for i in range(num_frames):
        frame_dict = {'frame' : i+1, 'predictions' : []}
        ped_pred = {}
        ped_bbox = {}
        for j in range(len(pedFrames)):
            for k in range(len(pedFrames[j])):
                if i in pedFrames[j][k]:
                    ped_pred[j] = [tensor.item() for tensor in prediction[j][k]] #get the prediction and the confidence score for the ped in the frame
                    ped_bbox[j] = bb_order[j][k][pedFrames[j][k].index(i)] #get the bbox for the ped in the frame for visiualization

                #if the ped is not in the frame, add None to the prediction and the bbox
                elif j not in ped_pred.keys():
                    ped_pred[j] = [None,None] 
                    ped_bbox[j] = None

        #add the predictions and the bboxs to the frame dictionary
        for ped in ped_pred.keys():
            frame_dict['predictions'].append({'pred' : ped_pred[ped][0],'confidence': ped_pred[ped][1], 'bbox' : ped_bbox[ped]})
        output.append(frame_dict)

    inference = {'project' : 'Pedestrian Intention Prediction',
                 'output' : output}

    #save the prediction in a json file
    with open(join(opts.data_path, "datagen/infer_DB/infer_pred/"+ opts.filename + ".json"), 'w') as fp:
        json.dump(inference, fp)
    
    print('INFO: Prediction saved in {}'.format(join('datagen/infer_DB/infer_pred/'+ opts.filename + ".json")))

class Inference(object):
    def __init__(self, data_path = ''):
        
        self._fps = 30
        self.nbr_frame_seq = 30
        self._t_pred = 1.0

        # Paths
        self._data_path = data_path if data_path else dirname(abspath('__file__'))
        assert exists(self._data_path), \
            'Data path does not exist: {}'.format(self._data_path)
        self._infer_folder_path = join(self._data_path, 'datagen/infer_DB')
        self._infer_clips_path = join(self._infer_folder_path, 'infer_clips')
        self._infer_out_path = join(self._infer_folder_path, 'infer_out')
        self._infer_pkl_path = join(self._data_path, 'datagen/data')
        self._infer_pred_path = join(self._infer_folder_path, 'infer_pred')
        
        

    def _get_2dkp_vid(self, vid, processor):
        """
        Extracts the 2d keypoints of each pedestrian in the whole 
        video with OpenPifPaf
        :param vid: The video id
        :param processor: The processor used to get the 2d keypoints
        :return: an array of list of 2d keypoints for a sequence
        """

        f = join(self._infer_clips_path, vid + '.mp4')
        vidcap = cv2.VideoCapture(f)
        success, image = vidcap.read()

        kp_OPP = []
        pil_imgs = []
        
        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_imgs.append(PIL.Image.fromarray(image))
            success, image = vidcap.read()

        data = openpifpaf.datasets.PilImageList(pil_imgs)

        batch_size = 5
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True)
        
        print('Creating 2D keypoints')
        for images_batch, _, _ in tqdm(loader):
            images_batch = images_batch.cuda()
            fields_batch = processor.fields(images_batch)

            for i in range(len(fields_batch)):
              predictions = processor.annotations(fields_batch[i])
              
              if not predictions:
                  kp_OPP.append([np.zeros((17, 3))])
              else:
                  tmp_2dkp = []
                  for pred in predictions:
                      tmp_2dkp.append(pred.data)
                  kp_OPP.append(tmp_2dkp)

        return np.array(kp_OPP, dtype=object)
    

    
    def generate_data_OPP(self, filename):
        """
        Extracts an interference Dataset based
        on the input video

        Dictionary structure:
        'vid_id'(str):      str
        'num_frames':       int
        'width':            int
        'height':           int
        'seqs':             list (dict)
            'seq_id':           int
            'frames':           list (dict)
                'frame_id':         int
                'peds':             list (dict)
                    'ped_id':           int
                    'occlusion':        int
                    'bbox':             [x1 (float), y1 (float), x2 (float), y2 (float)]
                    '2dkp':             array(array)
        """

        clip_file_path = join(self._infer_clips_path, filename + ".mp4")

        if not isfile(clip_file_path):
            print("No pickle file found at " + self._infer_pkl_path)
            return

        forecast_step = int(self._t_pred * self._fps)

        net_cpu, _ = openpifpaf.network.factory(checkpoint='resnet101')
        net = net_cpu.cuda()
        decode = openpifpaf.decoder.factory_decode(net, seed_threshold=0.5)
        processor = openpifpaf.decoder.Processor(net, decode, instance_threshold=0.2, keypoint_threshold=0.3)

        vidcap = cv2.VideoCapture(clip_file_path)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        kp_OPP = self._get_2dkp_vid(filename, processor)
        nbr_seq_vid = int((len(kp_OPP))/(forecast_step))

        data_seq = {'vid_id': filename,'num_seq': nbr_seq_vid,'forecast_step': forecast_step,
                'nbr_frame_seq': self.nbr_frame_seq, 'total_frame_vid': num_frames, 'width': int(width), 
                'height': int(height), 'per_seq_ped': [], 'seqs': []}

        for idx_seq in range(0, nbr_seq_vid):
            
            data_seq['seqs'].append({'seq_id' : idx_seq, 'frames' : []})
            sequence = kp_OPP[idx_seq*forecast_step:(idx_seq+1)*forecast_step]

            if np.any(sequence):
                for idx_frame, frame in enumerate(sequence):
                    data_seq['seqs'][-1]['frames'].append({'frame_id' : idx_frame, 'peds': []})
                    per_frame_ped = []
                    for idx_ped, kp_ped in enumerate(frame):
                        filtered_keypoints = kp_ped[kp_ped[:, 2] > 0]
                        if np.any(filtered_keypoints):
                            xtl = np.min(filtered_keypoints[:, 0]) # top left x
                            ytl = np.min(filtered_keypoints[:, 1]) # top left y
                            xbr = np.max(filtered_keypoints[:, 0]) # bottom right x
                            ybr = np.max(filtered_keypoints[:, 1]) # bottom right y
                            if per_frame_ped.count(idx_ped) == 0:
                                per_frame_ped.append(idx_ped)
                        else :
                            xtl, ytl, xbr, ybr = 0, 0, 0, 0
                            
                        new_frame_data = {'ped_id': idx_ped,
                                          'occlusion' : 0,
                                          'bbox' : [float(xtl), float(ytl), float(xbr), float(ybr)],
                                          '2dkp' : kp_ped}
                        data_seq['seqs'][-1]['frames'][-1]['peds'].append(new_frame_data)
                    
                if len(per_frame_ped) == 0:
                    data_seq['per_seq_ped'].append(None)
                else:
                    data_seq['per_seq_ped'].append(per_frame_ped)

        vidcap.release()
        
        with open(join(self._infer_pkl_path, filename + "_OPP.pkl"), "wb") as f:
            pickle.dump(data_seq, f, pickle.HIGHEST_PROTOCOL)
        
        print('Dataset sucessfully created')
        print('Path ', self._infer_pkl_path)

        return

    

    def processing_data_4_PIP(self, filename):
        """
        Processes an interference Dataset based on 
        the structure output by generate_data_OPP, and 
        outputs a more convenient structure for the PIP model
        :param filename: Name of the file without the extension

        Dictionary structure:
        'vid_id':           str
        'num_seq':          int
        'forecast_step':    int
        'nbr_frame_seq':    int
        'total_frame_vid':  int
        'width':            int
        'height':           int
        'per_seq_ped':      list (list (int))
        'ped_annotations':      list (dict) 
            'frames':               list (int)
            'occlusion':            list (int)
            'bbox':                 list ([x1 (float), y1 (float), x2 (float), y2 (float)])
            '2dkp':                 list (array(array))
        """

        pkl_file_path = join(self._infer_pkl_path, filename + "_OPP.pkl")

        if not isfile(pkl_file_path):
            print("No pickle file found at " + self._infer_pkl_path)
            return
        
        with open(join(self._infer_pkl_path, pkl_file_path), "rb") as f:
                infer_data = pickle.load(f)

        data_PIP = {'vid_id': infer_data['vid_id'],'num_seq': infer_data['num_seq'],'forecast_step': infer_data['forecast_step'],
                'nbr_frame_seq': infer_data['nbr_frame_seq'], 'total_frame_vid': infer_data['total_frame_vid'], 'width': infer_data['width'], 
                'height': infer_data['height'], 'per_seq_ped': infer_data['per_seq_ped'], 'ped_annotations': []}
        
        
        max_psp = 0
        for psp in enumerate(data_PIP['per_seq_ped']):
            max_psp = max_psp if max_psp > len(psp) else len(psp)

        for ped_id in range(0, max_psp):
            ped_annotations = []
            for seq in infer_data['seqs']:
                frames_tmp = []
                occlusion_tmp = []
                bbox_tmp = []
                kps_tmp = []
                if not data_PIP['per_seq_ped'][seq['seq_id']] is None:
                    if ped_id in data_PIP['per_seq_ped'][seq['seq_id']]:
                        for frame in seq['frames']:
                            frames_tmp.append(self.nbr_frame_seq*seq['seq_id']+frame['frame_id'])
                            if ped_id < len(frame['peds']):
                                occlusion_tmp.append(frame['peds'][ped_id]['occlusion'])
                                bbox_tmp.append(frame['peds'][ped_id]['bbox'])
                                kps_tmp.append(frame['peds'][ped_id]['2dkp'])
                            else:
                                occlusion_tmp.append(0)
                                bbox_tmp.append([0., 0., 0., 0.])
                                kps_tmp.append(np.zeros((17, 3)))
                            
                if frames_tmp:
                    ped_seq = {'frames' : frames_tmp,
                            'occlusion' : occlusion_tmp,
                            'bbox' : bbox_tmp,
                            '2dkp' : np.array(kps_tmp)}
                    ped_annotations.append(ped_seq)
            data_PIP['ped_annotations'].append(ped_annotations)
        
        with open(join(self._infer_pkl_path, filename + "_PIP.pkl"), "wb") as f:
            pickle.dump(data_PIP, f, pickle.HIGHEST_PROTOCOL)

        print('Dataset sucessfully created')
        print('Path ', self._infer_pkl_path)

        return
    


    def reconstrust_video(self, filename):
        """
        Reconstructs the video clip with prediction vizualisation. 
        To this end are used the json prediction file, 
        the pickle Dataset file, and the video clip.
        :param filename: Name of the file without the extension
        """

        pred_file_path = join(self._infer_pred_path, filename + ".json")
        pkl_file_path = join(self._infer_pkl_path, filename + "_PIP.pkl")
        clip_file_path = join(self._infer_clips_path, filename + ".mp4")

        if not isfile(pred_file_path):
            print("No json file found at " + self._infer_folder_path)
            return
        if not isfile(pkl_file_path):
            print("No pickle file found at " + self._infer_folder_path)
            return
        if not isfile(clip_file_path):
            print("No clip file found at " + self._infer_clips_path)
            return
        
        with open(pred_file_path, 'rb') as f:
            pred = json.load(f)
        
        with open(pkl_file_path, 'rb') as f:
            pkl_file = pickle.load(f)

        # Dictionary structure:
        # 'vid_id'(str):      str
        # 'num_seq':          int
        # 'forecast_step':    int
        # 'nbr_frame_seq':    int
        # 'total_frame_vid':  int
        # 'width':            int
        # 'height':           int
        # 'per_seq_ped':      list (list (int))
        # 'ped_annotations':      list (dict) 
        #     'ped_id'(str):          list (dict) 
        #         'frames':               list (int)
        #         'occlusion':            list (int)
        #         'bbox':                 list ([x1 (float), y1 (float), x2 (float), y2 (float)])
        #         '2dkp':                 list (array(array))
        
        width = pkl_file['width']
        height = pkl_file['height']

        # print(len(pkl_file['ped_annotations']))
        # for idx in range(pkl_file['total_frame_vid']):
        #     for ped in pkl_file['ped_annotations']:
        #         seq = int(idx/pkl_file['nbr_frame_seq'])

        vidcap = cv2.VideoCapture(clip_file_path)
        out = cv2.VideoWriter(join(self._infer_out_path, filename+'_out.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame_idx, frame in enumerate(pred["output"]):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, image = vidcap.read()

            for ped in frame["predictions"]:
                if not ped["pred"] is None:
                     bbox = ped["bbox"]
                     conf = ped["confidence"]
                     pred = ped["pred"]

                     legend = "C" if pred else "NC"
                     color = (36,255,12) if pred else (36,12,255)
                     color_conf = (255*(1-conf),255*conf,12) if pred else (255*conf,255*(1-conf),255)
                     
                     cv2.rectangle(image, (int(bbox[0][0]), int(bbox[0][1])),
                                   (int(bbox[1][0]), int(bbox[1][1])), color_conf, 1)
                     cv2.putText(image, legend, (int(bbox[0][0]), int(bbox[0][1])-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            out.write(image)
        out.release()

        print('Video sucessfully created')
        print('Path ', self._infer_out_path)
        return
        
            

if __name__ == "__main__":

    opts = parse_args()
    args = get_config(opts.config)
    infer = Inference(data_path=opts.data_path)
    
    #Generate the inference dataset
    infer.generate_data_OPP(filename=opts.filename)
    infer.processing_data_4_PIP(filename=opts.filename)

    #Perform the inference
    infer_model(opts, args)

    #Reconstruct the video for visualization
    infer.reconstrust_video(filename=opts.filename)
