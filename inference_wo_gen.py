import numpy as np
import argparse
from tqdm import tqdm
import json
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
    parser.add_argument('--config', type=str, default='configs/inference.yaml', help='Path to the config file.')
    parser.add_argument('data_path', type=str, help='Path to the data.')
    opts = parser.parse_args()
    return opts

def infer(args):
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
    jaad_ts = KPInfDataset(data_path=opts.data_path)
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
    with open(args.fname, 'w') as fp:
        json.dump(inference, fp)
    
    print('INFO: Prediction saved in {}'.format(args.fname))
        

if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)
    infer(args)
