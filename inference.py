import numpy as np
import argparse
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.dataset_inference import KPInfDataset
from lib.model.model_action import ActionNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference.yaml', help='Path to the config file.')
    parser.add_argument('-p','--path', default='data/inference.pkl', help='Path to the inference data.')
    opts = parser.parse_args()
    return opts

def infer(args):
    print('INFO: Inference mode')
    
    print('INFO: Loading model')
    m_backbone = load_backbone(args)
    model = ActionNet(backbone=m_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = nn.CrossEntropyLoss()
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

    jaad_ts = KPInfDataset(data_path=args.data_path,is_train=False)
    test_loader = DataLoader(jaad_ts, **testloader_params)
    model.load_state_dict(torch.load(args.chk_path, map_location=lambda storage, loc: storage)['model'], strict=True)

    print('INFO: Inference on {} sequences'.format(len(test_loader)))
    
    model.eval() #put the model in eval mode
    preds = []

    with torch.no_grad():
        for idx, (batch, label) in tqdm(enumerate(test_loader)):
            if torch.cuda.is_available():
                label = label.cuda()
                batch = batch.cuda()
            output = model(batch)
            pred = output.argmax(dim=1)

            #update the metrics
            preds.append(pred)
    
    print('INFO: Inference done')
    
    #now we need to put the predictions back to the right sequence of the right pedestrian
    #we need to know the number of sequences per pedestrian
    prediction = []
    seqs = jaad_ts.get_seqs()
    for i in range(len(seqs)):
        seq = seqs[i]
        prediction.append(preds[:seq])
        preds = preds[seq:]

    num_frames, pedFrames = jaad_ts.get_frames()
    #now we need to put the predictions back to the right frame of the right pedestrian
    output = []
    for i in range(num_frames):
        frame_dict = {'frame' : i, 'predictions' : []}
        for j in range(len(pedFrames)):
            for k in range(len(pedFrames[j])):
                if pedFrames[j][k] == i:
                    frame_dict['predictions'].append(prediction[j][k].item())
                else:
                    frame_dict['predictions'].append(None)
        output.append(frame_dict)

    inference = {'project' : 'Pedestrian Intention Prediction',
                 'output' : output}

    #save the prediction in a json file
    with open('data/inference.json', 'w') as fp:
        json.dump(inference, fp)
    
    print('INFO: Prediction saved in data/prediction.json')
        

if __name__ == '__main__':
    opts = parse_args()
    args = get_config(opts.config)
    infer(args)
