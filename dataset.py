"""
MIT License

Copyright (c) 2018 I. Kotseruba
Copyright (c) 2023 D. Jacquemont, K. Hinard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pickle5 as pickle
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from os.path import join, abspath, exists, dirname, isfile, splitext
import os
import cv2
import PIL
import openpifpaf
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=dirname(abspath('__file__')), help="Path to the folder of the repository")
    parser.add_argument('--compute_kps', help="Compute keypoints", action="store_true")
    parser.add_argument('--regen', help="Regenerate the dataset", action="store_true")
    opts = parser.parse_args()
    return opts

class JAAD(object):
    def __init__(self, data_path = ''):
        """
        Constructor of the jaad class
        :param data_path: Path to the folder of the dataset
        """
        self._year = '2016'
        self._name = 'JAAD'
        self._image_ext = '.png'
        
        self._fps = 30
        self._t_pred = 1
        self._label_frames = 30

        # Paths
        self._data_path = data_path if data_path else dirname(abspath('__file__'))
        assert exists(self._data_path), \
            'Jaad path does not exist: {}'.format(self._data_path)
        self._annotation_path = join(self._data_path, 'datagen/JAAD_DS/annotations')
        self._inference_path = join(self._data_path, 'datagen/JAAD_DS/Inference')
        self._videos_path = join(self._data_path, 'datagen/JAAD_DS/JAAD_clips')
        self._checkpoint_path = join(self._data_path, 'datagen/data/jaad_dataset.pkl')



    def _get_video_ids(self):
        """
        Returns a list of all video ids
        :return: The list of video ids
        """
        return [vid.split('.')[0] for vid in os.listdir(self._annotation_path)]



    # Annotation processing helpers
    def _map_text_to_scalar(self, label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'occlusion': {'none': 0, 'part': 1, 'full': 2},
                   'action': {'standing': 0, 'walking': 1},
                   'nod': {'__undefined__': 0, 'nodding': 1},
                   'look': {'not-looking': 0, 'looking': 1},
                   'hand_gesture': {'__undefined__': 0, 'greet': 1, 'yield': 2,
                                    'rightofway': 3, 'other': 4},
                   'reaction': {'__undefined__': 0, 'clear_path': 1, 'speed_up': 2,
                                'slow_down': 3},
                   'cross': {'not-crossing': 0, 'crossing': 1, 'irrelevant': -1},
                   'age': {'child': 0, 'young': 1, 'adult': 2, 'senior': 3},
                   'designated': {'ND': 0, 'D': 1},
                   'gender': {'n/a': 0, 'female': 1, 'male': 2},
                   'intersection': {'no': 0, 'yes': 1},
                   'motion_direction': {'n/a': 0, 'LAT': 1, 'LONG': 2},
                   'traffic_direction': {'OW': 0, 'TW': 1},
                   'signalized': {'n/a': 0, 'NS': 1, 'S': 2},
                   'vehicle': {'stopped': 0, 'moving_slow': 1, 'moving_fast': 2,
                               'decelerating': 3, 'accelerating': 4},
                   'road_type': {'street': 0, 'parking_lot': 1, 'garage': 2},
                   'traffic_light': {'n/a': 0, 'red': 1, 'green': 2}}
        return map_dic[label_type][value]



    def get_stats(self):
        """
        Outputs the statistics of the dataset
        """
        with open(self._checkpoint_path, 'rb') as f:
              database = pickle.load(f)

        tot_seq, tot_C, tot_NC = 0, 0, 0

        for vid in list(database['annotations'].keys()):
            print('\n-----------------------------------------------')
            print('Video ', vid)

            annotations = database['annotations'][vid]['ped_annotations']

            for ped in list(annotations.keys()):
                tot_seq += len(annotations[ped])

                C, NC = 0, 0
                for info in annotations[ped]:
                    if info['cross']:
                        C += 1
                    else:
                        NC += 1
                
                print('Pedestrian {:10s}    nbr seq {:3}'.format(ped, 
                                        str(len(annotations[ped]))), end ='')
                print('    C {:1}    NC {:1}'.format(str(C), str(NC)))
                tot_C += C
                tot_NC += NC

        print('\n----------------------END----------------------')
        print('Total number sequences      ', str(tot_seq))
        print('Total number train videos   ', str(len(database['split']['train_ID'])))
        print('Total number test videos    ', str(len(database['split']['test_ID'])))
        print('Total number C labels    {:5}   {:3}%'.format(str(tot_C), 
                                          str(int(100*tot_C/(tot_C+tot_NC)))))
        print('Total number NC labels   {:5}   {:3}%'.format(str(tot_NC), 
                                          str(int(100*tot_NC/(tot_C+tot_NC)))))
        return



    def get_sample(self, vid = 'video_0007', ped = 0, seq = 0):
        """
        Create vizualisation for a sample of the dataset
        The function creates a video JAAD_DS/sample_DS.mp4
        :param vid: The video id
        :param ped: The pedestrian id
        :param seq: The sequence number in the video
        """

        if not isfile(self._checkpoint_path):
            print("No database found in " + self._data_path)
            return

        with open(self._checkpoint_path, 'rb') as f:
            database = pickle.load(f)

        annotations = database['annotations'][vid]['ped_annotations']
        bboxes = annotations[ped][seq]['bbox'].astype(int)
        kps = annotations[ped][seq]['2dkp']
        frames = annotations[ped][seq]['frames']
        cross = annotations[ped][seq]['cross']

        f = join(self._videos_path, vid + '.mp4')
        vidcap = cv2.VideoCapture(f)

        out = cv2.VideoWriter(join(self._data_path,'datagen/JAAD_DS/sample_DS.mp4'),
                              cv2.VideoWriter_fourcc(*'DIVX'),15, (1920, 1080))

        for idx in range(self._fps*self._t_pred-1):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frames[idx])
            success, image = vidcap.read()

            legend = "C" if cross else "NC"
            cv2.rectangle(image, (bboxes[idx][0], bboxes[idx][1]), 
                (bboxes[idx][2], bboxes[idx][3]), (36,255,12), 1)
            cv2.putText(image, legend, (bboxes[idx][0], bboxes[idx][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            for kp in kps[idx]:
                cv2.circle(image, (int(kp[0]),int(kp[1])), int(4*kp[2]), 
                 (0,0,255), -1)

            out.write(image)

        out.release()

        print('Sample video sucessfully created')
        print('Path ', join(self._data_path,'datagen/JAAD_DS/samples/sample_DS.mp4'))
        return



    def _get_2dkp_seq(self, kp_seq, bbox_ped):
        """
        Finds the right 2d keypoints for a given pedestrian by matching 
        the 2d mean of the bounding box of a pedestrian with the 
        2d mean of the keypoints
        :param kp_seq: An array of list of 2d keypoints for a sequence
        :param bbox_ped: A list of bounding boxes for a sequence
        :return: a list of 2d keypoints for each sequence
        """

        output_kp_seq = []

        for frame_nbr, bbox in enumerate(bbox_ped):
            max_valid_dist = (abs(bbox[0] - bbox[2]) + abs(bbox[1] - bbox[3]))/2

            x_mean_bbox = (bbox[0] + bbox[2])/2
            y_mean_bbox = (bbox[1] + bbox[3])/2

            for ped, kp_ped in enumerate(kp_seq[frame_nbr]):

                x_mean_kp, y_mean_kp = 0, 0
                nbr_valid_kp = 0
                for kp in kp_ped:
                    if kp[2] != 0:
                        x_mean_kp += kp[0]
                        y_mean_kp += kp[1]
                        nbr_valid_kp += 1

                if nbr_valid_kp:
                    x_mean_kp /= nbr_valid_kp
                    y_mean_kp /= nbr_valid_kp

                    if np.linalg.norm([x_mean_bbox - x_mean_kp, 
                                       y_mean_bbox - y_mean_kp]) < max_valid_dist:
                        output_kp_seq.append(kp_ped)
                        break
                    else:
                        if ped == len(kp_seq[frame_nbr]) - 1:
                            output_kp_seq.append(np.zeros((17, 3)))
                else:
                    if ped == len(kp_seq[frame_nbr]) - 1:
                        output_kp_seq.append(np.zeros((17, 3)))
        return np.array(output_kp_seq)
    


    def _get_2dkp_vid(self, vid, processor):
        """
        Extracts the 2d keypoints of each pedestrian in the whole 
        video with OpenPifPaf
        :param vid: The video id
        :param processor: The processor used to get the 2d keypoints
        :return: an array of list of 2d keypoints for a sequence
        """

        f = join(self._videos_path, vid + '.mp4')
        vidcap = cv2.VideoCapture(f)
        success, image = vidcap.read()

        pred_data = []
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
                  pred_data.append([np.zeros((17, 3))])
              else:
                  tmp_2dkp = []
                  for pred in predictions:
                      tmp_2dkp.append(pred.data)
                  pred_data.append(tmp_2dkp)
        return np.array(pred_data, dtype=object)



    def _get_annotations(self, vid, compute_kps, processor = None):
        """
        Generates a dictinary of annotations by parsing the video XML file 
        and generating 2d keypoints for the all pedestrians in the whole 
        video (if compute_kps = True)
        :param vid: The id of video to parse
        :param processor: The processor used to get the 2d keypoints
        :return: A dictionary of annotations, and the number of sequences per video
        """

        forecast_step = int(self._t_pred * self._fps / 2)

        path_to_file = join(self._annotation_path, vid + '.xml')
        tree = ET.parse(path_to_file)
        ped_annt = 'ped_annotations'

        annotations = {}
        annotations['num_frames'] = int(tree.find("./meta/task/size").text)
        annotations['width'] = int(tree.find("./meta/task/original_size/width").text)
        annotations['height'] = int(tree.find("./meta/task/original_size/height").text)
        annotations[ped_annt] = {}

        nbr_seq_vid = 0
        ped_tracks = tree.findall("./track")
        pred_data = None

        for t in ped_tracks:    

            boxes = t.findall('./box')
            new_id = boxes[0].find('./attribute[@name=\"id\"]').text
                  
            if 'b' in new_id:

                old_id = boxes[0].find('./attribute[@name=\"old_id\"]').text

                tmp_bbox = []
                tmp_occ = []
                tmp_cross = []
                tmp_frames = []

                if 'pedestrian' in old_id:
                    # Gathering annotations from xml files
                    for b in boxes:
                        tmp_bbox.append(
                            [float(b.get('xtl')), float(b.get('ytl')), float(b.get('xbr')), float(b.get('ybr'))])
                        tmp_occ.append(self._map_text_to_scalar('occlusion',
                                                                b.find('./attribute[@name=\"occlusion\"]').text))
                        tmp_frames.append(int(b.get('frame')))
                        tmp_cross.append(self._map_text_to_scalar('cross', b.find('./attribute[@name=\"cross\"]').text))

                    # Creating 2d KP
                    if pred_data is None and compute_kps:
                        pred_data = self._get_2dkp_vid(vid, processor)

                    # Dividing DS in multiple 2s sequences
                    nbr_seq_ped = int((len(tmp_bbox) - forecast_step - 1)/(forecast_step))
                    if nbr_seq_ped >= 1 :

                        annotations[ped_annt][new_id] = []
                        nbr_seq_vid += nbr_seq_ped

                        for i in range(0, nbr_seq_ped):

                            annotation_occ = tmp_occ[i*forecast_step:(i+2)*forecast_step]
                            annotation_frames = tmp_frames[i*forecast_step:(i+2)*forecast_step]
                            
                            annotation_bbox = np.array(tmp_bbox[i*forecast_step:(i+2)*forecast_step])

                            end_idx_cross = min(len(tmp_cross[(i+2)*forecast_step:]), self._label_frames)
                            annotation_cross = np.amax(np.array(tmp_cross[(i+2)*forecast_step:(i+2)*forecast_step+end_idx_cross]))

                            if compute_kps : 
                                annotations_2dkp = self._get_2dkp_seq(pred_data[annotation_frames], annotation_bbox)
                                annotations_dict = {'old_id': old_id, 'frames': annotation_frames,
                                                'bbox': annotation_bbox, '2dkp': annotations_2dkp, 'occlusion': annotation_occ, 'cross': annotation_cross}
                            else:
                                annotations_dict = {'old_id': old_id, 'frames': annotation_frames,
                                                'bbox': annotation_bbox, 'occlusion': annotation_occ, 'cross': annotation_cross}

                            annotations[ped_annt][new_id].append(annotations_dict)
        return annotations, nbr_seq_vid


    
    def generate_dataset(self, compute_kps, regenerate):
        """
        Generate a dataset based on JAAD database
        :param compute_kps: If True, 2d keypoints are computed
        :param regenerate: If True, the dataset is regenerated
        
        Dictionary structure:
        'annotations': 
            'vid_id'(str): 
                'num_frames':               int
                'width':                    int
                'height':                   int
                'ped_annotations'(str):     list (dict)
                    'ped_id'(str):              list (dict) {
                        'old_id':                   str
                        'frames':                   list (int)
                        'occlusion':                list (int)
                        'bbox':                     list ([x1 (float), y1 (float), x2 (float), y2 (float)])
                        '2dkp':                     list (array(array))
                        'cross':                    list (int)
        'split': 
            'train_ID':     list (str)
            'test_ID':      list (str)
        'ckpt': str
        'seq_per_vid': list (int)
        """
        if compute_kps:
            net_cpu, _ = openpifpaf.network.factory(checkpoint='resnet101')
            net = net_cpu.cuda()
            decode = openpifpaf.decoder.factory_decode(net, seed_threshold=0.5)
            processor = openpifpaf.decoder.Processor(net, decode, instance_threshold=0.2, keypoint_threshold=0.3)

        video_ids = sorted(self._get_video_ids())

        if os.path.isfile(self._checkpoint_path) and not regenerate:
            with open(self._checkpoint_path, 'rb') as f:
                database = pickle.load(f)
            database_vid_ID = database['split']['train_ID']+database['split']['test_ID']
            nbr_seq_vid_ID = database['seq_per_vid']

            next_vid = database['ckpt'][:-3] + str(int(database['ckpt'][-3:])+1).zfill(3)

            if next_vid == 'video_0347':
                print("\nDataset already created")
                raise SystemExit

            video_ids = video_ids[video_ids.index(next_vid):]
            print("Resuming database generation from ", next_vid)
        else:
            database = {'ckpt': None,'seq_per_vid': [],'split': {'train_ID':[], 'test_ID': []}, 'annotations': {}}
            database_vid_ID, nbr_seq_vid_ID = [], []
            print("Generating database for jaad\n")

        for vid in video_ids:

            print('\nGetting annotations for %s' % vid)
            database['ckpt'] = vid

            processor = processor if compute_kps else None
            vid_annotations, nbr_seq = self._get_annotations(vid, processor, compute_kps)

            if (nbr_seq != 0):
                database['annotations'][vid] = vid_annotations
                database_vid_ID.append(vid)
                nbr_seq_vid_ID.append(nbr_seq)
            
            if compute_kps or vid == video_ids[-1]:

                # Creating testset/trainset
                cumsum = np.cumsum(nbr_seq_vid_ID)/sum(nbr_seq_vid_ID)
                database['seq_per_vid'] = nbr_seq_vid_ID
                res = next(x for x, val in enumerate(cumsum) if val > 0.2)
                database['split']['train_ID'] = database_vid_ID[res:]
                database['split']['test_ID'] = database_vid_ID[:res]

                with open(self._checkpoint_path, 'wb') as f:
                    pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)

                if vid == video_ids[-1]:
                  print('\nDatabase written to {}'.format(self._checkpoint_path))
                else:
                  print('\nCheckpoint saved to {}'.format(self._checkpoint_path))
        return 



if __name__ == "__main__":

    opts = parse_args()
    DS = JAAD(data_path=opts.data_path)
    DS.generate_dataset(opts.compute_kps, opts.regen)