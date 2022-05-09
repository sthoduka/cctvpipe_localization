import os
import random
import glob
import json
import numpy as np
#np.random.seed(1)
import math
import cv2
import pdb
import yaml

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
#torch.manual_seed(17)

def load_data_cctv(root, group):
    ann_file = os.path.join(root, 'cctv_pipe_train.json')
    train_file = os.path.join(root, 'train_keys_full.json')
    val_file = os.path.join(root, 'val_keys_full.json')
    test_file = os.path.join(root, 'test_keys.json')
    with open(ann_file, 'r') as fp:
        data = json.load(fp)
    with open(train_file, 'r') as fp:
        train_keys = json.load(fp)
    with open(val_file, 'r') as fp:
        val_keys = json.load(fp)
    with open(test_file, 'r') as fp:
        test_keys = json.load(fp)

    if group == 5:
        train_keys.extend(val_keys)

    # for test and validation,sample frames at 1 FPS
    test_frame_lengths = []
    total_test_frame_length = 0
    test_frame_ids = {}
    for key in test_keys:
        duration = float(data['database'][key]['duration'])
        fps = float(data['database'][key]['fps'])
        num_frames = int(fps * (duration -1.0))
        test_frame_ids[key] = np.arange(0, num_frames, int(fps))
        total_test_frame_length += len(test_frame_ids[key])
        test_frame_lengths.append(len(test_frame_ids[key]))

    val_frame_lengths = []
    total_val_frame_length = 0
    val_frame_ids = {}
    for key in val_keys:
        duration = float(data['database'][key]['duration'])
        fps = float(data['database'][key]['fps'])
        num_frames = int(fps * (duration -1.0))
        val_frame_ids[key] = np.arange(0, num_frames, int(fps))
        total_val_frame_length += len(val_frame_ids[key])
        val_frame_lengths.append(len(val_frame_ids[key]))

    # for training, sample frames based on proximity to annotations
    train_frame_lengths = []
    total_train_frame_length = 0
    train_frame_ids = {}
    for key in train_keys:
        duration = float(data['database'][key]['duration'])
        fps = float(data['database'][key]['fps'])
        annotations = data['database'][key]['annotations']
        all_selected_frame_ids = []
        for ann in annotations:
            moment = float(ann['moment'])
            frame_id = int(fps * moment)
            # Plus/minus 2.5 seconds
            window = int(fps * 2.5)
            rate = int(fps / 10)
            tot_frames = 15
            if frame_id + window < num_frames:
                frame_ids = np.arange(frame_id, frame_id + window, rate).tolist()
                while frame_ids[-1] >= num_frames-1:
                    frame_ids = frame_ids[:-1]
            else:
                frame_ids = [frame_id]
            if frame_id - window > 0:
                frame_ids.extend(np.arange(frame_id - window, frame_id, rate).tolist())
            frame_ids = np.array(frame_ids)
            if len(frame_ids) < tot_frames:
                frame_ids = sorted(frame_ids[np.random.choice(len(frame_ids), size=tot_frames, replace=True)].tolist())
            else:
                frame_ids = sorted(frame_ids[np.random.choice(len(frame_ids), size=tot_frames, replace=False)].tolist())
            all_selected_frame_ids.extend(frame_ids)

        # only add nominal frames for group > 1 (default)
        if group != 1:
            defect_len = int(len(all_selected_frame_ids) / 2.0)
            num_frames = int(fps * (duration - 1.0))
            all_frames = np.arange(0, num_frames, 1)
            selected_other_frames = []
            if defect_len > len(all_frames):
                selected_other_frames = all_frames[np.random.choice(len(all_frames), size=defect_len, replace=True)]
            else:
                selected_other_frames = all_frames[np.random.choice(len(all_frames), size=defect_len, replace=False)]
            all_selected_frame_ids.extend(selected_other_frames)

        train_frame_ids[key] = sorted(all_selected_frame_ids)
        total_train_frame_length += len(train_frame_ids[key])
        train_frame_lengths.append(len(train_frame_ids[key]))

    all_labels = []
    for key in train_keys:
        if data['database'][key]['subset'] == 'training':
            annotations = data['database'][key]['annotations']
            for ann in annotations:
                all_labels.append(ann['label'])
    all_labels = np.unique(all_labels)
    label_to_idx = dict([lbl, idx] for idx, lbl in enumerate(all_labels))
    idx_to_label = dict([idx, lbl] for idx, lbl in enumerate(all_labels))
    # TODO: return a dictionary instead of this
    return data['database'], train_keys, val_keys, test_keys, label_to_idx, idx_to_label, total_test_frame_length, test_frame_lengths, test_frame_ids, total_val_frame_length, val_frame_lengths, val_frame_ids, total_train_frame_length, train_frame_lengths, train_frame_ids



def clip_loader_test(path, frame_id, vid_data, label_to_idx):
    cap = cv2.VideoCapture(path)
    if frame_id != 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    if not success or frame is None:
        print('could not read %d from %s' % (frame_id, path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0
        success, frame = cap.read()
    if frame is None:
        print('could not read %d from %s' % (frame_id, path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0
        success, frame = cap.read()
    frame = frame[:, :, ::-1].copy()
    cap.release()

    if len(vid_data['annotations']) > 0:
        defect_moments = []
        moment_labels = {}
        for ann in vid_data['annotations']:
            moment = int(ann['moment'])
            defect_moments.append(int(ann['moment']))
            lbl = label_to_idx[ann['label']]
            if moment in moment_labels.keys():
                moment_labels[moment].append(lbl)
            else:
                moment_labels[moment] = [lbl]

        defect_moments = np.array(defect_moments)

        moment = frame_id / float(vid_data['fps'])

        closest_dist = (np.abs(defect_moments - moment)).min()
        closest_dist_arg = (np.abs(defect_moments - moment)).argmin()
        if closest_dist > 5: # be at least 5 seconds away from a defect
            label = 16
        else:
            label = moment_labels[defect_moments[closest_dist_arg]]
    else:
        label = 16
    return frame, label


class CCTVPipeDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_type='train', training_type='single_img', group=4, transform = None, evaluation=False):
        self.root = root
        self.data, train_keys, val_keys, test_keys, self.label_to_idx, self.idx_to_label, self.total_test_frame_length, self.test_frame_lengths, self.test_frame_ids, self.total_val_frame_length, self.val_frame_lengths, self.val_frame_ids, self.total_train_frame_length, self.train_frame_lengths, self.train_frame_ids = load_data_cctv(root, group)
        self.test_frame_cum_sum = np.cumsum(self.test_frame_lengths)
        self.val_frame_cum_sum = np.cumsum(self.val_frame_lengths)
        self.train_frame_cum_sum = np.cumsum(self.train_frame_lengths)
        if dataset_type == 'train':
            self.keys = train_keys
        elif dataset_type == 'val':
            self.keys = val_keys
        else:
            self.keys = test_keys

        if len(self.keys) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.evaluation = evaluation
        self.transform = transform
        self.dataset_type = dataset_type
        self.training_type = training_type # multiclass or binary (unused)
        self.num_classes = 17 # 16 defect classes + 1 nominal class

    def __getitem__(self, index):
        if self.dataset_type == 'test':
            video_id = np.argmax(self.test_frame_cum_sum > index)
            if video_id != 0:
                frame_id = index - self.test_frame_cum_sum[video_id-1]
            else:
                frame_id = index
            video_key = self.keys[video_id]
            video_frame_ids = self.test_frame_ids[video_key]
            video_frame_id = video_frame_ids[frame_id]
        elif self.dataset_type == 'val':
            video_id = np.argmax(self.val_frame_cum_sum > index)
            if video_id != 0:
                frame_id = index - self.val_frame_cum_sum[video_id-1]
            else:
                frame_id = index
            video_key = self.keys[video_id]
            video_frame_ids = self.val_frame_ids[video_key]
            video_frame_id = video_frame_ids[frame_id]
        elif self.dataset_type == 'train':
            video_id = np.argmax(self.train_frame_cum_sum > index)
            if video_id != 0:
                frame_id = index - self.train_frame_cum_sum[video_id-1]
            else:
                frame_id = index
            video_key = self.keys[video_id]
            video_frame_ids = self.train_frame_ids[video_key]
            video_frame_id = video_frame_ids[frame_id]
        else:
            video_key = self.keys[index]

        vid_data = self.data[video_key]
        video_name = vid_data['file_name']

        moment = video_frame_id / float(vid_data['fps'])

        video_file = os.path.join(self.root, 'track2_raw_video', video_name)

        if not os.path.exists(video_file):
            print('path %s does not exist' % video_file)
        img, cls_label = clip_loader_test(video_file, video_frame_id, vid_data, self.label_to_idx)
        if self.transform is not None:
            clip = torch.from_numpy(img)
            clip = clip.permute((2, 0, 1)).contiguous()
            clip = clip.to(dtype=torch.get_default_dtype()).div(255)
            clip = self.transform(clip)

        labels = torch.tensor(cls_label)
        labels = torch.zeros(self.num_classes).scatter_(0, labels, 1.)
        all_labels = labels

        if self.dataset_type == 'train':
            return clip, all_labels, video_key
        else:
            return clip, all_labels, video_key, moment

    def __len__(self):
        if self.dataset_type == 'train':
            return self.total_train_frame_length
        elif self.dataset_type == 'val':
            return self.total_val_frame_length
        else:
            return self.total_test_frame_length

def main():
    ### only used for testing the CCTVPipeDataset class
    transform = transforms.Compose(
        [
            transforms.Resize(300),
            transforms.RandomCrop(300),
            transforms.RandomAdjustSharpness(1.5),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
        ]
    )
    dataset_type = 'train'
    dataset = CCTVPipeDataset('video_track', dataset_type=dataset_type, training_type='single_img', group=4, transform=transform, evaluation=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=True, num_workers=4, pin_memory=False)
    if dataset_type == 'train':
        clip, labels, name = dataset[10]
    else:
        clip, labels, name, moment = dataset[9900]
    pdb.set_trace()

if __name__ == '__main__':
    main()
