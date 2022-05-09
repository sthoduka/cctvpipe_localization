import json
import os
import numpy as np
import pdb
import torch
from sklearn import metrics


def main():
    target = 'val_result_full.json'
    with open(target, 'r') as fp:
        data = json.load(fp)

    root = 'video_track'
    ann_file = os.path.join(root, 'cctv_pipe_train.json')
    with open(ann_file, 'r') as fp:
        ann_data = json.load(fp)

    label_scores = {}

    new_res = {}
    new_res['results'] = {}
    tot_objs = 0
    scores = []
    all_labels = []
    for vid in data['results'].keys():
        new_res['results'][vid] = []
        tot_objs += len(data['results'][vid])
        vid_moments = []
        moment_labels = {}
        for gt_obj in ann_data['database'][vid]['annotations']:
            mom = float(gt_obj['moment'])
            vid_moments.append(mom)
            if mom in moment_labels.keys():
                moment_labels[mom].append(gt_obj['label'])
            else:
                moment_labels[mom] = [gt_obj['label']]
            all_labels.append(gt_obj['label'])
        vid_moments = np.array(vid_moments)

        added_defect = False
        current_moment = float(data['results'][vid][0]['moment'])
        prev_obj = None
        for obj in data['results'][vid]:
            res_score = float(obj['score'])
            moment = float(obj['moment'])

            if moment != current_moment:
                nom_score = float(prev_obj['nominal_score'])
                if added_defect:
                    nom_label = 0
                else:
                    nom_label = 1
                # note; we get nom_score from the last vid obj
                if 'nominal' in label_scores.keys():
                    label_scores['nominal'].append([nom_score, nom_label])
                else:
                    label_scores['nominal'] = [[nom_score, nom_label]]
                added_defect = False

            gt_score = 0.0
            if np.min(np.abs(vid_moments - moment)) <= 5:
                lbl_moment = vid_moments[np.argmin(np.abs(vid_moments - moment))]
                labels = moment_labels[lbl_moment]
                if obj['label'] in labels:
                    gt_score = 1.0
            elif np.min(np.abs(vid_moments - moment)) <= 10:
                lbl_moment = vid_moments[np.argmin(np.abs(vid_moments - moment))]
                labels = moment_labels[lbl_moment]
                if obj['label'] in labels:
                    gt_score = 1.0
            elif np.min(np.abs(vid_moments - moment)) <= 20:
                lbl_moment = vid_moments[np.argmin(np.abs(vid_moments - moment))]
                labels = moment_labels[lbl_moment]
                if obj['label'] in labels:
                    gt_score = 1.0
            if obj['label'] in label_scores.keys():
                label_scores[obj['label']].append([res_score, gt_score])
            else:
                label_scores[obj['label']] = [[res_score, gt_score]]

            if gt_score > 0.0:
                added_defect = True

            prev_obj = obj
            current_moment = moment

    label_thresholds = {}
    for key in label_scores.keys():
        output = label_scores[key]
        output = np.array(output)
        target = output[:, 1]
        score = output[:, 0]
        target = target.astype(np.uint8)
        if np.max(target) == 0: # this means we're completely missing this class in the validation set
            label_thresholds[key] = 0.1
            print(key)
            continue
        fpr, tpr, thresholds = metrics.roc_curve(target, score, pos_label=1)
        geometric_mean = np.sqrt(tpr * (1 - fpr))
        best_threshold_idx = np.argmax(geometric_mean)
        best_threshold = thresholds[best_threshold_idx]
        label_thresholds[key] = best_threshold
        print('%s %.3f' % (key, best_threshold))
    with open('thresholds.json', 'w') as fp:
        json.dump(label_thresholds, fp)




if __name__ == "__main__":
    main()
