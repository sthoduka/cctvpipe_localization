import json
import os
import numpy as np
import pdb
import torch


def main():
    target = 'test_result_full.json'
    label_thresholds_f = 'thresholds.json'

    with open(target, 'r') as fp:
        data = json.load(fp)

    with open(label_thresholds_f, 'r') as fp:
        thresholds = json.load(fp)

    new_res = {}
    new_res['results'] = {}
    tot_objs = 0
    scores = []
    removed_nom = 0
    for vid in data['results'].keys():
        new_res['results'][vid] = []
        tot_objs += len(data['results'][vid])
        for obj in data['results'][vid]:
            score = float(obj['score'])
            nominal_score = float(obj['nominal_score'])
            if score < thresholds[obj['label']]:
                pass
            elif nominal_score > 0.285: # we adjust the nominal threshold slightly for different submissions
                removed_nom += 1
            else:
                new_obj = {'label' : obj['label'], 'score' : obj['score'], 'moment' : obj['moment']}
                new_res['results'][vid].append(new_obj)

    with open('test_result.json', 'w') as fp:
        json.dump(new_res, fp)


if __name__ == "__main__":
    main()
